import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_qkv_state_dict(state_dict):
    """
    Remap a checkpoint saved from the original SparseWindowAttention (separate
    `query`/`key`/`value` Linear layers) onto the fused single-`qkv`-Linear
    layout used here now. This is purely a reorganization of the same trained
    weights -- concatenating the existing query/key/value weight and bias
    tensors along the output dimension, in the same order `SparseWindowAttention.
    forward` splits them back apart (`qkv.chunk(3, dim=-1)` expects
    [query_features, key_features, value_features]) -- so it produces
    bit-exact results versus the original three-linear-layer computation.
    Returns a new dict; does not mutate the one passed in. Keys that don't
    match the old query/key/value pattern (e.g. everything outside
    SparseWindowAttention) are passed through untouched.
    """
    state_dict = dict(state_dict)
    query_keys = [k for k in state_dict if k.endswith('.query.weight')]
    for qk in query_keys:
        prefix = qk[: -len('.query.weight')]
        kk, vk = f'{prefix}.key.weight', f'{prefix}.value.weight'
        qb, kb, vb = f'{prefix}.query.bias', f'{prefix}.key.bias', f'{prefix}.value.bias'
        if kk not in state_dict or vk not in state_dict:
            continue  # not actually a query/key/value attention triple
        state_dict[f'{prefix}.qkv.weight'] = torch.cat(
            [state_dict.pop(qk), state_dict.pop(kk), state_dict.pop(vk)], dim=0)
        if qb in state_dict and kb in state_dict and vb in state_dict:
            state_dict[f'{prefix}.qkv.bias'] = torch.cat(
                [state_dict.pop(qb), state_dict.pop(kb), state_dict.pop(vb)], dim=0)
    return state_dict

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

    def forward(self, x, b, output_size, chunk_size=None):
        f_h = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        # x is (b*t, c, h, w): unfold+embed is fully independent per item on
        # this leading dim (no cross-frame or cross-batch mixing), so this
        # can be chunked over it with bit-exact output regardless of
        # chunk_size -- the raw unfold tensor this produces is ~12x larger
        # than its embedded (hidden-dim) output, making it one of the
        # single largest transient tensors in the whole model.
        if chunk_size is None or chunk_size >= x.size(0):
            feat = self.t2t(x)
            feat = feat.permute(0, 2, 1)
            # feat shape [b*t, num_vec, ks*ks*c]
            feat = self.embedding(feat)
        else:
            feat_chunks = []
            for i in range(0, x.size(0), chunk_size):
                sub = self.t2t(x[i:i + chunk_size])
                sub = sub.permute(0, 2, 1)
                feat_chunks.append(self.embedding(sub))
            feat = torch.cat(feat_chunks, dim=0)
            del feat_chunks
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def _forward_one(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat

    def forward(self, x, t, output_size, chunk_size=None):
        # x is (b_, t, f_h, f_w, c_): embed+fold is independent per frame,
        # same reasoning as SoftSplit above, EXCEPT the internal reshape
        # here assumes each batch item's frames are laid out contiguously,
        # so this chunked path is only used when b_==1 (always true in
        # this pipeline, which processes one video at a time) -- falls
        # back to the original one-shot computation otherwise.
        b_ = x.size(0)
        if chunk_size is None or chunk_size >= t or b_ != 1:
            return self._forward_one(x, t, output_size)
        feat_chunks = []
        for i in range(0, t, chunk_size):
            sub_x = x[:, i:i + chunk_size]
            feat_chunks.append(self._forward_one(sub_x, sub_x.size(1), output_size))
        feat = torch.cat(feat_chunks, dim=0)
        del feat_chunks
        return feat


class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1960, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        assert t2t_params is not None
        self.t2t_params = t2t_params
        self.kernel_shape = reduce((lambda x, y: x * y), t2t_params['kernel_size']) # 49
        # The fold "normalizer" computed in forward() below depends only on
        # (b, n_vecs, output_size, dtype, device) -- never on the actual
        # feature values -- yet this module runs once per transformer depth
        # (8x per model() call) with the exact same shape every single time
        # within a run (same b=1, same output_size, same frame count per
        # window). Recomputing an all-ones fold from scratch 8x per call is
        # pure waste, so it's cached here keyed by the values it actually
        # depends on. Bit-exact: identical computation, just done once and
        # reused instead of repeated.
        self._normalizer_cache = {}

    def _get_normalizer(self, b_fold, n_vecs, output_size, dtype, device):
        # b_fold is the effective fold batch size (original code reshapes
        # (b, n, kernel_shape) to (-1, n_vecs, kernel_shape) before folding,
        # i.e. b_fold = b * n // n_vecs -- one "batch" item per frame, since
        # n = T * n_vecs when x holds T frames' worth of tokens).
        key = (b_fold, n_vecs, output_size, dtype, device)
        normalizer = self._normalizer_cache.get(key)
        if normalizer is None:
            ones = torch.ones(b_fold, n_vecs, self.kernel_shape, dtype=dtype, device=device)
            normalizer = F.fold(ones.permute(0, 2, 1),
                                output_size=output_size,
                                kernel_size=self.t2t_params['kernel_size'],
                                padding=self.t2t_params['padding'],
                                stride=self.t2t_params['stride'])
            self._normalizer_cache[key] = normalizer
        return normalizer

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = self._get_normalizer(b * n // n_vecs, n_vecs, output_size, x.dtype, x.device)

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        x = self.fc2(x)
        return x


def window_partition(x, window_size, n_head):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, n_head, T, window_size, window_size, C//n_head)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1], window_size[1], n_head, C//n_head)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows

class SparseWindowAttention(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0., 
                pooling_token=True):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        # Fused query/key/value projection: one Linear(dim, 3*dim) instead of
        # three separate Linear(dim, dim) calls. Mathematically identical to
        # three separate projections (the weights are just concatenated along
        # the output dimension) but issues one larger matmul instead of three
        # smaller ones, which GPUs generally execute more efficiently. Output
        # of self.qkv(x) is ordered [query_features, key_features,
        # value_features] along the last dim so `.chunk(3, dim=-1)` recovers
        # q, k, v in that order. A checkpoint trained with the original
        # separate query/key/value layers is remapped onto this layout by
        # `fuse_qkv_state_dict` (see below) before loading.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=ks, stride=stride, padding=(0, 0), groups=dim)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        # self.expand_size = tuple(i // 2 for i in window_size)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))


    def forward(self, x, mask=None, T_ind=None, attn_mask=None):
        b, t, h, w, c = x.shape # 20 36
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.n_head
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0] # 20
        new_w = n_ww * self.window_size[1] # 36
        pad_r = new_w - w
        pad_b = new_h - h
        # reverse order
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 
            mask = F.pad(mask,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        del qkv
        win_q = window_partition(q.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_k = window_partition(k.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_v = window_partition(v.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        del q

        # pool_k and pool_v -- computed here (moved ahead of the roll section
        # below) since this only depends on `x`, not on anything the roll
        # section produces. This lets `x` -- a full (b,t,h,w,c) tensor -- be
        # freed before the roll section starts instead of staying alive
        # for the rest of the method. The final concatenation order into
        # win_k/win_v (base, then rolled, then pooled) is unchanged, so
        # this is bit-exact with the original -- purely a reordering of
        # *when* things are computed and freed, not *what* is computed.
        if self.pooling_token:
            pool_x = self.pool_layer(x.view(b*t, new_h, new_w, c).permute(0,3,1,2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0,2,3,1).view(b, t, p_h, p_w, c)
            # Only key/value are needed for the pooled tokens (they're never
            # used as queries), so slice the fused qkv weight/bias down to
            # just the key+value rows instead of running the full qkv
            # projection (which would waste compute producing an unused
            # pooled query) or calling two separate linears. Bit-exact with
            # calling self.key(pool_x) and self.value(pool_x) separately,
            # since those are literally the same weight rows.
            kv_weight = self.qkv.weight[self.dim:]
            kv_bias = self.qkv.bias[self.dim:] if self.qkv.bias is not None else None
            pool_kv = F.linear(pool_x, kv_weight, kv_bias)
            pool_k, pool_v = pool_kv.chunk(2, dim=-1)
            del pool_kv
            # broadcast (not copy) across windows: every window uses the same
            # pooled global tokens, so `.expand()` here is a zero-cost view
            # instead of `.repeat()`'s immediate n_wh*n_ww-fold physical copy.
            # The `.contiguous()` calls below still materialize the final
            # layout exactly once (unavoidable given the reshape), but this
            # way there is only ever one physical copy in memory instead of
            # two (repeat's copy, then contiguous's copy).
            pool_k = pool_k.unsqueeze(1).expand(-1, n_wh*n_ww, -1, -1, -1, -1) # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_k = pool_k.reshape(b, n_wh*n_ww, t, p_h, p_w, self.n_head, c_head).permute(0,1,5,2,3,4,6)
            pool_k = pool_k.contiguous().view(b, n_wh*n_ww, self.n_head, t, p_h*p_w, c_head)
            pool_v = pool_v.unsqueeze(1).expand(-1, n_wh*n_ww, -1, -1, -1, -1) # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_v = pool_v.reshape(b, n_wh*n_ww, t, p_h, p_w, self.n_head, c_head).permute(0,1,5,2,3,4,6)
            pool_v = pool_v.contiguous().view(b, n_wh*n_ww, self.n_head, t, p_h*p_w, c_head)
            del pool_x
        del x

        # roll_k and roll_v
        if any(i > 0 for i in self.expand_size):
            # Process one of the four (top-left/top-right/bottom-left/
            # bottom-right) directions at a time, instead of materializing
            # all four full-size (b,t,h,w,c) rolled copies of k AND v
            # simultaneously (8 full-size tensors alive at once in the
            # original code, on top of k, v, and x themselves). Each
            # rolled tensor is windowed immediately and discarded before
            # the next direction starts, so at most one rolled-k and one
            # rolled-v (much smaller once windowed) are alive together.
            # Bit-exact: identical operations in the identical order, only
            # the memory lifetime of each intermediate changes.
            shifts = (
                (-self.expand_size[0], -self.expand_size[1]),  # top-left
                (-self.expand_size[0], self.expand_size[1]),   # top-right
                (self.expand_size[0], -self.expand_size[1]),   # bottom-left
                (self.expand_size[0], self.expand_size[1]),    # bottom-right
            )
            k_windows_list = []
            v_windows_list = []
            for shift in shifts:
                k_rolled = torch.roll(k, shifts=shift, dims=(2, 3))
                k_windows_list.append(
                    window_partition(k_rolled, self.window_size, self.n_head).view(
                        b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head))
                del k_rolled
                v_rolled = torch.roll(v, shifts=shift, dims=(2, 3))
                v_windows_list.append(
                    window_partition(v_rolled, self.window_size, self.n_head).view(
                        b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head))
                del v_rolled

            rool_k = torch.cat(k_windows_list, 4).contiguous()
            rool_v = torch.cat(v_windows_list, 4).contiguous() # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            del k_windows_list, v_windows_list
            # mask out tokens in current window
            rool_k = rool_k[:, :, :, :, self.valid_ind_rolled]
            rool_v = rool_v[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_k.shape[4]
            rool_k = rool_k.view(b, n_wh*n_ww, self.n_head, t, roll_N, c // self.n_head)
            rool_v = rool_v.view(b, n_wh*n_ww, self.n_head, t, roll_N, c // self.n_head)
            win_k = torch.cat((win_k, rool_k), dim=4)
            win_v = torch.cat((win_v, rool_v), dim=4)
            del rool_k, rool_v
        else:
            win_k = win_k
            win_v = win_v
        del k, v

        # pool_k and pool_v
        if self.pooling_token:
            win_k = torch.cat((win_k, pool_k), dim=4)
            win_v = torch.cat((win_v, pool_v), dim=4)
            del pool_k, pool_v

        # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
        out = torch.zeros_like(win_q)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(b * l_t, new_h, new_w))
        mask = mask.view(b, l_t, n_wh*n_ww)
        mask = torch.sum(mask, dim=1) # [b, n_wh*n_ww]
        for i in range(win_q.shape[0]):
            ### For masked windows
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(mask_n, self.n_head, t*w_h*w_w, c_head)
                win_k_t = win_k[i, mask_ind_i] 
                win_v_t = win_v[i, mask_ind_i] 
                # mask out key and value
                if T_ind is not None:
                    # key [n_wh*n_ww, n_head, t, w_h*w_w, c_head]
                    win_k_t = win_k_t[:, :, T_ind.view(-1)].view(mask_n, self.n_head, -1, c_head)
                    # value
                    win_v_t = win_v_t[:, :, T_ind.view(-1)].view(mask_n, self.n_head, -1, c_head)
                else:
                    win_k_t = win_k_t.view(n_wh*n_ww, self.n_head, t*w_h*w_w, c_head)
                    win_v_t = win_v_t.view(n_wh*n_ww, self.n_head, t*w_h*w_w, c_head)

                # Fused scaled-dot-product-attention kernel (flash-attention /
                # memory-efficient-attention on CUDA) instead of materializing
                # the full [.., seq, seq] attention matrix by hand. q/k/v are
                # already in the [batch, heads, seq, head_dim] layout SDPA
                # expects, and there's no additional mask to apply here (the
                # masked/unmasked key-value sets were already selected above),
                # so this is a direct drop-in. Not bit-exact vs. the manual
                # matmul+softmax+matmul (different kernel/summation order)
                # but numerically equivalent; attn_drop is 0 at inference so
                # dropout_p=0 here matches the (disabled) self.attn_drop.
                y_t = F.scaled_dot_product_attention(win_q_t, win_k_t, win_v_t)

                out[i, mask_ind_i] = y_t.view(-1, self.n_head, t, w_h*w_w, c_head)

            ### For unmasked windows
            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            win_q_s = win_q[i, unmask_ind_i]
            win_k_s = win_k[i, unmask_ind_i, :, :, :w_h*w_w]
            win_v_s = win_v[i, unmask_ind_i, :, :, :w_h*w_w]

            # Same fused-kernel swap as the masked branch above. SDPA treats
            # every dim except the last two (sequence, head_dim) as batch, so
            # it works directly on these 5D (unmask_n, n_head, t, w_h*w_w,
            # c_head) tensors -- attention stays independent per-t here, same
            # as the original per-t broadcasted matmul.
            y_s = F.scaled_dot_product_attention(win_q_s, win_k_s, win_v_s)
            out[i, unmask_ind_i] = y_s

        # re-assemble all head outputs side by side
        out = out.view(b, n_wh, n_ww, self.n_head, t, w_h, w_w, c_head)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)


        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]

        # output projection
        out = self.proj_drop(self.proj(out))
        return out


class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size,
                norm_layer=nn.LayerNorm, t2t_params=None):
        super().__init__()
        self.window_size = window_size
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, t2t_params=t2t_params)

    def forward(self, x, fold_x_size, mask=None, T_ind=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        B, T, H, W, C = x.shape # 20 36

        shortcut = x
        x = self.norm1(x)
        att_x = self.attention(x, mask, T_ind)

        # FFN
        x = shortcut + att_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), fold_x_size).view(B, T, H, W, C)

        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t_params=None):
        super().__init__()
        blocks = []
        for i in range(depths):
             blocks.append(
                TemporalSparseTransformer(dim, n_head, window_size, pool_size, t2t_params=t2t_params)
             )
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths

    def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            l_mask: local mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        T = x.size(1)
        T_ind = [torch.arange(i, T, t_dilation) for i in range(t_dilation)] * (self.depths // t_dilation)

        for i in range(0, self.depths):
            x = self.transformer[i](x, fold_x_size, l_mask, T_ind[i])

        return x
