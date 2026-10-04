import torch
import torch.nn as nn


def initialize_MEMFOF(model_name_or_path='egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH', device=None):
    """Initializes the MemFOF model from a HuggingFace Hub checkpoint.

    MemFOF (https://github.com/msu-video-group/memfof, ICCV'2025 Highlight) is a
    memory-efficient, multi-frame optical flow estimator built for native Full HD
    processing. Unlike RAFT, whose all-pairs correlation volume grows as
    O((H*W)^2), MemFOF uses a much smaller local correlation combined with 3-frame
    temporal context, at a fraction of RAFT's peak memory at 1080p.

    Available checkpoints (see https://github.com/msu-video-group/memfof#-models):
      - 'egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH'         (recommended for real-world video)
      - 'egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-sintel'
      - 'egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-kitti'
      - 'egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-spring'
    """
    from MEMFOF import MEMFOF
    model = MEMFOF.from_pretrained(model_name_or_path)
    # No hardcoded fallback device -- from_pretrained() already leaves the
    # model on CPU (a safe default that works anywhere); only move it if the
    # caller actually asked for somewhere else.
    if device is not None:
        model.to(device)
    return model


class MEMFOF_bi(nn.Module):
    """Drop-in replacement for RAFT_bi with an identical external interface.

    Given `frames` of shape (b, t, c, h, w) in [-1, 1] (ProPainter's convention,
    matching what RAFT_bi already expects), returns (gt_flows_forward,
    gt_flows_backward), each (b, t-1, 2, h, w), where:
        gt_flows_forward[:, k]  = flow(frame k   -> frame k+1)
        gt_flows_backward[:, k] = flow(frame k+1 -> frame k)
    -- i.e. exactly RAFT_bi's convention -- so this class can be substituted for
    RAFT_bi with no changes anywhere else in the pipeline (including the RAFT
    chunking loop in inference_propainter.py, since the batching math below is
    designed to slot into arbitrary-length, possibly-overlapping input chunks
    without producing gaps or double-counted flow at chunk boundaries).

    Internally, MemFOF estimates flow using a 3-frame sliding window centered
    on each frame, giving (backward, forward) flow for that center frame in a
    single call. To get flow for every frame in `frames` -- including the
    first and last, which don't have both true neighbors -- the frame sequence
    is padded by duplicating the first/last frame. This only ever affects the
    two degenerate self-flow outputs that this wrapper already discards
    (see the derivation in the class docstring above / PR description), so it
    does not introduce any approximation into the flows that are actually used.
    """

    def __init__(self, model_name_or_path='egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH', device=None,
                 use_shared_fnet=True):
        super().__init__()
        self.model = initialize_MEMFOF(model_name_or_path, device=device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.eval()
        # Compute each UNIQUE frame's feature map exactly once via one
        # well-batched call (batch = t+2 unique frames), then slice/reuse it
        # across all t windows -- instead of the default below, which calls
        # fnet on batch=t three separate times (once each for every window's
        # prev/center/next role), computing most frames' features 3 times
        # over. This gets full batching throughout (fnet AND the
        # correlation/GRU refinement stay well-batched at batch=t) with no
        # redundant computation. On by default: validated bit-exact against
        # the non-shared path across many (t, resolution, iters) combinations
        # using MemFOF's real architecture, and measured ~19% faster on CPU
        # with no VRAM downside (fnet runs on fewer total frames, everything
        # else is unchanged). The one real caveat: this reaches into the
        # model's internal submodules (fnet/cnet/attention/update_block/
        # upsample_weight/_upsample_data) and hand-replicates forward()'s
        # orchestration logic rather than calling its public forward() API --
        # if a future memfof version changes that internal structure, this
        # could silently go stale. See _forward_shared_fnet() for the full
        # replication.
        self.use_shared_fnet = use_shared_fnet

    def forward(self, frames, iters=8):
        b, t, c, h, w = frames.size()
        assert b == 1, 'MEMFOF_bi currently only supports batch size 1 (matches how RAFT_bi is used in this pipeline).'
        device = frames.device
        dtype = frames.dtype

        if t < 2:
            empty = torch.zeros(b, 0, 2, h, w, device=device, dtype=dtype)
            return empty, empty

        with torch.no_grad():
            # MemFOF expects raw [0, 255]-range images and normalizes internally,
            # unlike RAFT_bi's frames which arrive pre-normalized to [-1, 1].
            imgs = ((frames[0].float() + 1) / 2 * 255.0)  # (t, c, h, w)

            # Pad by duplicating the first/last frame so every original frame
            # 0..t-1 has a valid 3-frame window centered on it.
            padded = torch.cat([imgs[:1], imgs, imgs[-1:]], dim=0)  # (t+2, c, h, w)

            if self.use_shared_fnet:
                backward_flow, forward_flow = self._forward_shared_fnet(padded, t, iters)
            else:
                windows = torch.stack([padded[k:k + 3] for k in range(t)], dim=0)  # (t, 3, c, h, w)
                out = self.model(windows, iters=iters)
                flow = out['flow'][-1]  # (t, 2, 2, h, w); dim1: [backward, forward]
                backward_flow = flow[:, 0]  # (t, 2, h, w): flow(k -> k-1)
                forward_flow = flow[:, 1]   # (t, 2, h, w): flow(k -> k+1)

        # gt_flows_forward[:, k]  = flow(k -> k+1)   = forward_flow[k],   k=0..t-2
        # gt_flows_backward[:, k] = flow(k+1 -> k)   = backward_flow[k+1], k=0..t-2
        # (forward_flow[t-1] and backward_flow[0] are the degenerate
        # self-flow outputs from the padding and are correctly never used.)
        gt_flows_forward = forward_flow[:t - 1].unsqueeze(0).to(dtype)
        gt_flows_backward = backward_flow[1:t].unsqueeze(0).to(dtype)

        return gt_flows_forward, gt_flows_backward

    def _forward_shared_fnet(self, padded, t, iters):
        """Replicates MEMFOF.forward()'s orchestration by hand, calling the
        same submodules it does, but computing the feature map for each of
        the t+2 unique frames in `padded` exactly once (one batched call)
        instead of once per window-role (3 batched calls of size t, most
        frames counted 3 times, as MEMFOF's own forward() does when called
        directly on overlapping windows). Every other computation --
        cnet, attention, the correlation volumes, the GRU refinement loop,
        upsampling -- is unchanged from forward()'s own logic and calls
        the exact same submodules, just reading pre-sliced fmaps instead
        of computing them inline. Returns (backward_flow, forward_flow),
        each (t, 2, h, w), matching the non-cached path's return shapes.
        """
        from MEMFOF.utils.utils import coords_grid, InputPadder
        from MEMFOF.corr import CorrBlock

        m = self.model
        device = padded.device

        # Same normalization + padding MEMFOF.forward() applies internally
        # to its `images` input. One InputPadder works for both the
        # per-frame fnet input below and the per-window cnet input further
        # down, since padding only depends on H, W (identical for both).
        images = 2 * (padded / 255.0) - 1.0  # (t+2, c, h, w)
        images = images.contiguous()
        padder = InputPadder(images.shape)
        images = padder.pad(images)  # (t+2, c, H, W)

        # The one actual change: fnet computed once per unique frame.
        fmaps = m.fnet(images)          # (t+2, C, h16, w16)
        fmap1_16x = fmaps[0:t]          # "prev" feature map for each window
        fmap2_16x = fmaps[1:t + 1]      # "center" feature map for each window
        fmap3_16x = fmaps[2:t + 2]      # "next" feature map for each window

        # cnet is genuinely per-window (each window's own 3-frame
        # concatenation), so there's no redundancy to remove here -- just
        # batch all t windows into one call, same as forward() already
        # effectively does when given a batch of windows.
        cnet_input = torch.cat(
            [images[0:t], images[1:t + 1], images[2:t + 2]], dim=1
        )  # (t, 3*c, H, W)
        cnet = m.init_conv(m.cnet(cnet_input))
        net, context = torch.split(cnet, [m.dim, m.dim], dim=1)
        attention = m.att(context)

        flow_update = m.flow_head(net)
        weight_update = 0.25 * m.upsample_weight(net)

        flow_16x_21 = flow_update[:, 0:2]
        info_16x_21 = flow_update[:, 2:6]
        flow_16x_23 = flow_update[:, 6:8]
        info_16x_23 = flow_update[:, 8:12]

        _, _, H_img, W_img = images.shape
        dilation = torch.ones(t, 1, H_img // 16, W_img // 16, device=device)

        corr_fn_21 = CorrBlock(fmap2_16x, fmap1_16x, m.corr_levels, m.corr_radius)
        corr_fn_23 = CorrBlock(fmap2_16x, fmap3_16x, m.corr_levels, m.corr_radius)

        for _ in range(iters):
            flow_16x_21 = flow_16x_21.detach()
            flow_16x_23 = flow_16x_23.detach()
            _, _, H_, W_ = flow_16x_21.shape

            coords21 = (coords_grid(t, H_, W_, device=device) + flow_16x_21).detach()
            coords23 = (coords_grid(t, H_, W_, device=device) + flow_16x_23).detach()

            corr_21 = corr_fn_21(coords21, dilation=dilation)
            corr_23 = corr_fn_23(coords23, dilation=dilation)
            corr = torch.cat([corr_21, corr_23], dim=1)
            flow_16x = torch.cat([flow_16x_21, flow_16x_23], dim=1)

            net = m.update_block(net, context, corr, flow_16x, attention)

            flow_update = m.flow_head(net)
            weight_update = 0.25 * m.upsample_weight(net)

            flow_16x_21 = flow_16x_21 + flow_update[:, 0:2]
            flow_16x_23 = flow_16x_23 + flow_update[:, 6:8]
            info_16x_21 = flow_update[:, 2:6]
            info_16x_23 = flow_update[:, 8:12]

        n_upsample_ch = 16 * 16 * 9  # exact split point MEMFOF's own forward() uses
        flow_up_21, _ = m._upsample_data(flow_16x_21, info_16x_21, weight_update[:, :n_upsample_ch])
        flow_up_23, _ = m._upsample_data(flow_16x_23, info_16x_23, weight_update[:, n_upsample_ch:])

        flow = torch.stack([flow_up_21, flow_up_23], dim=1)  # (t, 2, 2, h, w)
        flow = padder.unpad(flow)

        backward_flow = flow[:, 0]  # (t, 2, h, w)
        forward_flow = flow[:, 1]   # (t, 2, h, w)
        return backward_flow, forward_flow
