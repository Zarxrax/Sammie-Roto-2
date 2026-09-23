from dataclasses import dataclass, field


@dataclass
class LongTermConfig:
    count_usage: bool = True
    max_mem_frames: int = 10
    min_mem_frames: int = 5
    num_prototypes: int = 128
    max_num_tokens: int = 10000
    buffer_tokens: int = 2000


@dataclass
class PixelEncoderConfig:
    type: str = "resnet50"
    ms_dims: tuple[int, ...] = (1024, 512, 256, 64, 3)


@dataclass
class MaskEncoderConfig:
    type: str = "resnet18"
    final_dim: int = 256


@dataclass
class ReadFromPixelConfig:
    input_norm: bool = False
    input_add_pe: bool = False
    add_pe_to_qkv: tuple[bool, bool, bool] = (True, True, False)


@dataclass
class AttentionConfig:
    add_pe_to_qkv: tuple[bool, bool, bool] = (True, True, False)


@dataclass
class ReadFromQueryConfig:
    add_pe_to_qkv: tuple[bool, bool, bool] = (True, True, False)
    output_norm: bool = False


@dataclass
class ObjectTransformerConfig:
    embed_dim: int = 256
    ff_dim: int = 2048
    num_heads: int = 8
    num_blocks: int = 3
    num_queries: int = 16

    read_from_pixel: ReadFromPixelConfig = field(
        default_factory=ReadFromPixelConfig
    )
    read_from_past: AttentionConfig = field(
        default_factory=AttentionConfig
    )
    read_from_memory: AttentionConfig = field(
        default_factory=AttentionConfig
    )
    read_from_query: ReadFromQueryConfig = field(
        default_factory=ReadFromQueryConfig
    )
    query_self_attention: AttentionConfig = field(
        default_factory=AttentionConfig
    )
    pixel_self_attention: AttentionConfig = field(
        default_factory=AttentionConfig
    )


@dataclass
class ObjectSummarizerConfig:
    embed_dim: int = 256
    num_summaries: int = 16
    add_pe: bool = True


@dataclass
class AuxComponentConfig:
    enabled: bool = True
    weight: float = 0.01


@dataclass
class AuxLossConfig:
    sensory: AuxComponentConfig = field(
        default_factory=AuxComponentConfig
    )
    query: AuxComponentConfig = field(
        default_factory=AuxComponentConfig
    )


@dataclass
class MaskDecoderConfig:
    up_dims: tuple[int, ...] = (256, 128, 128, 64, 16)


@dataclass
class ModelConfig:
    pixel_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    pixel_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    pixel_dim: int = 256
    key_dim: int = 64
    value_dim: int = 256
    sensory_dim: int = 256
    embed_dim: int = 256

    pixel_encoder: PixelEncoderConfig = field(
        default_factory=PixelEncoderConfig
    )
    mask_encoder: MaskEncoderConfig = field(
        default_factory=MaskEncoderConfig
    )

    pixel_pe_scale: int = 32
    pixel_pe_temperature: int = 128

    object_transformer: ObjectTransformerConfig = field(
        default_factory=ObjectTransformerConfig
    )
    object_summarizer: ObjectSummarizerConfig = field(
        default_factory=ObjectSummarizerConfig
    )

    aux_loss: AuxLossConfig = field(
        default_factory=AuxLossConfig
    )

    mask_decoder: MaskDecoderConfig = field(
        default_factory=MaskDecoderConfig
    )


@dataclass
class MatAnyoneConfig:
    # Inference configuration
    flip_aug: bool = False
    max_internal_size: int = -1

    use_long_term: bool = False
    mem_every: int = 5
    max_mem_frames: int = 5

    long_term: LongTermConfig = field(
        default_factory=LongTermConfig
    )

    top_k: int = 30
    stagger_updates: int = 5

    # -1 means unlimited
    chunk_size: int = -1

    save_aux: bool = False

    # Model configuration
    model: ModelConfig = field(
        default_factory=ModelConfig
    )