"""VideoMAE V2 backbones and verified distilled-checkpoint loading.

The transformer definition follows the official OpenGVLab VideoMAE V2
fine-tuning model while using PyTorch scaled-dot-product attention to reduce
attention-memory pressure on modern GPUs.  Released checkpoints are pinned to
one Hugging Face revision and verified by SHA-256 before deserialization.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint as activation_checkpoint


@dataclass(frozen=True)
class VideoMAEV2CheckpointSpec:
    """Architecture and provenance for an official distilled checkpoint."""

    architecture: str
    variant: str
    repo_id: str
    revision: str
    filename: str
    sha256: str
    file_size: int
    pretraining_classes: int
    embed_dim: int
    depth: int
    num_heads: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable checkpoint metadata."""
        return asdict(self)


_VIDEO_MAE_V2_REVISION: Final = "706cc172d65ebd4dedbee3f9c0183a93df9fa125"
_VIDEO_MAE_V2_REPO: Final = "OpenGVLab/VideoMAE2"
_VIDEO_MAE_V2_SPECS: Final[dict[str, VideoMAEV2CheckpointSpec]] = {
    "videomaev2_vit_s_distilled": VideoMAEV2CheckpointSpec(
        architecture="videomaev2_vit_s_distilled",
        variant="small",
        repo_id=_VIDEO_MAE_V2_REPO,
        revision=_VIDEO_MAE_V2_REVISION,
        filename="distill/vit_s_k710_dl_from_giant.pth",
        sha256="24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172",
        file_size=44_334_609,
        pretraining_classes=710,
        embed_dim=384,
        depth=12,
        num_heads=6,
    ),
    "videomaev2_vit_b_distilled": VideoMAEV2CheckpointSpec(
        architecture="videomaev2_vit_b_distilled",
        variant="base",
        repo_id=_VIDEO_MAE_V2_REPO,
        revision=_VIDEO_MAE_V2_REVISION,
        filename="distill/vit_b_k710_dl_from_giant.pth",
        sha256="8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec",
        file_size=173_574_417,
        pretraining_classes=710,
        embed_dim=768,
        depth=12,
        num_heads=12,
    ),
}


def checkpoint_spec_for_architecture(
    architecture: str,
) -> VideoMAEV2CheckpointSpec:
    """Return the pinned checkpoint specification for a supported architecture."""
    try:
        return _VIDEO_MAE_V2_SPECS[architecture]
    except KeyError as error:
        supported = ", ".join(sorted(_VIDEO_MAE_V2_SPECS))
        raise ValueError(
            f"Unsupported VideoMAE V2 architecture {architecture!r}; "
            f"supported values are: {supported}"
        ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_file(
    path: str | Path,
    spec: VideoMAEV2CheckpointSpec,
) -> Path:
    """Verify the official checkpoint size and SHA-256 digest."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VideoMAE V2 checkpoint not found: {checkpoint_path}")
    actual_size = checkpoint_path.stat().st_size
    if actual_size != spec.file_size:
        raise RuntimeError(
            "VideoMAE V2 checkpoint size mismatch: "
            f"expected {spec.file_size:,} bytes, found {actual_size:,} bytes "
            f"at {checkpoint_path}"
        )
    actual_sha256 = _sha256_file(checkpoint_path)
    if actual_sha256.lower() != spec.sha256.lower():
        raise RuntimeError(
            "VideoMAE V2 checkpoint SHA-256 mismatch: "
            f"expected {spec.sha256}, found {actual_sha256} at {checkpoint_path}"
        )
    return checkpoint_path


def download_verified_checkpoint(
    architecture: str,
    cache_dir: str | Path | None = None,
    verify_sha256: bool = True,
) -> Path:
    """Download a pinned official checkpoint and optionally verify its digest."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "huggingface_hub is required for VideoMAE V2 pretrained weights. "
            "Install the repository requirements before training."
        ) from error

    spec = checkpoint_spec_for_architecture(architecture)
    downloaded = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    if verify_sha256:
        return verify_checkpoint_file(downloaded, spec)
    if not downloaded.is_file():
        raise FileNotFoundError(
            f"Hugging Face reported a checkpoint path that does not exist: {downloaded}"
        )
    return downloaded


def _drop_path(
    tensor: Tensor,
    drop_probability: float,
    training: bool,
) -> Tensor:
    if drop_probability <= 0.0 or not training:
        return tensor
    keep_probability = 1.0 - drop_probability
    random_shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
    random_tensor = keep_probability + torch.rand(
        random_shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    random_tensor.floor_()
    return tensor.div(keep_probability) * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_probability: float = 0.0) -> None:
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, tensor: Tensor) -> Tensor:
        return _drop_path(tensor, self.drop_probability, self.training)


class Mlp(nn.Module):
    """Transformer feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.fc1(tensor)
        tensor = self.act(tensor)
        tensor = self.fc2(tensor)
        return self.drop(tensor)


class Attention(nn.Module):
    """Multi-head self-attention with official VideoMAE V2 parameter names."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("q_bias", None)
            self.register_parameter("v_bias", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, tensor: Tensor) -> Tensor:
        batch_size, token_count, embedding_dim = tensor.shape
        qkv_bias = None
        if self.q_bias is not None and self.v_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = functional.linear(tensor, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        # SDPA selects FlashAttention or the memory-efficient kernel when the
        # current GPU/PyTorch combination supports it, and otherwise uses math.
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            token_count,
            embedding_dim,
        )
        attended = self.proj(attended)
        return self.proj_drop(attended)


class Block(nn.Module):
    """Pre-normalized VideoMAE V2 transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        init_values: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.register_parameter("gamma_1", None)
            self.register_parameter("gamma_2", None)

    def forward(self, tensor: Tensor) -> Tensor:
        if self.gamma_1 is None or self.gamma_2 is None:
            tensor = tensor + self.drop_path(self.attn(self.norm1(tensor)))
            tensor = tensor + self.drop_path(self.mlp(self.norm2(tensor)))
            return tensor
        tensor = tensor + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(tensor))
        )
        tensor = tensor + self.drop_path(
            self.gamma_2 * self.mlp(self.norm2(tensor))
        )
        return tensor


class PatchEmbed(nn.Module):
    """Tubelet and spatial-patch embedding using a 3-D convolution."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        num_frames: int = 16,
        tubelet_size: int = 2,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if num_frames % tubelet_size != 0:
            raise ValueError("num_frames must be divisible by tubelet_size")
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.tubelet_size = tubelet_size
        spatial_patches = (img_size // patch_size) ** 2
        self.num_patches = spatial_patches * (num_frames // tubelet_size)
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        _batch, _channels, frames, height, width = tensor.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input image size {height}x{width} does not match "
                f"{self.img_size[0]}x{self.img_size[1]}"
            )
        if frames % self.tubelet_size != 0:
            raise ValueError(
                f"Input frame count {frames} is not divisible by tubelet size "
                f"{self.tubelet_size}"
            )
        return self.proj(tensor).flatten(2).transpose(1, 2)


def _sinusoid_encoding_table(position_count: int, dimension: int) -> Tensor:
    positions = torch.arange(position_count, dtype=torch.float32).unsqueeze(1)
    even_indices = torch.arange(0, dimension, 2, dtype=torch.float32)
    divisors = torch.exp(-math.log(10000.0) * even_indices / dimension)
    table = torch.zeros(position_count, dimension, dtype=torch.float32)
    table[:, 0::2] = torch.sin(positions * divisors)
    if dimension > 1:
        odd_width = table[:, 1::2].shape[1]
        table[:, 1::2] = torch.cos(positions * divisors[:odd_width])
    return table.unsqueeze(0)


class VideoMAEV2VisionTransformer(nn.Module):
    """VideoMAE V2 ViT encoder compatible with official distilled weights."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        head_drop_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
        init_values: float = 0.0,
        num_frames: int = 16,
        tubelet_size: int = 2,
        use_mean_pooling: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
        self.register_buffer(
            "pos_embed",
            _sinusoid_encoding_table(self.patch_embed.num_patches, embed_dim),
            persistent=False,
        )
        self.pos_drop = nn.Dropout(drop_rate)
        normalization = lambda dimension: nn.LayerNorm(  # noqa: E731
            dimension,
            eps=layer_norm_eps,
        )
        drop_path_values = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=float(drop_path_values[index]),
                    norm_layer=normalization,
                    init_values=init_values,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else normalization(embed_dim)
        self.fc_norm = normalization(embed_dim) if use_mean_pooling else None
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def get_num_layers(self) -> int:
        """Return the number of transformer blocks for layer-wise LR decay."""
        return len(self.blocks)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable or disable activation checkpointing for transformer blocks."""
        self.gradient_checkpointing = bool(enabled)

    def forward_features(self, tensor: Tensor) -> Tensor:
        tensor = self.patch_embed(tensor)
        if tensor.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Token count {tensor.shape[1]} does not match pretrained "
                f"position table length {self.pos_embed.shape[1]}"
            )
        tensor = tensor + self.pos_embed.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
        tensor = self.pos_drop(tensor)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and tensor.requires_grad:
                tensor = activation_checkpoint(
                    block,
                    tensor,
                    use_reentrant=False,
                )
            else:
                tensor = block(tensor)
        if self.fc_norm is not None:
            return self.fc_norm(tensor.mean(dim=1))
        return self.norm(tensor[:, 0])

    def forward(self, tensor: Tensor) -> Tensor:
        features = self.forward_features(tensor)
        return self.head(self.head_dropout(features))


_PREFIXES: Final = (
    "module.",
    "model.",
    "backbone.",
    "encoder.",
    "student.",
    "student_model.",
    "video_encoder.",
)


def _candidate_checkpoint_keys(raw_key: str) -> list[str]:
    """Return prefix-stripped candidates in breadth-first order."""
    queue: deque[str] = deque([raw_key])
    visited: set[str] = set()
    candidates: list[str] = []
    while queue:
        candidate = queue.popleft()
        if candidate in visited:
            continue
        visited.add(candidate)
        candidates.append(candidate)
        for prefix in _PREFIXES:
            if candidate.startswith(prefix):
                queue.append(candidate[len(prefix) :])
    return candidates


def _extract_tensor_mapping(payload: Any) -> dict[str, Tensor]:
    """Find the tensor state dictionary inside common checkpoint wrappers."""
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            "The VideoMAE V2 checkpoint is not a mapping and cannot be loaded safely"
        )
    for wrapper_key in (
        "model",
        "state_dict",
        "model_state_dict",
        "module",
        "student",
    ):
        nested = payload.get(wrapper_key)
        if isinstance(nested, Mapping):
            try:
                return _extract_tensor_mapping(nested)
            except RuntimeError:
                pass
    tensors = {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, Tensor)
    }
    if tensors:
        return tensors
    raise RuntimeError(
        "No tensor state dictionary was found in the VideoMAE V2 checkpoint"
    )


def _match_checkpoint_state_dict(
    model: nn.Module,
    checkpoint_state: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Match a possibly prefixed checkpoint to the local model by key and shape."""
    model_state = model.state_dict()
    matched: dict[str, Tensor] = {}
    collisions: list[str] = []
    for raw_key, value in checkpoint_state.items():
        for candidate in _candidate_checkpoint_keys(str(raw_key)):
            expected = model_state.get(candidate)
            if expected is None or tuple(expected.shape) != tuple(value.shape):
                continue
            if candidate in matched:
                collisions.append(candidate)
                break
            matched[candidate] = value
            break

    backbone_keys = [
        key for key in model_state if not key.startswith("head.")
    ]
    total_backbone_parameters = sum(model_state[key].numel() for key in backbone_keys)
    matched_backbone_parameters = sum(
        model_state[key].numel() for key in backbone_keys if key in matched
    )
    match_fraction = (
        matched_backbone_parameters / total_backbone_parameters
        if total_backbone_parameters
        else 0.0
    )
    missing_backbone_keys = [key for key in backbone_keys if key not in matched]
    report = {
        "checkpoint_tensor_count": len(checkpoint_state),
        "matched_tensor_count": len(matched),
        "matched_backbone_parameter_fraction": match_fraction,
        "missing_backbone_keys": missing_backbone_keys,
        "collisions": sorted(set(collisions)),
    }
    return matched, report


def _load_checkpoint_payload(
    checkpoint_path: Path,
    *,
    allow_verified_legacy_pickle: bool,
) -> Any:
    """Load tensors safely, with a legacy fallback only for a verified file."""
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Older PyTorch releases do not expose ``weights_only``. The repository
        # requires torch>=2.2, but retaining this branch makes the error clearer.
        if not allow_verified_legacy_pickle:
            raise RuntimeError(
                "This PyTorch build cannot use weights_only loading, and the "
                "checkpoint was not SHA-256 verified. Refusing legacy pickle "
                "deserialization."
            )
        return torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError as error:
        # Some older official checkpoints contain otherwise harmless wrappers
        # that the restricted unpickler may reject. A full pickle load is used
        # only after exact size and SHA-256 verification against the pinned file.
        if not allow_verified_legacy_pickle:
            raise RuntimeError(
                "The checkpoint requires legacy pickle loading, but exact "
                "SHA-256 verification is disabled. Enable verification or use "
                "a trusted, converted state dictionary."
            ) from error
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )


def load_official_distilled_weights(
    model: VideoMAEV2VisionTransformer,
    architecture: str,
    cache_dir: str | Path | None = None,
    verify_sha256: bool = True,
    checkpoint_path: str | Path | None = None,
    minimum_backbone_match_fraction: float = 0.97,
) -> dict[str, Any]:
    """Load a pinned official K710-distilled checkpoint into ``model``."""
    spec = checkpoint_spec_for_architecture(architecture)
    if checkpoint_path is None:
        resolved_checkpoint = download_verified_checkpoint(
            architecture,
            cache_dir=cache_dir,
            verify_sha256=verify_sha256,
        )
    else:
        resolved_checkpoint = Path(checkpoint_path)
        if verify_sha256:
            verify_checkpoint_file(resolved_checkpoint, spec)
        elif not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                f"VideoMAE V2 checkpoint not found: {resolved_checkpoint}"
            )
    payload = _load_checkpoint_payload(
        resolved_checkpoint,
        allow_verified_legacy_pickle=bool(verify_sha256),
    )
    checkpoint_state = _extract_tensor_mapping(payload)
    matched, report = _match_checkpoint_state_dict(model, checkpoint_state)
    if report["matched_backbone_parameter_fraction"] < minimum_backbone_match_fraction:
        missing = report["missing_backbone_keys"][:12]
        raw_examples = list(checkpoint_state)[:12]
        raise RuntimeError(
            "The downloaded VideoMAE V2 checkpoint is incompatible with the "
            f"local architecture. Matched only "
            f"{report['matched_backbone_parameter_fraction']:.2%} of backbone "
            f"parameters. Missing examples: {missing}. Checkpoint key examples: "
            f"{raw_examples}"
        )
    incompatible = model.load_state_dict(matched, strict=False)
    report.update(
        {
            "missing_after_load": list(incompatible.missing_keys),
            "unexpected_after_load": list(incompatible.unexpected_keys),
            "checkpoint": spec.to_dict(),
            "resolved_checkpoint_path": str(resolved_checkpoint.resolve()),
        }
    )
    print(
        "Loaded official VideoMAE V2 distilled weights: "
        f"{architecture}, backbone match="
        f"{report['matched_backbone_parameter_fraction']:.2%}, "
        f"matched tensors={report['matched_tensor_count']}"
    )
    return report


def build_videomaev2_backbone(
    architecture: str,
    *,
    pretrained: bool = True,
    gradient_checkpointing: bool = True,
    drop_path_rate: float = 0.1,
    cache_dir: str | Path | None = None,
    verify_sha256: bool = True,
    pretrained_checkpoint: str | Path | None = None,
) -> tuple[VideoMAEV2VisionTransformer, int]:
    """Build a small or base VideoMAE V2 encoder for downstream fine-tuning."""
    spec = checkpoint_spec_for_architecture(architecture)
    model = VideoMAEV2VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=spec.pretraining_classes if pretrained else 0,
        embed_dim=spec.embed_dim,
        depth=spec.depth,
        num_heads=spec.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        head_drop_rate=0.0,
        layer_norm_eps=1e-6,
        init_values=0.0,
        num_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        gradient_checkpointing=gradient_checkpointing,
    )
    if pretrained:
        load_report = load_official_distilled_weights(
            model,
            architecture,
            cache_dir=cache_dir,
            verify_sha256=verify_sha256,
            checkpoint_path=pretrained_checkpoint,
        )
        model.pretrained_checkpoint_info = load_report["checkpoint"]
        model.pretrained_load_report = {
            key: value
            for key, value in load_report.items()
            if key != "checkpoint"
        }
    else:
        model.pretrained_checkpoint_info = None
        model.pretrained_load_report = None
    # The released K710 head is useful for weight loading but HMDB51 receives a
    # fresh classifier outside the backbone wrapper.
    model.head = nn.Identity()
    model.num_classes = 0
    return model, spec.embed_dim


def preflight_videomaev2_checkpoint(
    architecture: str,
    *,
    cache_dir: str | Path | None = None,
    verify_sha256: bool = True,
    pretrained_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Download, verify, and structurally load a VideoMAE V2 checkpoint on CPU.

    This deliberately constructs the real small/base encoder and requires at
    least a 97% backbone-parameter match before returning. It is intended as a
    Colab setup check before a long GPU run.
    """
    model, feature_size = build_videomaev2_backbone(
        architecture,
        pretrained=True,
        gradient_checkpointing=False,
        drop_path_rate=0.0,
        cache_dir=cache_dir,
        verify_sha256=verify_sha256,
        pretrained_checkpoint=pretrained_checkpoint,
    )
    report = dict(model.pretrained_load_report or {})
    report.update(
        {
            "architecture": architecture,
            "feature_size": feature_size,
            "backbone_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    )
    return report
