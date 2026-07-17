"""Neural-network models for video classification."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn
from torchvision import models

from torchvision.models.video import (
    MViT_V2_S_Weights,
    R2Plus1D_18_Weights,
    mvit_v2_s,
    r2plus1d_18,
)

from videomaev2 import build_videomaev2_backbone

_BACKBONES: Final = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
    "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
}


def _set_batch_norm_trainable(module: nn.Module, trainable: bool) -> None:
    """Enable or disable BatchNorm affine parameters beneath a module."""
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            for parameter in child.parameters():
                parameter.requires_grad = trainable


def _freeze_batch_norm_statistics(module: nn.Module) -> None:
    """Keep BatchNorm running statistics fixed during small-batch fine-tuning."""
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


class TemporalAttention(nn.Module):
    """Learn a normalized importance weight for each valid time step."""

    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.Tanh(),
            nn.Linear(feature_size // 2, 1),
        )

    def forward(self, sequence: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return an attention-weighted sequence representation."""
        scores = self.score(sequence).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class LRCN(nn.Module):
    """CNN + bidirectional LSTM + temporal-attention video classifier."""

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        dropout_rate: float,
        n_classes: int,
        pretrained: bool = True,
        cnn_model: str = "resnet34",
        bidirectional: bool = True,
        freeze_backbone: bool = False,
        freeze_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if cnn_model not in _BACKBONES:
            raise ValueError(f"Unsupported CNN backbone: {cnn_model}")

        constructor, default_weights = _BACKBONES[cnn_model]
        backbone = constructor(weights=default_weights if pretrained else None)
        feature_size = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.base_model = backbone
        self.feature_size = feature_size
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        self.set_backbone_trainable(not freeze_backbone)

        recurrent_dropout = dropout_rate if n_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = TemporalAttention(output_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout_rate),
            nn.Linear(output_size, n_classes),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the CNN backbone while optionally fixing BatchNorm."""
        self.freeze_backbone = not trainable
        for parameter in self.base_model.parameters():
            parameter.requires_grad = trainable
        if trainable and self.freeze_batch_norm:
            _set_batch_norm_trainable(self.base_model, False)

    def train(self, mode: bool = True):
        """Set training mode while preserving frozen-backbone and BatchNorm state."""
        super().train(mode)
        if mode and self.freeze_backbone:
            self.base_model.eval()
        elif mode and self.freeze_batch_norm:
            _freeze_batch_norm_statistics(self.base_model)
        return self

    def _forward_single_clip(self, inputs: Tensor, lengths: Tensor | None = None) -> Tensor:
        batch_size, time_steps, channels, height, width = inputs.shape
        frames = inputs.reshape(batch_size * time_steps, channels, height, width)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.base_model(frames)
        else:
            features = self.base_model(frames)
        features = features.reshape(batch_size, time_steps, self.feature_size)
        recurrent_output, _ = self.rnn(features)
        mask = None
        if lengths is not None:
            positions = torch.arange(time_steps, device=inputs.device).unsqueeze(0)
            mask = positions < lengths.unsqueeze(1)
        return self.classifier(self.attention(recurrent_output, mask))

    def forward(self, inputs: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Classify clips shaped ``[B,T,C,H,W]`` or ``[B,K,T,C,H,W]``."""
        if inputs.ndim == 5:
            return self._forward_single_clip(inputs, lengths)
        if inputs.ndim != 6:
            raise ValueError(f"Expected a 5-D or 6-D tensor, received {tuple(inputs.shape)}")
        batch_size, clips, time_steps, channels, height, width = inputs.shape
        flat = inputs.reshape(batch_size * clips, time_steps, channels, height, width)
        logits = self._forward_single_clip(flat, None)
        return logits.reshape(batch_size, clips, -1).mean(dim=1)


class _VideoBackboneClassifier(nn.Module):
    """Shared multi-clip wrapper for torchvision video backbones."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        n_classes: int,
        dropout_rate: float,
        freeze_backbone: bool,
        freeze_batch_norm: bool,
        clip_forward_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.base_model = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, n_classes),
        )
        if clip_forward_batch_size is not None and clip_forward_batch_size <= 0:
            raise ValueError("clip_forward_batch_size must be positive or None")
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        self.clip_forward_batch_size = clip_forward_batch_size
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the spatiotemporal backbone."""
        self.freeze_backbone = not trainable
        for parameter in self.base_model.parameters():
            parameter.requires_grad = trainable
        if trainable and self.freeze_batch_norm:
            _set_batch_norm_trainable(self.base_model, False)

    def train(self, mode: bool = True):
        """Set training mode while preserving frozen-backbone and BatchNorm state."""
        super().train(mode)
        if mode and self.freeze_backbone:
            self.base_model.eval()
        elif mode and self.freeze_batch_norm:
            _freeze_batch_norm_statistics(self.base_model)
        return self

    def _forward_single_clip(self, clips: Tensor) -> Tensor:
        # Dataset layout is [B,T,C,H,W]; torchvision video models expect [B,C,T,H,W].
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.base_model(clips)
        else:
            features = self.base_model(clips)
        return self.classifier(features)

    def forward(self, inputs: Tensor, _lengths: Tensor | None = None) -> Tensor:
        """Classify ``[B,T,C,H,W]`` or average ``[B,K,T,C,H,W]`` clips."""
        if inputs.ndim == 5:
            return self._forward_single_clip(inputs)
        if inputs.ndim != 6:
            raise ValueError(f"Expected a 5-D or 6-D tensor, received {tuple(inputs.shape)}")
        batch_size, clip_count, time_steps, channels, height, width = inputs.shape
        flat = inputs.reshape(
            batch_size * clip_count,
            time_steps,
            channels,
            height,
            width,
        )
        chunk_size = self.clip_forward_batch_size
        if chunk_size is None or flat.shape[0] <= chunk_size:
            logits = self._forward_single_clip(flat)
        else:
            logits = torch.cat(
                [
                    self._forward_single_clip(flat[start : start + chunk_size])
                    for start in range(0, flat.shape[0], chunk_size)
                ],
                dim=0,
            )
        return logits.reshape(batch_size, clip_count, -1).mean(dim=1)


class VideoR2Plus1D(_VideoBackboneClassifier):
    """Kinetics-pretrained R(2+1)D-18 classifier with multi-clip averaging."""

    def __init__(
        self,
        n_classes: int,
        dropout_rate: float = 0.25,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        freeze_batch_norm: bool = True,
    ) -> None:
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        backbone = r2plus1d_18(weights=weights)
        feature_size = backbone.fc.in_features
        backbone.fc = nn.Identity()
        super().__init__(
            backbone=backbone,
            feature_size=feature_size,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            freeze_batch_norm=freeze_batch_norm,
        )


class VideoMViTV2(_VideoBackboneClassifier):
    """Kinetics-pretrained MViT-V2-S classifier with multi-clip averaging."""

    def __init__(
        self,
        n_classes: int,
        dropout_rate: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        freeze_batch_norm: bool = True,
    ) -> None:
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        backbone = mvit_v2_s(weights=weights)
        if not isinstance(backbone.head, nn.Sequential) or len(backbone.head) < 2:
            raise RuntimeError("Unexpected torchvision MViT head structure")
        classifier_layer = backbone.head[-1]
        if not isinstance(classifier_layer, nn.Linear):
            raise RuntimeError("Unexpected torchvision MViT classifier layer")
        feature_size = classifier_layer.in_features
        backbone.head = nn.Identity()
        super().__init__(
            backbone=backbone,
            feature_size=feature_size,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            freeze_batch_norm=freeze_batch_norm,
        )

class VideoMAEV2Distilled(_VideoBackboneClassifier):
    """Official K710-distilled VideoMAE V2 encoder with an HMDB51 head."""

    def __init__(
        self,
        architecture: str,
        n_classes: int,
        dropout_rate: float = 0.35,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        gradient_checkpointing: bool = True,
        drop_path_rate: float = 0.1,
        pretrained_checkpoint: str | None = None,
        pretrained_cache_dir: str | None = None,
        verify_pretrained_sha256: bool = True,
        head_init_scale: float = 0.001,
        clip_forward_batch_size: int = 1,
    ) -> None:
        if head_init_scale <= 0:
            raise ValueError("head_init_scale must be positive")
        backbone, feature_size = build_videomaev2_backbone(
            architecture,
            pretrained=pretrained,
            gradient_checkpointing=gradient_checkpointing,
            drop_path_rate=drop_path_rate,
            cache_dir=pretrained_cache_dir,
            verify_sha256=verify_pretrained_sha256,
            pretrained_checkpoint=pretrained_checkpoint,
        )
        super().__init__(
            backbone=backbone,
            feature_size=feature_size,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            freeze_batch_norm=False,
            clip_forward_batch_size=clip_forward_batch_size,
        )
        self.architecture = architecture
        classifier_layer = self.classifier[-1]
        if not isinstance(classifier_layer, nn.Linear):
            raise RuntimeError("Unexpected VideoMAE V2 classifier structure")
        nn.init.trunc_normal_(classifier_layer.weight, std=0.02)
        classifier_layer.weight.data.mul_(head_init_scale)
        nn.init.zeros_(classifier_layer.bias)

    def optimizer_num_layers(self) -> int:
        """Return patch + transformer-block + final-norm optimizer depths."""
        return int(self.base_model.get_num_layers()) + 2

    def optimizer_layer_id(self, parameter_name: str) -> int | None:
        """Map a backbone parameter name to a layer-wise LR-decay depth."""
        prefix = "base_model."
        if not parameter_name.startswith(prefix):
            return None
        local_name = parameter_name[len(prefix) :]
        if local_name.startswith("patch_embed."):
            return 0
        if local_name.startswith("blocks."):
            parts = local_name.split(".", maxsplit=2)
            try:
                block_index = int(parts[1])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"Could not parse VideoMAE V2 block name: {parameter_name}"
                ) from error
            return block_index + 1
        return self.optimizer_num_layers() - 1

    def pretrained_metadata(self) -> dict | None:
        """Return checkpoint provenance and the structural loading report."""
        checkpoint = getattr(self.base_model, "pretrained_checkpoint_info", None)
        load_report = getattr(self.base_model, "pretrained_load_report", None)
        if checkpoint is None and load_report is None:
            return None
        return {
            "checkpoint": dict(checkpoint) if checkpoint is not None else None,
            "load_report": dict(load_report) if load_report is not None else None,
        }
