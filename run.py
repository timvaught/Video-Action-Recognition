"""Command-line entry point for leakage-conscious HMDB51 training and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from models import LRCN, VideoMAEV2Distilled, VideoMViTV2, VideoR2Plus1D
from train import evaluate, train
from utils import (
    TrainAugmentationConfig,
    compose_data_transforms,
    compose_dataloaders,
    seed_everything,
)
from video_datasets import (
    VideoDataset,
    dataset_audit,
    dataset_split,
    load_dataset,
    official_dataset_split,
    validate_split_integrity,
)


def parse_bool(value: str | bool) -> bool:
    """Parse common command-line Boolean spellings."""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid Boolean value: {value}")


def args_parser() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Leakage-conscious HMDB51 training")
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--ckpt")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--split_file")
    parser.add_argument(
        "--reuse_existing_split",
        type=parse_bool,
        default=False,
        help=(
            "In train mode, load and audit --split_file instead of creating a "
            "new split. This keeps architecture comparisons on identical data."
        ),
    )

    parser.add_argument("--n_classes", type=int, default=51)
    parser.add_argument("--split_protocol", choices=("random", "official"), default="random")
    parser.add_argument("--train_size", type=float, default=0.70)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.15,
        help="Fraction of the official training subset reserved for validation",
    )
    parser.add_argument("--official_split_dir")
    parser.add_argument("--official_split_number", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--official_duplicate_policy",
        choices=("error", "drop_train", "allow"),
        default="drop_train",
        help=(
            "How to handle exact extracted-frame copies crossing the official "
            "train/test boundary. drop_train preserves test and removes training "
            "copies; allow preserves the untouched official benchmark; error aborts."
        ),
    )

    parser.add_argument(
        "--architecture",
        choices=(
            "r2plus1d_18",
            "mvit_v2_s",
            "videomaev2_vit_s_distilled",
            "videomaev2_vit_b_distilled",
            "lrcn",
        ),
        default="videomaev2_vit_s_distilled",
    )
    parser.add_argument("--fr_per_vid", type=int, default=16)
    parser.add_argument("--temporal_stride", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_clips", type=int, default=10)
    parser.add_argument("--test_clips", type=int, default=10)
    parser.add_argument("--flip_tta", type=parse_bool, default=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of mini-batches per optimizer update",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--eval_clip_chunk_size",
        type=int,
        default=1,
        help=(
            "Number of temporal clips forwarded together during multi-clip "
            "evaluation. A value of 1 minimizes VideoMAE V2 VRAM use."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--cnn_backbone", default="resnet50")
    parser.add_argument("--pretrained", type=parse_bool, default=True)
    parser.add_argument(
        "--pretrained_checkpoint",
        help=(
            "Optional local VideoMAE V2 checkpoint. When omitted, the pinned "
            "official checkpoint is downloaded and checksum-verified."
        ),
    )
    parser.add_argument(
        "--pretrained_cache_dir",
        help="Optional persistent directory for downloaded VideoMAE V2 weights",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=parse_bool,
        default=True,
        help="Checkpoint transformer blocks to reduce VideoMAE V2 VRAM use",
    )
    parser.add_argument(
        "--verify_pretrained_sha256",
        type=parse_bool,
        default=True,
        help="Verify the pinned VideoMAE V2 checkpoint before deserialization",
    )
    parser.add_argument("--drop_path_rate", type=float, default=0.10)
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.90,
        help="Layer-wise learning-rate decay for transformer backbones",
    )
    parser.add_argument("--head_init_scale", type=float, default=0.001)
    parser.add_argument("--freeze_batch_norm", type=parse_bool, default=False)
    parser.add_argument("--rnn_hidden_size", type=int, default=256)
    parser.add_argument("--rnn_n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--train_crop_scale_min", type=float, default=0.60)
    parser.add_argument("--train_crop_scale_max", type=float, default=1.0)
    parser.add_argument("--train_crop_ratio_min", type=float, default=0.75)
    parser.add_argument("--train_crop_ratio_max", type=float, default=1.333)
    parser.add_argument("--color_jitter_brightness", type=float, default=0.20)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.20)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.20)
    parser.add_argument("--color_jitter_hue", type=float, default=0.05)
    parser.add_argument(
        "--train_horizontal_flip_probability",
        type=float,
        default=0.5,
    )
    parser.add_argument("--random_erasing_probability", type=float, default=0.10)
    parser.add_argument("--random_erasing_scale_min", type=float, default=0.02)
    parser.add_argument("--random_erasing_scale_max", type=float, default=0.20)
    parser.add_argument("--random_erasing_ratio_min", type=float, default=0.30)
    parser.add_argument("--random_erasing_ratio_max", type=float, default=3.30)

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for the temporal/classification head",
    )
    parser.add_argument(
        "--backbone_lr_multiplier",
        type=float,
        default=0.10,
        help="Top-backbone learning-rate multiplier after unfreezing",
    )
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=45)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--minimum_lr_factor", type=float, default=0.01)
    parser.add_argument("--minimum_epochs", type=int, default=15)
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help=(
            "Seed used only to create train/validation/test assignments. "
            "Defaults to --seed. Set it explicitly to keep the split fixed "
            "while varying model-training seeds."
        ),
    )
    parser.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=2,
        help="1-based epoch at which to unfreeze the backbone; use 0 to keep it frozen",
    )
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument(
        "--fingerprint_split_audit",
        type=parse_bool,
        default=True,
        help=(
            "Find duplicate candidates with representative frames, confirm them "
            "by hashing every extracted frame, and audit split boundaries"
        ),
    )

    parser.add_argument("--wandb_project", default="hmdb51-video-classification")
    parser.add_argument("--wandb_entity")
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = {
        "n_classes": args.n_classes,
        "fr_per_vid": args.fr_per_vid,
        "temporal_stride": args.temporal_stride,
        "image_size": args.image_size,
        "val_clips": args.val_clips,
        "test_clips": args.test_clips,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "eval_clip_chunk_size": args.eval_clip_chunk_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "n_epochs": args.n_epochs,
    }
    invalid_positive = [
        name for name, value in positive_integer_fields.items() if value <= 0
    ]
    if invalid_positive:
        raise ValueError(
            "These arguments must be positive: " + ", ".join(invalid_positive)
        )
    if args.workers < 0:
        raise ValueError("--workers must be zero or greater")
    if args.learning_rate <= 0 or args.backbone_lr_multiplier <= 0:
        raise ValueError("Learning rates and multipliers must be greater than zero")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be zero or greater")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.unfreeze_epoch < 0 or args.warmup_epochs < 0:
        raise ValueError("Unfreeze and warmup epochs must be zero or greater")
    if args.minimum_epochs < 0 or args.early_stopping_patience < 0:
        raise ValueError("Epoch limits must be zero or greater")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label_smoothing must be in [0, 1)")
    TrainAugmentationConfig(
        crop_scale=(args.train_crop_scale_min, args.train_crop_scale_max),
        crop_ratio=(args.train_crop_ratio_min, args.train_crop_ratio_max),
        brightness=args.color_jitter_brightness,
        contrast=args.color_jitter_contrast,
        saturation=args.color_jitter_saturation,
        hue=args.color_jitter_hue,
        horizontal_flip_probability=args.train_horizontal_flip_probability,
        random_erasing_probability=args.random_erasing_probability,
        random_erasing_scale=(
            args.random_erasing_scale_min,
            args.random_erasing_scale_max,
        ),
        random_erasing_ratio=(
            args.random_erasing_ratio_min,
            args.random_erasing_ratio_max,
        ),
    )
    if not 0 < args.layer_decay <= 1:
        raise ValueError("--layer_decay must be in (0, 1]")
    if not 0 <= args.drop_path_rate < 1:
        raise ValueError("--drop_path_rate must be in [0, 1)")
    if args.head_init_scale <= 0:
        raise ValueError("--head_init_scale must be positive")
    if not 0 < args.minimum_lr_factor <= 1:
        raise ValueError("--minimum_lr_factor must be in (0, 1]")
    if args.split_protocol == "random":
        if (
            args.train_size <= 0
            or args.test_size <= 0
            or args.train_size + args.test_size >= 1
        ):
            raise ValueError(
                "Random split ratios must be positive and train+test must be < 1"
            )
    elif not 0 < args.validation_size < 1:
        raise ValueError("--validation_size must be between zero and one")
    if args.architecture == "mvit_v2_s":
        if args.fr_per_vid != 16 or args.image_size != 224:
            raise ValueError(
                "mvit_v2_s uses its pretrained 16-frame, 224x224 input. "
                "Set --fr_per_vid 16 --image_size 224."
            )
    if args.architecture.startswith("videomaev2_"):
        if args.fr_per_vid != 16 or args.image_size != 224:
            raise ValueError(
                "VideoMAE V2 distilled checkpoints require 16 frames at "
                "224x224. Set --fr_per_vid 16 --image_size 224."
            )
        if not args.pretrained and args.mode == "train":
            print(
                "Warning: VideoMAE V2 is being trained without the distilled "
                "K710 checkpoint; this is unlikely to be competitive on HMDB51."
            )


def _sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest for a metadata file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_to_json(sample: tuple[str, int] | list[Any], frame_root: Path) -> list[Any]:
    path = Path(sample[0])
    try:
        stored_path = str(path.resolve().relative_to(frame_root.resolve()))
    except ValueError:
        stored_path = str(path)
    return [stored_path, int(sample[1])]


def _sample_from_json(sample: list[Any], frame_root: Path) -> tuple[str, int]:
    path = Path(str(sample[0]))
    if not path.is_absolute():
        path = frame_root / path
    return str(path), int(sample[1])


def _save_splits(
    path: Path,
    train_split,
    val_split,
    test_split,
    label_dict: dict[str, int],
    frame_dir: str,
    metadata: dict[str, Any],
) -> None:
    frame_root = Path(frame_dir)
    payload = {
        "path_format": "relative_to_frame_dir",
        "train": [_sample_to_json(sample, frame_root) for sample in train_split],
        "val": [_sample_to_json(sample, frame_root) for sample in val_split],
        "test": [_sample_to_json(sample, frame_root) for sample in test_split],
        "label_dict": label_dict,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_splits(path: Path, frame_dir: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame_root = Path(frame_dir)
    for split_name in ("train", "val", "test"):
        payload[split_name] = [
            _sample_from_json(sample, frame_root) for sample in payload[split_name]
        ]
    payload["label_dict"] = {
        str(name): int(label) for name, label in payload["label_dict"].items()
    }
    return payload


def _validate_label_mapping(
    label_dict: dict[str, int],
    splits: tuple[list[Any], list[Any], list[Any]],
    n_classes: int,
) -> None:
    """Require contiguous class IDs and labels that agree with the mapping."""
    expected = set(range(n_classes))
    actual = set(label_dict.values())
    if actual != expected:
        raise ValueError(
            "Class IDs must be contiguous from 0 to n_classes-1; "
            f"found {sorted(actual)}"
        )
    for split_name, samples in zip(("train", "val", "test"), splits):
        invalid = [sample for sample in samples if int(sample[1]) not in expected]
        if invalid:
            raise ValueError(
                f"{split_name} contains labels outside the class mapping: "
                f"{invalid[:3]}"
            )


def _build_model(args: argparse.Namespace, training: bool) -> nn.Module:
    # Evaluation reconstructs the architecture without downloading source weights;
    # the selected HMDB51 checkpoint immediately overwrites the initialized state.
    load_source_weights = bool(args.pretrained and training)
    common = {
        "n_classes": args.n_classes,
        "dropout_rate": args.dropout,
        "pretrained": load_source_weights,
        "freeze_backbone": training,
        "freeze_batch_norm": args.freeze_batch_norm,
    }
    if args.architecture == "r2plus1d_18":
        return VideoR2Plus1D(**common)
    if args.architecture == "mvit_v2_s":
        return VideoMViTV2(**common)
    if args.architecture.startswith("videomaev2_"):
        return VideoMAEV2Distilled(
            architecture=args.architecture,
            n_classes=args.n_classes,
            dropout_rate=args.dropout,
            pretrained=load_source_weights,
            freeze_backbone=training,
            drop_path_rate=args.drop_path_rate,
            gradient_checkpointing=args.gradient_checkpointing,
            pretrained_checkpoint=(
                args.pretrained_checkpoint if load_source_weights else None
            ),
            pretrained_cache_dir=(
                args.pretrained_cache_dir if load_source_weights else None
            ),
            verify_pretrained_sha256=args.verify_pretrained_sha256,
            head_init_scale=args.head_init_scale,
            clip_forward_batch_size=args.eval_clip_chunk_size,
        )
    return LRCN(
        hidden_size=args.rnn_hidden_size,
        n_layers=args.rnn_n_layers,
        dropout_rate=args.dropout,
        n_classes=args.n_classes,
        pretrained=load_source_weights,
        cnn_model=args.cnn_backbone,
        bidirectional=True,
        freeze_backbone=training,
        freeze_batch_norm=args.freeze_batch_norm,
    )


def _learning_rate_factor(
    epoch_index: int,
    total_epochs: int,
    warmup_epochs: int,
    minimum_factor: float,
) -> float:
    """Return a shared warmup/cosine multiplier that preserves LR group ratios."""
    warmup_epochs = min(max(warmup_epochs, 0), max(total_epochs - 1, 0))
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        if warmup_epochs == 1:
            return 1.0
        return 0.1 + 0.9 * epoch_index / (warmup_epochs - 1)

    decay_epochs = max(total_epochs - warmup_epochs, 1)
    if decay_epochs == 1:
        progress = 1.0
    else:
        progress = (epoch_index - warmup_epochs) / (decay_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_factor + (1.0 - minimum_factor) * cosine


def _print_dataset_audit(audit: dict[str, float | int]) -> None:
    print("Dataset temporal audit:")
    print(
        "  frames per video: "
        f"min={audit['minimum_frames']}, median={audit['median_frames']:.1f}, "
        f"max={audit['maximum_frames']}"
    )
    print(f"  requested temporal span: {audit['required_span']} source frames")
    print(
        "  videos using uniform full-duration sampling: "
        f"{audit['videos_using_uniform_sampling']}/{audit['videos']} "
        f"({audit['uniform_sampling_fraction']:.1%})"
    )
    print(
        "  videos with fewer source frames than output frames: "
        f"{audit['videos_with_fewer_frames_than_output']}"
    )


def _build_datasets_and_loaders(
    args: argparse.Namespace,
    train_split,
    val_split,
    test_split,
):
    train_augmentation = TrainAugmentationConfig(
        crop_scale=(args.train_crop_scale_min, args.train_crop_scale_max),
        crop_ratio=(args.train_crop_ratio_min, args.train_crop_ratio_max),
        brightness=args.color_jitter_brightness,
        contrast=args.color_jitter_contrast,
        saturation=args.color_jitter_saturation,
        hue=args.color_jitter_hue,
        horizontal_flip_probability=args.train_horizontal_flip_probability,
        random_erasing_probability=args.random_erasing_probability,
        random_erasing_scale=(
            args.random_erasing_scale_min,
            args.random_erasing_scale_max,
        ),
        random_erasing_ratio=(
            args.random_erasing_ratio_min,
            args.random_erasing_ratio_max,
        ),
    )
    train_transform, eval_transform = compose_data_transforms(
        args.architecture,
        args.image_size,
        train_augmentation=train_augmentation,
    )
    datasets = {
        "train": VideoDataset(
            train_split,
            args.fr_per_vid,
            train_transform,
            training=True,
            eval_clips=1,
            temporal_stride=args.temporal_stride,
        ),
        "val": VideoDataset(
            val_split,
            args.fr_per_vid,
            eval_transform,
            eval_clips=args.val_clips,
            temporal_stride=args.temporal_stride,
        ),
        "test": VideoDataset(
            test_split,
            args.fr_per_vid,
            eval_transform,
            eval_clips=args.test_clips,
            temporal_stride=args.temporal_stride,
        ),
    }
    loaders = compose_dataloaders(
        datasets["train"],
        datasets["val"],
        datasets["test"],
        args.batch_size,
        args.workers,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    return datasets, loaders


def _create_training_splits(args: argparse.Namespace):
    samples, label_dict = load_dataset(args.frame_dir)
    if len(label_dict) != args.n_classes:
        raise ValueError(
            f"Found {len(label_dict)} classes but --n_classes={args.n_classes}"
        )

    official_duplicate_audit: dict[str, Any] | None = None
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.split_protocol == "official":
        if not args.official_split_dir:
            raise ValueError("--official_split_dir is required for official splits")
        (
            train_split,
            val_split,
            test_split,
            official_duplicate_audit,
        ) = official_dataset_split(
            samples,
            label_dict,
            args.official_split_dir,
            split_number=args.official_split_number,
            validation_ratio=args.validation_size,
            seed=split_seed,
            duplicate_policy=args.official_duplicate_policy,
            return_audit=True,
        )
    else:
        train_split, val_split, test_split = dataset_split(
            samples,
            args.train_size,
            args.test_size,
            split_seed,
        )

    integrity_policy = (
        "report"
        if args.split_protocol == "official"
        and args.official_duplicate_policy == "allow"
        else "error"
    )
    integrity = validate_split_integrity(
        train_split,
        val_split,
        test_split,
        check_fingerprints=args.fingerprint_split_audit,
        cross_split_duplicate_policy=integrity_policy,
    )
    print(
        f"Split sizes: train={len(train_split)}, val={len(val_split)}, "
        f"test={len(test_split)}"
    )
    print("Split integrity audit passed:", integrity)
    print("Training-split temporal statistics only:")
    _print_dataset_audit(
        dataset_audit(train_split, args.fr_per_vid, args.temporal_stride)
    )
    metadata: dict[str, Any] = {
        "split_protocol": args.split_protocol,
        "seed": split_seed,
        "split_seed": split_seed,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "official_split_number": args.official_split_number,
        "official_duplicate_policy": (
            args.official_duplicate_policy
            if args.split_protocol == "official"
            else None
        ),
        "fingerprint_split_audit": args.fingerprint_split_audit,
        "integrity": integrity,
    }
    if official_duplicate_audit is not None:
        metadata["official_duplicate_audit"] = official_duplicate_audit
    return train_split, val_split, test_split, label_dict, metadata

def _save_history(path: Path, loss_history, accuracy_history) -> None:
    path.write_text(
        json.dumps(
            {"loss": loss_history, "accuracy": accuracy_history},
            indent=2,
        ),
        encoding="utf-8",
    )


def _optimizer_parameter_groups(
    model: nn.Module,
    learning_rate: float,
    backbone_lr_multiplier: float,
    weight_decay: float,
    layer_decay: float = 1.0,
) -> list[dict[str, Any]]:
    """Build AdamW groups with no-decay tensors and optional transformer LLRD."""
    if not 0 < layer_decay <= 1:
        raise ValueError("layer_decay must be in (0, 1]")

    has_layer_mapping = all(
        hasattr(model, attribute)
        for attribute in ("optimizer_layer_id", "optimizer_num_layers")
    )
    number_of_layers = (
        int(model.optimizer_num_layers()) if has_layer_mapping else 1
    )
    if number_of_layers <= 0:
        raise ValueError("optimizer_num_layers() must return a positive integer")

    buckets: dict[tuple[str, bool, int | None, float], list[nn.Parameter]] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        role = "backbone" if name.startswith("base_model.") else "head"
        use_decay = (
            parameter.ndim > 1
            and not name.endswith(".bias")
            and not name.endswith(".scale")
        )
        layer_id: int | None = None
        lr_scale = 1.0
        if role == "backbone" and has_layer_mapping:
            layer_id = model.optimizer_layer_id(name)
            if layer_id is None or not 0 <= int(layer_id) < number_of_layers:
                raise ValueError(
                    f"Invalid optimizer layer ID {layer_id!r} for {name}; "
                    f"expected 0..{number_of_layers - 1}"
                )
            layer_id = int(layer_id)
            lr_scale = layer_decay ** (number_of_layers - 1 - layer_id)
        key = (role, use_decay, layer_id, lr_scale)
        buckets.setdefault(key, []).append(parameter)

    groups: list[dict[str, Any]] = []
    sorted_buckets = sorted(
        buckets.items(),
        key=lambda item: (
            0 if item[0][0] == "backbone" else 1,
            -1 if item[0][2] is None else item[0][2],
            0 if item[0][1] else 1,
        ),
    )
    for (role, use_decay, layer_id, lr_scale), parameters in sorted_buckets:
        role_lr = (
            learning_rate * backbone_lr_multiplier * lr_scale
            if role == "backbone"
            else learning_rate
        )
        layer_label = "head" if layer_id is None else f"layer_{layer_id:02d}"
        groups.append(
            {
                "params": parameters,
                "lr": role_lr,
                "weight_decay": weight_decay if use_decay else 0.0,
                "role": role,
                "layer_id": layer_id,
                "lr_scale": lr_scale,
                "decay": "decay" if use_decay else "no_decay",
                "name": f"{role}_{layer_label}_{'decay' if use_decay else 'no_decay'}",
            }
        )
    return groups


def _augmentation_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Return a JSON-serializable description of training augmentation."""
    return {
        "crop_scale": [args.train_crop_scale_min, args.train_crop_scale_max],
        "crop_ratio": [args.train_crop_ratio_min, args.train_crop_ratio_max],
        "brightness": args.color_jitter_brightness,
        "contrast": args.color_jitter_contrast,
        "saturation": args.color_jitter_saturation,
        "hue": args.color_jitter_hue,
        "horizontal_flip_probability": (
            args.train_horizontal_flip_probability
        ),
        "random_erasing_probability": args.random_erasing_probability,
        "random_erasing_scale": [
            args.random_erasing_scale_min,
            args.random_erasing_scale_max,
        ],
        "random_erasing_ratio": [
            args.random_erasing_ratio_min,
            args.random_erasing_ratio_max,
        ],
    }


def _validate_checkpoint_configuration(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    split_path: Path,
) -> None:
    """Reject silent evaluation under a different model/input/split configuration."""
    checkpoint_architecture = checkpoint.get("architecture")
    if checkpoint_architecture and checkpoint_architecture != args.architecture:
        raise ValueError(
            f"Checkpoint architecture is {checkpoint_architecture}, but "
            f"--architecture={args.architecture}"
        )

    expected = {
        "n_classes": args.n_classes,
        "fr_per_vid": args.fr_per_vid,
        "image_size": args.image_size,
        "temporal_stride": args.temporal_stride,
    }
    saved = checkpoint.get("model_config", {})
    mismatches = {
        key: (saved[key], value)
        for key, value in expected.items()
        if key in saved and saved[key] != value
    }
    if mismatches:
        formatted = ", ".join(
            f"{key}: checkpoint={old!r}, command={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(f"Checkpoint/input configuration mismatch: {formatted}")

    saved_split_hash = checkpoint.get("split_sha256")
    if saved_split_hash:
        current_split_hash = _sha256_file(split_path)
        if current_split_hash != saved_split_hash:
            raise ValueError(
                "The supplied split file is not the split file used to select this "
                "checkpoint (SHA-256 mismatch)."
            )


def _load_and_audit_existing_splits(
    args: argparse.Namespace,
    split_path: Path,
):
    """Load a saved split file, verify its metadata, and rerun integrity checks."""
    if not split_path.is_file():
        purpose = (
            "reuse in train mode"
            if args.mode == "train"
            else "evaluation"
        )
        raise FileNotFoundError(
            f"Split metadata required for {purpose} was not found: {split_path}"
        )
    splits = _load_splits(split_path, args.frame_dir)
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]
    label_dict = splits["label_dict"]
    if len(label_dict) != args.n_classes:
        raise ValueError(
            f"Split file contains {len(label_dict)} classes but "
            f"--n_classes={args.n_classes}"
        )
    metadata = splits.get("metadata", {})
    saved_protocol = metadata.get("split_protocol")
    if saved_protocol and saved_protocol != args.split_protocol:
        raise ValueError(
            f"Saved split protocol is {saved_protocol!r}, but the command uses "
            f"--split_protocol={args.split_protocol!r}"
        )
    saved_split_number = metadata.get("official_split_number")
    if (
        args.split_protocol == "official"
        and saved_split_number is not None
        and int(saved_split_number) != args.official_split_number
    ):
        raise ValueError(
            f"Saved official split number is {saved_split_number}, but the "
            f"command uses {args.official_split_number}"
        )
    saved_split_seed = metadata.get("split_seed", metadata.get("seed"))
    if (
        args.mode == "train"
        and args.split_seed is not None
        and saved_split_seed is not None
        and int(saved_split_seed) != args.split_seed
    ):
        raise ValueError(
            f"Saved split seed is {saved_split_seed}, but --split_seed="
            f"{args.split_seed}. Use the saved value when reusing a split."
        )
    saved_duplicate_policy = metadata.get("official_duplicate_policy", "error")
    integrity = validate_split_integrity(
        train_split,
        val_split,
        test_split,
        check_fingerprints=args.fingerprint_split_audit,
        cross_split_duplicate_policy=(
            "report" if saved_duplicate_policy == "allow" else "error"
        ),
    )
    print(
        f"Loaded split sizes: train={len(train_split)}, val={len(val_split)}, "
        f"test={len(test_split)}"
    )
    print("Loaded split integrity audit passed:", integrity)
    if args.mode == "train":
        print("Training-split temporal statistics only:")
        _print_dataset_audit(
            dataset_audit(train_split, args.fr_per_vid, args.temporal_stride)
        )
        metadata = dict(metadata)
        metadata["integrity_recheck"] = integrity
        metadata["reused_split_sha256"] = _sha256_file(split_path)
    return train_split, val_split, test_split, label_dict, metadata


def main(args: argparse.Namespace) -> None:
    """Train using validation only, or explicitly evaluate a saved checkpoint."""
    _validate_args(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.split_file) if args.split_file else output_dir / "splits.json"

    if args.mode == "train" and not args.reuse_existing_split:
        train_split, val_split, test_split, label_dict, split_metadata = (
            _create_training_splits(args)
        )
        _save_splits(
            split_path,
            train_split,
            val_split,
            test_split,
            label_dict,
            args.frame_dir,
            split_metadata,
        )
    else:
        if args.mode == "eval" and not args.ckpt:
            raise ValueError("--ckpt is required in eval mode")
        (
            train_split,
            val_split,
            test_split,
            label_dict,
            split_metadata,
        ) = _load_and_audit_existing_splits(args, split_path)
        if args.mode == "train":
            print(
                "Reusing the exact saved split for this architecture comparison: "
                f"{split_path}"
            )

    official_audit = split_metadata.get("official_duplicate_audit")
    if official_audit is not None:
        official_audit_path = output_dir / "official_duplicate_audit.json"
        official_audit_path.write_text(
            json.dumps(official_audit, indent=2),
            encoding="utf-8",
        )
        print(f"Official duplicate audit: {official_audit_path}")

    _validate_label_mapping(
        label_dict,
        (train_split, val_split, test_split),
        args.n_classes,
    )
    if args.mode == "train":
        run_config = vars(args).copy()
        run_config["resolved_split_seed"] = split_metadata.get(
            "split_seed",
            args.seed if args.split_seed is None else args.split_seed,
        )
        run_config["split_reused"] = bool(args.reuse_existing_split)
        run_config["train_augmentation"] = _augmentation_metadata(args)
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _, loaders = _build_datasets_and_loaders(
        args,
        train_split,
        val_split,
        test_split,
    )
    model = _build_model(args, training=args.mode == "train").to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.mode == "train":
        wandb_run = None
        if args.wandb_mode != "disabled":
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                config=vars(args),
            )
            wandb.watch(model, log="gradients", log_freq=100)

        parameter_groups = _optimizer_parameter_groups(
            model,
            args.learning_rate,
            args.backbone_lr_multiplier,
            args.weight_decay,
            layer_decay=(
                args.layer_decay
                if args.architecture.startswith("videomaev2_")
                else 1.0
            ),
        )
        optimizer = optim.AdamW(parameter_groups)
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        backbone_lrs = [
            float(group["lr"])
            for group in parameter_groups
            if group.get("role") == "backbone"
        ]
        print(
            "Base optimizer learning rates: "
            f"backbone top={max(backbone_lrs, default=float('nan')):.2e}, "
            f"backbone bottom={min(backbone_lrs, default=float('nan')):.2e}, "
            f"head={args.learning_rate:.2e}, "
            f"layer decay={args.layer_decay if args.architecture.startswith('videomaev2_') else 1.0:.3f}"
        )
        print(
            f"Mini-batch={args.batch_size}, accumulation="
            f"{args.gradient_accumulation_steps}, effective batch="
            f"{effective_batch_size}"
        )
        print(
            "Training augmentation: "
            + json.dumps(_augmentation_metadata(args), sort_keys=True)
        )
        print(
            "Training seed="
            f"{args.seed}, split seed="
            f"{args.seed if args.split_seed is None else args.split_seed}"
        )
        if len(optimizer.param_groups) <= 10:
            groups_to_print = optimizer.param_groups
        else:
            # LLRD creates two groups per transformer depth. Print a compact
            # summary while retaining full group metadata in the optimizer.
            groups_to_print = [
                group
                for index, group in enumerate(optimizer.param_groups)
                if index in {0, 1, len(optimizer.param_groups) - 4,
                             len(optimizer.param_groups) - 3,
                             len(optimizer.param_groups) - 2,
                             len(optimizer.param_groups) - 1}
            ]
        for group in groups_to_print:
            parameter_count = sum(parameter.numel() for parameter in group["params"])
            layer_text = (
                "" if group.get("layer_id") is None
                else f"/layer={group['layer_id']}"
            )
            print(
                f"  optimizer group {group['role']}/{group['decay']}{layer_text}: "
                f"parameters={parameter_count:,}, lr={group['lr']:.2e}, "
                f"weight_decay={group['weight_decay']:.2e}"
            )
        if len(optimizer.param_groups) > len(groups_to_print):
            print(
                f"  ... {len(optimizer.param_groups) - len(groups_to_print)} "
                "intermediate layer-wise optimizer groups omitted from display"
            )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _learning_rate_factor(
                epoch,
                args.n_epochs,
                args.warmup_epochs,
                args.minimum_lr_factor,
            ),
        )

        checkpoint_metadata = {
            "architecture": args.architecture,
            "model_config": {
                "n_classes": args.n_classes,
                "dropout": args.dropout,
                "fr_per_vid": args.fr_per_vid,
                "image_size": args.image_size,
                "temporal_stride": args.temporal_stride,
                "freeze_batch_norm": args.freeze_batch_norm,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "drop_path_rate": args.drop_path_rate,
                "gradient_checkpointing": args.gradient_checkpointing,
                "layer_decay": args.layer_decay,
                "head_init_scale": args.head_init_scale,
                "eval_clip_chunk_size": args.eval_clip_chunk_size,
            },
            "pretraining": (
                model.pretrained_metadata()
                if hasattr(model, "pretrained_metadata")
                else None
            ),
            "train_augmentation": _augmentation_metadata(args),
            "training_seed": args.seed,
            "split_seed": split_metadata.get(
                "split_seed",
                args.seed if args.split_seed is None else args.split_seed,
            ),
            "split_reused": bool(args.reuse_existing_split),
            "split_sha256": _sha256_file(split_path),
            "split_protocol": args.split_protocol,
            "official_duplicate_policy": (
                args.official_duplicate_policy
                if args.split_protocol == "official"
                else None
            ),
        }
        model, loss_history, accuracy_history = train(
            loaders,
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            str(output_dir),
            args.n_epochs,
            wandb_run,
            args.unfreeze_epoch,
            args.early_stopping_patience,
            minimum_epochs=args.minimum_epochs,
            amp=args.amp,
            checkpoint_metadata=checkpoint_metadata,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        _save_history(output_dir / "training_history.json", loss_history, accuracy_history)

        best_val_accuracy = max(accuracy_history["val"], default=float("nan"))
        print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2%}")
        print(f"Best checkpoint: {output_dir / 'best_model_wts.pt'}")
        print(f"Split metadata: {split_path}")
        print(
            "The test set was not evaluated. Run --mode eval once after model "
            "and hyperparameter selection is complete."
        )
        if wandb_run is not None:
            wandb_run.summary["best_val_accuracy"] = best_val_accuracy
            wandb_run.finish()
        return

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        _validate_checkpoint_configuration(checkpoint, args, split_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    test_loss, test_accuracy, targets, predictions = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        horizontal_flip_tta=args.flip_tta,
        amp=args.amp,
    )
    names = [name for name, _ in sorted(label_dict.items(), key=lambda item: item[1])]
    report_text = classification_report(
        targets.numpy(),
        predictions.numpy(),
        labels=list(range(len(names))),
        target_names=names,
        zero_division=0,
    )
    report_dict = classification_report(
        targets.numpy(),
        predictions.numpy(),
        labels=list(range(len(names))),
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(report_text)

    (output_dir / "test_metrics.json").write_text(
        json.dumps(
            {"test_loss": test_loss, "test_accuracy": test_accuracy},
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(args_parser())
