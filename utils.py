"""Clip-consistent transforms, reproducibility, and DataLoader helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as vision_functional

from video_datasets import VideoDataset, collate_fn_video


@dataclass(frozen=True)
class TransformProfile:
    """Preprocessing values associated with a pretrained model family."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize_size: int
    interpolation: InterpolationMode = InterpolationMode.BILINEAR


@dataclass(frozen=True)
class TrainAugmentationConfig:
    """Clip-consistent training augmentation configuration.

    All sampled parameters, including the optional erased rectangle, are shared
    by every frame in a clip. This avoids artificial frame-to-frame flicker.
    """

    crop_scale: tuple[float, float] = (0.8, 1.0)
    crop_ratio: tuple[float, float] = (0.9, 1.1)
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.03
    horizontal_flip_probability: float = 0.5
    random_erasing_probability: float = 0.0
    random_erasing_scale: tuple[float, float] = (0.02, 0.20)
    random_erasing_ratio: tuple[float, float] = (0.3, 3.3)

    def __post_init__(self) -> None:
        scale_min, scale_max = self.crop_scale
        ratio_min, ratio_max = self.crop_ratio
        erase_scale_min, erase_scale_max = self.random_erasing_scale
        erase_ratio_min, erase_ratio_max = self.random_erasing_ratio
        if not 0 < scale_min <= scale_max <= 1:
            raise ValueError(
                "crop_scale must satisfy 0 < minimum <= maximum <= 1"
            )
        if not 0 < ratio_min <= ratio_max:
            raise ValueError(
                "crop_ratio must satisfy 0 < minimum <= maximum"
            )
        for name, value in (
            ("brightness", self.brightness),
            ("contrast", self.contrast),
            ("saturation", self.saturation),
        ):
            if value < 0:
                raise ValueError(f"{name} must be zero or greater")
        if not 0 <= self.hue <= 0.5:
            raise ValueError("hue must be in [0, 0.5]")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if not 0 <= self.random_erasing_probability <= 1:
            raise ValueError("random_erasing_probability must be in [0, 1]")
        if not 0 < erase_scale_min <= erase_scale_max <= 1:
            raise ValueError(
                "random_erasing_scale must satisfy 0 < minimum <= maximum <= 1"
            )
        if not 0 < erase_ratio_min <= erase_ratio_max:
            raise ValueError(
                "random_erasing_ratio must satisfy 0 < minimum <= maximum"
            )


def transform_profile(architecture: str, image_size: int) -> TransformProfile:
    """Return normalization and evaluation resize settings for an architecture."""
    if architecture == "r2plus1d_18":
        return TransformProfile(
            mean=(0.43216, 0.394666, 0.37645),
            std=(0.22803, 0.22145, 0.216989),
            resize_size=128 if image_size == 112 else image_size + 32,
        )
    if architecture == "mvit_v2_s":
        return TransformProfile(
            mean=(0.45, 0.45, 0.45),
            std=(0.225, 0.225, 0.225),
            resize_size=256 if image_size == 224 else image_size + 32,
        )
    if architecture in {
        "videomaev2_vit_s_distilled",
        "videomaev2_vit_b_distilled",
    }:
        return TransformProfile(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            resize_size=224 if image_size == 224 else image_size,
            interpolation=InterpolationMode.BICUBIC,
        )
    return TransformProfile(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        resize_size=image_size + 32,
    )


def _sample_erasing_rectangle(
    height: int,
    width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
) -> tuple[int, int, int, int] | None:
    """Sample one Random-Erasing rectangle, or return ``None`` after retries."""
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * random.uniform(scale[0], scale[1])
        aspect_ratio = math.exp(random.uniform(*log_ratio))
        erase_height = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_width = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < erase_height < height and 0 < erase_width < width:
            top = random.randint(0, height - erase_height)
            left = random.randint(0, width - erase_width)
            return top, left, erase_height, erase_width
    return None


class ClipTrainTransform:
    """Apply one random augmentation consistently to all clip frames."""

    def __init__(
        self,
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        augmentation: TrainAugmentationConfig | None = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.image_size = image_size
        self.mean = list(mean)
        self.std = list(std)
        self.augmentation = augmentation or TrainAugmentationConfig()
        self.interpolation = interpolation
        self.jitter = ColorJitter(
            brightness=self.augmentation.brightness,
            contrast=self.augmentation.contrast,
            saturation=self.augmentation.saturation,
            hue=self.augmentation.hue,
        )

    def __call__(self, images: list[Image.Image]) -> list[Tensor]:
        if not images:
            raise ValueError("ClipTrainTransform requires at least one frame")
        crop = RandomResizedCrop.get_params(
            images[0],
            scale=self.augmentation.crop_scale,
            ratio=self.augmentation.crop_ratio,
        )
        top, left, height, width = crop
        flip = random.random() < self.augmentation.horizontal_flip_probability
        jitter_order, brightness, contrast, saturation, hue = self.jitter.get_params(
            self.jitter.brightness,
            self.jitter.contrast,
            self.jitter.saturation,
            self.jitter.hue,
        )

        output: list[Tensor] = []
        for image in images:
            transformed = vision_functional.resized_crop(
                image,
                top,
                left,
                height,
                width,
                [self.image_size, self.image_size],
                self.interpolation,
                antialias=True,
            )
            if flip:
                transformed = vision_functional.hflip(transformed)
            for operation in jitter_order:
                operation_index = int(operation)
                if operation_index == 0 and brightness is not None:
                    transformed = vision_functional.adjust_brightness(
                        transformed, brightness
                    )
                elif operation_index == 1 and contrast is not None:
                    transformed = vision_functional.adjust_contrast(
                        transformed, contrast
                    )
                elif operation_index == 2 and saturation is not None:
                    transformed = vision_functional.adjust_saturation(
                        transformed, saturation
                    )
                elif operation_index == 3 and hue is not None:
                    transformed = vision_functional.adjust_hue(transformed, hue)
            tensor = vision_functional.to_tensor(transformed)
            output.append(
                vision_functional.normalize(tensor, self.mean, self.std)
            )

        if (
            self.augmentation.random_erasing_probability > 0
            and random.random() < self.augmentation.random_erasing_probability
        ):
            rectangle = _sample_erasing_rectangle(
                self.image_size,
                self.image_size,
                self.augmentation.random_erasing_scale,
                self.augmentation.random_erasing_ratio,
            )
            if rectangle is not None:
                erase_top, erase_left, erase_height, erase_width = rectangle
                # One shared normalized-space fill avoids temporal flicker.
                for tensor in output:
                    tensor[
                        :,
                        erase_top : erase_top + erase_height,
                        erase_left : erase_left + erase_width,
                    ] = 0.0
        return output


class ClipEvalTransform:
    """Apply deterministic preprocessing matching the selected pretrained model."""

    def __init__(
        self,
        image_size: int,
        resize_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.image_size = image_size
        self.resize_size = resize_size
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def __call__(self, images: list[Image.Image]) -> list[Tensor]:
        output: list[Tensor] = []
        for image in images:
            transformed = vision_functional.resize(
                image,
                self.resize_size,
                interpolation=self.interpolation,
                antialias=True,
            )
            transformed = vision_functional.center_crop(
                transformed,
                [self.image_size, self.image_size],
            )
            tensor = vision_functional.to_tensor(transformed)
            output.append(vision_functional.normalize(tensor, self.mean, self.std))
        return output


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and select deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compose_data_transforms(
    architecture: str,
    image_size: int,
    train_augmentation: TrainAugmentationConfig | None = None,
) -> tuple[ClipTrainTransform, ClipEvalTransform]:
    """Return clip-consistent train and deterministic evaluation transforms."""
    profile = transform_profile(architecture, image_size)
    return (
        ClipTrainTransform(
            image_size,
            profile.mean,
            profile.std,
            augmentation=train_augmentation,
            interpolation=profile.interpolation,
        ),
        ClipEvalTransform(
            image_size,
            profile.resize_size,
            profile.mean,
            profile.std,
            interpolation=profile.interpolation,
        ),
    )


def _loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def compose_dataloaders(
    train_dataset: VideoDataset,
    validation_dataset: VideoDataset,
    test_dataset: VideoDataset,
    batch_size: int,
    workers: int = 4,
    eval_batch_size: int | None = None,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Build reproducible train, validation, and test DataLoaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if eval_batch_size is None:
        eval_batch_size = batch_size
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")

    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn_video,
        "persistent_workers": workers > 0,
        "worker_init_fn": _seed_worker,
    }
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=_loader_generator(seed),
            **common,
        ),
        "val": DataLoader(
            validation_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            generator=_loader_generator(seed + 1),
            **common,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            generator=_loader_generator(seed + 2),
            **common,
        ),
    }
