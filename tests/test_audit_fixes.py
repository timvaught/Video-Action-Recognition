"""Regression tests for leakage, split integrity, and optimization fixes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from run import (
    _create_training_splits,
    _optimizer_parameter_groups,
    _validate_checkpoint_configuration,
)
from models import VideoMAEV2Distilled, _VideoBackboneClassifier
from videomaev2 import (
    VideoMAEV2VisionTransformer,
    _extract_tensor_mapping,
    _match_checkpoint_state_dict,
    checkpoint_spec_for_architecture,
)
from test import get_test_report
from train import _run_epoch
from utils import (
    TrainAugmentationConfig,
    compose_data_transforms,
    transform_profile,
)
from video_datasets import (
    decontaminate_official_train_test,
    load_dataset,
    official_dataset_split,
    resolve_official_split_dir,
    validate_split_integrity,
)


class _TinyDataset(Dataset):
    def __init__(self, count: int = 5) -> None:
        self.features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )[:count]
        self.labels = torch.tensor([0, 1, 0, 1, 1])[:count]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], torch.tensor(1)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, videos, _lengths):
        return self.linear(videos)


class _CountingSGD(SGD):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _ToyTransferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        self.classifier = nn.Linear(4, 2)


class _ClipCountingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(videos.shape[0]))
        return videos.mean(dim=(2, 3, 4))


class AuditFixTests(unittest.TestCase):
    def _video(self, root: Path, name: str, pixel: int) -> Path:
        directory = root / name
        directory.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(pixel, pixel, pixel)).save(
            directory / "frame0001.jpg"
        )
        return directory

    def test_fixed_split_seed_is_independent_of_training_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_root = root / "frames"
            split_root = root / "splits"
            split_root.mkdir(parents=True)

            for class_index, class_name in enumerate(("class_a", "class_b")):
                class_root = frame_root / class_name
                class_root.mkdir(parents=True)
                rows = []
                for video_index in range(6):
                    video_name = f"{class_name}_{video_index}"
                    self._video(
                        class_root,
                        video_name,
                        20 + class_index * 20 + video_index,
                    )
                    split_code = 1 if video_index < 4 else 2
                    rows.append(f"{video_name}.avi {split_code}\n")
                (split_root / f"{class_name}_test_split1.txt").write_text(
                    "".join(rows),
                    encoding="utf-8",
                )

            common = dict(
                frame_dir=str(frame_root),
                n_classes=2,
                split_protocol="official",
                official_split_dir=str(split_root),
                official_split_number=1,
                validation_size=0.25,
                split_seed=99,
                official_duplicate_policy="drop_train",
                train_size=0.7,
                test_size=0.15,
                fingerprint_split_audit=True,
                fr_per_vid=1,
                temporal_stride=1,
            )
            first = _create_training_splits(
                SimpleNamespace(seed=1, **common)
            )
            second = _create_training_splits(
                SimpleNamespace(seed=2, **common)
            )

            self.assertEqual(first[:4], second[:4])
            self.assertEqual(first[4], second[4])
            self.assertEqual(first[4]["seed"], 99)
            self.assertEqual(first[4]["split_seed"], 99)

    def test_custom_augmentation_profile_is_clip_consistent(self) -> None:
        config = TrainAugmentationConfig(
            crop_scale=(0.6, 1.0),
            crop_ratio=(0.75, 1.333),
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            horizontal_flip_probability=0.5,
        )
        train_transform, _ = compose_data_transforms(
            "mvit_v2_s",
            224,
            train_augmentation=config,
        )
        self.assertEqual(train_transform.augmentation, config)

        random_frame = Image.new("RGB", (256, 256), color=(80, 100, 120))
        transformed = train_transform([random_frame, random_frame.copy()])
        self.assertTrue(torch.equal(transformed[0], transformed[1]))

    def test_invalid_augmentation_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "crop_scale"):
            TrainAugmentationConfig(crop_scale=(0.9, 0.6))
        with self.assertRaisesRegex(ValueError, "hue"):
            TrainAugmentationConfig(hue=0.6)
        with self.assertRaisesRegex(ValueError, "horizontal_flip_probability"):
            TrainAugmentationConfig(horizontal_flip_probability=1.1)

    def test_gradient_accumulation_steps_at_boundaries(self) -> None:
        torch.manual_seed(7)
        model = _TinyModel()
        loader = DataLoader(_TinyDataset(), batch_size=1, shuffle=False)
        optimizer = _CountingSGD(model.parameters(), lr=0.01)
        _run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            optimizer=optimizer,
            amp=False,
            gradient_accumulation_steps=2,
        )
        self.assertEqual(optimizer.step_calls, 3)

    def test_split_path_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_video = self._video(root, "train_video", 10)
            val_video = self._video(root, "val_video", 20)
            test_video = self._video(root, "test_video", 30)
            with self.assertRaisesRegex(RuntimeError, "Path overlap"):
                validate_split_integrity(
                    [(str(train_video), 0)],
                    [(str(val_video), 0)],
                    [(str(train_video), 0), (str(test_video), 0)],
                    check_fingerprints=False,
                )

    def test_cross_split_exact_content_duplicate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_video = self._video(root, "train_video", 50)
            val_video = self._video(root, "val_video", 60)
            test_video = root / "test_video"
            test_video.mkdir()
            (test_video / "frame0001.jpg").write_bytes(
                (train_video / "frame0001.jpg").read_bytes()
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate crosses split"):
                validate_split_integrity(
                    [(str(train_video), 0)],
                    [(str(val_video), 0)],
                    [(str(test_video), 0)],
                    check_fingerprints=True,
                )

    def test_official_drop_train_preserves_test_and_removes_training_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_duplicate = self._video(root, "train_duplicate", 51)
            train_unique = self._video(root, "train_unique", 52)
            test_duplicate = root / "test_duplicate"
            test_duplicate.mkdir()
            (test_duplicate / "frame0001.jpg").write_bytes(
                (train_duplicate / "frame0001.jpg").read_bytes()
            )

            cleaned_train, cleaned_test, report = decontaminate_official_train_test(
                [(str(train_duplicate), 0), (str(train_unique), 0)],
                [(str(test_duplicate), 0)],
                policy="drop_train",
            )

            self.assertEqual(cleaned_train, [(str(train_unique), 0)])
            self.assertEqual(cleaned_test, [(str(test_duplicate), 0)])
            self.assertEqual(report["cross_split_exact_duplicate_groups"], 1)
            self.assertEqual(report["dropped_train_videos"], 1)

    def test_official_split_integration_drops_cross_boundary_train_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_root = root / "frames"
            class_root = frame_root / "fencing"
            class_root.mkdir(parents=True)
            split_root = root / "splits"
            split_root.mkdir()

            samples = []
            split_rows = []
            train_duplicate = self._video(class_root, "duplicate_train", 61)
            test_duplicate = class_root / "duplicate_test"
            test_duplicate.mkdir()
            (test_duplicate / "frame0001.jpg").write_bytes(
                (train_duplicate / "frame0001.jpg").read_bytes()
            )
            samples.extend(
                [(str(train_duplicate), 0), (str(test_duplicate), 0)]
            )
            split_rows.extend(
                ["duplicate_train.avi 1", "duplicate_test.avi 2"]
            )

            for index in range(4):
                video = self._video(class_root, f"unique_train_{index}", 70 + index)
                samples.append((str(video), 0))
                split_rows.append(f"unique_train_{index}.avi 1")
            unique_test = self._video(class_root, "unique_test", 90)
            samples.append((str(unique_test), 0))
            split_rows.append("unique_test.avi 2")

            (split_root / "fencing_test_split1.txt").write_text(
                "\n".join(split_rows) + "\n",
                encoding="utf-8",
            )

            train, validation, test, audit = official_dataset_split(
                samples,
                {"fencing": 0},
                split_root,
                split_number=1,
                validation_ratio=0.25,
                seed=42,
                duplicate_policy="drop_train",
                return_audit=True,
            )

            selected_paths = {path for path, _ in train + validation}
            self.assertNotIn(str(train_duplicate), selected_paths)
            self.assertIn(str(test_duplicate), {path for path, _ in test})
            self.assertEqual(audit["dropped_train_videos"], 1)
            validate_split_integrity(train, validation, test)

    def test_official_lookup_rejects_ambiguous_frame_directory_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            class_root = root / "frames" / "fencing"
            class_root.mkdir(parents=True)
            split_root = root / "splits"
            split_root.mkdir()

            first = self._video(class_root, "clip", 101)
            second = self._video(class_root, "clip.avi", 102)
            (split_root / "fencing_test_split1.txt").write_text(
                "clip.avi 1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "uniquely match"):
                official_dataset_split(
                    [(str(first), 0), (str(second), 0)],
                    {"fencing": 0},
                    split_root,
                    duplicate_policy="drop_train",
                )

    def test_report_policy_keeps_explicit_benchmark_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_video = self._video(root, "train_video", 53)
            val_video = self._video(root, "val_video", 54)
            test_video = root / "test_video"
            test_video.mkdir()
            (test_video / "frame0001.jpg").write_bytes(
                (train_video / "frame0001.jpg").read_bytes()
            )
            report = validate_split_integrity(
                [(str(train_video), 0)],
                [(str(val_video), 0)],
                [(str(test_video), 0)],
                check_fingerprints=True,
                cross_split_duplicate_policy="report",
            )
            self.assertEqual(report["cross_split_exact_duplicate_groups"], 1)
            self.assertEqual(report["cross_split_exact_duplicate_members"], 2)

    def test_representative_match_is_confirmed_with_all_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_video = root / "train_video"
            test_video = root / "test_video"
            train_video.mkdir()
            test_video.mkdir()
            for frame_index in range(8):
                train_frame = train_video / f"frame{frame_index:04d}.jpg"
                test_frame = test_video / f"frame{frame_index:04d}.jpg"
                Image.new(
                    "RGB",
                    (8, 8),
                    color=(frame_index, frame_index, frame_index),
                ).save(train_frame)
                if frame_index == 6:
                    Image.new("RGB", (8, 8), color=(250, 250, 250)).save(test_frame)
                else:
                    test_frame.write_bytes(train_frame.read_bytes())
            val_video = self._video(root, "val_video", 55)

            report = validate_split_integrity(
                [(str(train_video), 0)],
                [(str(val_video), 0)],
                [(str(test_video), 0)],
                check_fingerprints=True,
            )
            self.assertEqual(report["cross_split_exact_duplicate_groups"], 0)

    def test_exact_duplicate_with_conflicting_labels_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_video = self._video(root, "first_video", 70)
            second_video = root / "second_video"
            second_video.mkdir()
            (second_video / "frame0001.jpg").write_bytes(
                (first_video / "frame0001.jpg").read_bytes()
            )
            val_video = self._video(root, "val_video", 80)
            test_video = self._video(root, "test_video", 90)
            with self.assertRaisesRegex(RuntimeError, "conflicting class labels"):
                validate_split_integrity(
                    [(str(first_video), 0), (str(second_video), 1)],
                    [(str(val_video), 0)],
                    [(str(test_video), 0)],
                    check_fingerprints=True,
                )

    def test_optimizer_excludes_bias_and_one_dimensional_parameters_from_decay(self) -> None:
        model = _ToyTransferModel()
        groups = _optimizer_parameter_groups(model, 3e-4, 0.1, 0.02)
        assignment = {}
        for group in groups:
            for parameter in group["params"]:
                assignment[id(parameter)] = group

        for name, parameter in model.named_parameters():
            group = assignment[id(parameter)]
            expected_role = "backbone" if name.startswith("base_model.") else "head"
            self.assertEqual(group["role"], expected_role)
            expected_decay = 0.0 if parameter.ndim <= 1 or name.endswith(".bias") else 0.02
            self.assertEqual(group["weight_decay"], expected_decay)
            expected_lr = 3e-5 if expected_role == "backbone" else 3e-4
            self.assertAlmostEqual(group["lr"], expected_lr)

    def test_compatibility_report_handles_absent_classes(self) -> None:
        report = get_test_report([0, 0], [0, 0], ["class_0", "class_1"])
        self.assertIn("class_1", report)
        self.assertEqual(report["class_1"]["support"], 0.0)

    def test_official_split_integration_decontaminates_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_root = root / "frames"
            split_root = root / "splits"
            split_root.mkdir(parents=True)

            rows_by_class = {}
            for class_index, class_name in enumerate(("class_a", "class_b")):
                class_dir = frame_root / class_name
                class_dir.mkdir(parents=True)
                rows = []
                for video_index in range(6):
                    video_name = f"{class_name}_{video_index}"
                    video_dir = class_dir / video_name
                    video_dir.mkdir()
                    pixel = class_index * 40 + video_index + 10
                    if class_name == "class_a" and video_index == 4:
                        pixel = 10  # exact copy of class_a_0 across train/test
                    Image.new("RGB", (8, 8), color=(pixel, pixel, pixel)).save(
                        video_dir / "frame0001.jpg"
                    )
                    split_code = 1 if video_index < 4 else 2
                    rows.append(f"{video_name}.avi {split_code}\n")
                rows_by_class[class_name] = rows
                (split_root / f"{class_name}_test_split1.txt").write_text(
                    "".join(rows), encoding="utf-8"
                )

            samples, label_dict = load_dataset(str(frame_root))
            train_split, val_split, test_split, audit = official_dataset_split(
                samples,
                label_dict,
                split_root,
                split_number=1,
                validation_ratio=0.25,
                seed=5,
                duplicate_policy="drop_train",
                return_audit=True,
            )

            selected_paths = {path for path, _ in train_split + val_split}
            self.assertNotIn(str(frame_root / "class_a" / "class_a_0"), selected_paths)
            self.assertIn(
                str(frame_root / "class_a" / "class_a_4"),
                {path for path, _ in test_split},
            )
            self.assertEqual(audit["dropped_train_videos"], 1)
            validate_split_integrity(train_split, val_split, test_split)

    def test_nested_official_split_directory_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "testTrainMulti_7030_splits" / "inner"
            nested.mkdir(parents=True)
            class_names = ["brush_hair", "cartwheel"]
            for class_name in class_names:
                (nested / f"{class_name}_test_split1.txt").write_text(
                    "example.avi 1\n",
                    encoding="utf-8",
                )

            resolved = resolve_official_split_dir(root, class_names, split_number=1)
            self.assertEqual(resolved, nested.resolve())

    def test_incomplete_official_split_directory_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "brush_hair_test_split1.txt").write_text(
                "example.avi 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Could not locate a complete extracted HMDB51 official split directory",
            ):
                resolve_official_split_dir(
                    root,
                    ["brush_hair", "cartwheel"],
                    split_number=1,
                )

    def test_checkpoint_rejects_temporal_or_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            split_path = Path(temporary) / "splits.json"
            split_path.write_text("original", encoding="utf-8")
            checkpoint = {
                "architecture": "mvit_v2_s",
                "model_config": {
                    "n_classes": 51,
                    "fr_per_vid": 16,
                    "image_size": 224,
                    "temporal_stride": 4,
                },
            }
            args = SimpleNamespace(
                architecture="mvit_v2_s",
                n_classes=51,
                fr_per_vid=16,
                image_size=224,
                temporal_stride=2,
            )
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                _validate_checkpoint_configuration(checkpoint, args, split_path)


    def test_videomaev2_tiny_encoder_matches_prefixed_checkpoint(self) -> None:
        model = VideoMAEV2VisionTransformer(
            img_size=32,
            patch_size=16,
            num_classes=5,
            embed_dim=32,
            depth=2,
            num_heads=4,
            num_frames=4,
            tubelet_size=2,
            drop_path_rate=0.0,
            gradient_checkpointing=False,
        )
        source = {
            f"module.{name}": tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        extracted = _extract_tensor_mapping({"module": source})
        matched, report = _match_checkpoint_state_dict(model, extracted)
        self.assertGreater(report["matched_backbone_parameter_fraction"], 0.999)
        self.assertFalse(report["collisions"])
        incompatible = model.load_state_dict(matched, strict=False)
        self.assertFalse(incompatible.unexpected_keys)

        inputs = torch.randn(2, 3, 4, 32, 32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 5))

    def test_videomaev2_distilled_layerwise_lr_decay(self) -> None:
        model = VideoMAEV2Distilled(
            architecture="videomaev2_vit_s_distilled",
            n_classes=3,
            pretrained=False,
            freeze_backbone=True,
            gradient_checkpointing=False,
            drop_path_rate=0.0,
            dropout_rate=0.0,
        )
        groups = _optimizer_parameter_groups(
            model,
            learning_rate=2e-4,
            backbone_lr_multiplier=0.25,
            weight_decay=0.05,
            layer_decay=0.75,
        )
        backbone = [group for group in groups if group["role"] == "backbone"]
        head = [group for group in groups if group["role"] == "head"]
        self.assertTrue(backbone)
        self.assertTrue(head)
        self.assertAlmostEqual(max(group["lr"] for group in backbone), 5e-5)
        self.assertLess(min(group["lr"] for group in backbone), 5e-6)
        self.assertTrue(all(group["lr"] == 2e-4 for group in head))

        by_layer: dict[int, float] = {}
        for group in backbone:
            by_layer[int(group["layer_id"])] = max(
                by_layer.get(int(group["layer_id"]), 0.0),
                float(group["lr"]),
            )
        ordered = [by_layer[index] for index in sorted(by_layer)]
        self.assertEqual(ordered, sorted(ordered))


    def test_multiclip_forward_is_chunked_without_changing_logits(self) -> None:
        chunked_backbone = _ClipCountingBackbone()
        chunked = _VideoBackboneClassifier(
            backbone=chunked_backbone,
            feature_size=3,
            n_classes=2,
            dropout_rate=0.0,
            freeze_backbone=False,
            freeze_batch_norm=False,
            clip_forward_batch_size=2,
        )
        unchunked = _VideoBackboneClassifier(
            backbone=_ClipCountingBackbone(),
            feature_size=3,
            n_classes=2,
            dropout_rate=0.0,
            freeze_backbone=False,
            freeze_batch_norm=False,
            clip_forward_batch_size=None,
        )
        unchunked.load_state_dict(chunked.state_dict())
        inputs = torch.arange(2 * 5 * 1 * 3, dtype=torch.float32).reshape(
            2, 5, 1, 3, 1, 1
        )
        chunked.eval()
        unchunked.eval()
        with torch.no_grad():
            chunked_logits = chunked(inputs)
            unchunked_logits = unchunked(inputs)
        self.assertTrue(torch.allclose(chunked_logits, unchunked_logits))
        self.assertEqual(chunked_backbone.batch_sizes, [2, 2, 2, 2, 2])

    def test_videomaev2_transform_profile_and_erasing_are_clip_consistent(self) -> None:
        profile = transform_profile("videomaev2_vit_s_distilled", 224)
        self.assertEqual(profile.mean, (0.485, 0.456, 0.406))
        self.assertEqual(profile.std, (0.229, 0.224, 0.225))
        self.assertEqual(profile.interpolation.name, "BICUBIC")

        config = TrainAugmentationConfig(
            crop_scale=(1.0, 1.0),
            crop_ratio=(1.0, 1.0),
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
            hue=0.0,
            horizontal_flip_probability=0.0,
            random_erasing_probability=1.0,
            random_erasing_scale=(0.10, 0.10),
            random_erasing_ratio=(1.0, 1.0),
        )
        train_transform, _ = compose_data_transforms(
            "videomaev2_vit_s_distilled",
            224,
            train_augmentation=config,
        )
        frame = Image.new("RGB", (224, 224), color=(60, 90, 120))
        transformed = train_transform([frame, frame.copy()])
        self.assertTrue(torch.equal(transformed[0], transformed[1]))
        self.assertTrue(bool((transformed[0] == 0).any()))

    def test_videomaev2_checkpoint_specs_are_pinned(self) -> None:
        small = checkpoint_spec_for_architecture(
            "videomaev2_vit_s_distilled"
        )
        base = checkpoint_spec_for_architecture(
            "videomaev2_vit_b_distilled"
        )
        self.assertEqual(small.revision, base.revision)
        self.assertEqual(small.filename, "distill/vit_s_k710_dl_from_giant.pth")
        self.assertEqual(base.filename, "distill/vit_b_k710_dl_from_giant.pth")
        self.assertEqual(
            small.sha256,
            "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172",
        )
        self.assertEqual(
            base.sha256,
            "8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec",
        )
        self.assertEqual(small.file_size, 44_334_609)
        self.assertEqual(base.file_size, 173_574_417)
        self.assertLess(small.file_size, base.file_size)


if __name__ == "__main__":
    unittest.main()
