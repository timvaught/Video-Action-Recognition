"""Dataset discovery, split handling, temporal sampling, and video batching."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch.utils.data import Dataset

Sample = tuple[str, int]
ClipTransform = Callable[[list[Image.Image]], list[Tensor]]
SUPPORTED_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _natural_key(path: Path) -> tuple:
    """Return a natural-sort key for numbered frame filenames."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def list_frame_paths(video_dir: str | Path) -> list[Path]:
    """Return supported image frames in deterministic natural order."""
    directory = Path(video_dir)
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_FRAME_EXTENSIONS
        ),
        key=_natural_key,
    )


class VideoDataset(Dataset):
    """Load fixed-length clips from directories of extracted image frames."""

    def __init__(
        self,
        samples: Sequence[Sample],
        frames_per_video: int,
        transform: ClipTransform | None = None,
        training: bool = False,
        eval_clips: int = 1,
        temporal_stride: int = 1,
    ) -> None:
        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be positive")
        if eval_clips <= 0:
            raise ValueError("eval_clips must be positive")
        if temporal_stride <= 0:
            raise ValueError("temporal_stride must be positive")
        self.samples = list(samples)
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.training = training
        self.eval_clips = 1 if training else eval_clips
        self.temporal_stride = temporal_stride

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, frame_count: int, clip_index: int = 0) -> np.ndarray:
        """Sample one temporal clip without long runs of repeated tail frames.

        Videos long enough for the requested temporal span use a contiguous window
        with the configured stride. Shorter videos are sampled uniformly across
        their complete duration. Uniform sampling can repeat an index only when a
        video contains fewer source frames than requested output frames.
        """
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")

        required_span = 1 + (self.frames_per_video - 1) * self.temporal_stride
        if frame_count < required_span:
            return np.rint(
                np.linspace(0, frame_count - 1, self.frames_per_video)
            ).astype(int)

        max_start = frame_count - required_span
        if self.training:
            start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        elif self.eval_clips == 1 or max_start == 0:
            start = max_start // 2
        else:
            starts = np.linspace(0, max_start, self.eval_clips)
            start = int(round(float(starts[clip_index])))

        return (
            start + np.arange(self.frames_per_video) * self.temporal_stride
        ).astype(int)

    def _load_clip(self, paths: list[Path], indices: np.ndarray) -> Tensor:
        if self.transform is None:
            raise RuntimeError("A tensor-producing clip transform is required")
        images: list[Image.Image] = []
        for frame_index in indices:
            with Image.open(paths[int(frame_index)]) as image:
                images.append(image.convert("RGB").copy())
        return torch.stack(self.transform(images))

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        video_dir, label = self.samples[index]
        frame_paths = list_frame_paths(video_dir)
        if not frame_paths:
            raise RuntimeError(f"No supported image frames found in {video_dir}")

        clips = [
            self._load_clip(
                frame_paths,
                self._sample_indices(len(frame_paths), clip_index),
            )
            for clip_index in range(self.eval_clips)
        ]
        # Always include a clip dimension: [K,T,C,H,W]. Training uses K=1.
        return torch.stack(clips), int(label)


def load_dataset(frame_dir: str) -> tuple[list[Sample], dict[str, int]]:
    """Discover class/video directories deterministically."""
    root = Path(frame_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {root}")

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    label_dict = {path.name: index for index, path in enumerate(class_dirs)}
    samples: list[Sample] = []
    for class_dir in class_dirs:
        for video_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            if list_frame_paths(video_dir):
                samples.append((str(video_dir), label_dict[class_dir.name]))
    if not samples:
        raise RuntimeError(f"No frame directories found beneath {root}")
    return samples, label_dict


def _update_digest_with_frame(digest: Any, path: Path) -> None:
    """Add one frame to a video digest without depending on its filename."""
    data = path.read_bytes()
    digest.update(len(data).to_bytes(8, byteorder="big", signed=False))
    digest.update(data)


@lru_cache(maxsize=None)
def _coarse_video_fingerprint(video_dir: str) -> str:
    """Hash frame count and representative frame bytes for candidate discovery.

    This intentionally inexpensive hash is used only to find possible duplicate
    groups. Candidate groups are confirmed with a full-frame hash before they are
    treated as exact duplicates.
    """
    paths = list_frame_paths(video_dir)
    digest = hashlib.sha256()
    digest.update(len(paths).to_bytes(8, byteorder="big", signed=False))
    if not paths:
        digest.update(str(Path(video_dir).resolve()).encode("utf-8"))
        return digest.hexdigest()

    selected = np.linspace(0, len(paths) - 1, min(7, len(paths)), dtype=int)
    for frame_index in selected:
        digest.update(int(frame_index).to_bytes(8, byteorder="big", signed=False))
        _update_digest_with_frame(digest, paths[int(frame_index)])
    return digest.hexdigest()


@lru_cache(maxsize=None)
def video_fingerprint(video_dir: str) -> str:
    """Hash every extracted frame to identify exact byte-for-byte video copies."""
    paths = list_frame_paths(video_dir)
    digest = hashlib.sha256()
    digest.update(len(paths).to_bytes(8, byteorder="big", signed=False))
    if not paths:
        digest.update(str(Path(video_dir).resolve()).encode("utf-8"))
        return digest.hexdigest()

    for frame_index, frame_path in enumerate(paths):
        digest.update(frame_index.to_bytes(8, byteorder="big", signed=False))
        _update_digest_with_frame(digest, frame_path)
    return digest.hexdigest()


def _group_tagged_samples_by_exact_content(
    tagged_samples: Sequence[tuple[str, Sample]],
) -> list[list[tuple[str, Sample]]]:
    """Group exact copies while hashing all frames only for candidate groups."""
    coarse_groups: dict[str, list[tuple[str, Sample]]] = {}
    for tagged_sample in tagged_samples:
        _, sample = tagged_sample
        coarse_groups.setdefault(
            _coarse_video_fingerprint(sample[0]), []
        ).append(tagged_sample)

    exact_groups: list[list[tuple[str, Sample]]] = []
    for candidates in coarse_groups.values():
        if len(candidates) == 1:
            exact_groups.append(candidates)
            continue

        confirmed: dict[str, list[tuple[str, Sample]]] = {}
        for tagged_sample in candidates:
            _, sample = tagged_sample
            confirmed.setdefault(video_fingerprint(sample[0]), []).append(tagged_sample)
        exact_groups.extend(confirmed.values())
    return exact_groups


def _group_by_fingerprint(samples: Sequence[Sample]) -> list[list[Sample]]:
    groups = [
        [sample for _, sample in tagged_group]
        for tagged_group in _group_tagged_samples_by_exact_content(
            [("dataset", sample) for sample in samples]
        )
    ]

    conflicting = [
        group for group in groups
        if len({sample[1] for sample in group}) > 1
    ]
    if conflicting:
        examples = ", ".join(sample[0] for sample in conflicting[0][:3])
        raise RuntimeError(
            "Exact duplicate videos were found under different class labels. "
            f"Resolve the conflicting annotations before splitting. Example: {examples}"
        )
    return groups


def decontaminate_official_train_test(
    official_train: Sequence[Sample],
    official_test: Sequence[Sample],
    *,
    policy: Literal["error", "drop_train", "allow"] = "error",
) -> tuple[list[Sample], list[Sample], dict[str, Any]]:
    """Handle exact copies that cross an official train/test boundary.

    ``drop_train`` preserves every official test item and removes all training
    copies whose exact extracted-frame content also occurs in the test set. This
    produces a leakage-safe *official-derived* protocol rather than the untouched
    official benchmark. ``allow`` preserves the original official assignments and
    reports the contamination. ``error`` is the strict audit mode.
    """
    if policy not in {"error", "drop_train", "allow"}:
        raise ValueError(
            "official duplicate policy must be error, drop_train, or allow"
        )

    tagged = [
        *(('train', sample) for sample in official_train),
        *(('test', sample) for sample in official_test),
    ]
    duplicate_groups: list[list[tuple[str, Sample]]] = []
    for group in _group_tagged_samples_by_exact_content(tagged):
        owners = {split_name for split_name, _ in group}
        if len(group) > 1 and owners == {"train", "test"}:
            labels = {int(sample[1]) for _, sample in group}
            if len(labels) > 1:
                examples = ", ".join(
                    f"{owner}={sample[0]} (label {sample[1]})"
                    for owner, sample in group[:4]
                )
                raise RuntimeError(
                    "Exact-content official train/test duplicates have conflicting "
                    f"labels: {examples}"
                )
            duplicate_groups.append(group)

    examples: list[dict[str, Any]] = []
    train_paths_to_drop: set[str] = set()
    overlapping_test_paths: set[str] = set()
    train_members = 0
    test_members = 0
    for group in duplicate_groups:
        train_paths = [
            str(Path(sample[0]).resolve())
            for owner, sample in group
            if owner == "train"
        ]
        test_paths = [
            str(Path(sample[0]).resolve())
            for owner, sample in group
            if owner == "test"
        ]
        train_members += len(train_paths)
        test_members += len(test_paths)
        train_paths_to_drop.update(train_paths)
        overlapping_test_paths.update(test_paths)
        examples.append(
            {
                "label": int(group[0][1][1]),
                "train_paths": train_paths,
                "test_paths": test_paths,
            }
        )

    report: dict[str, Any] = {
        "policy": policy,
        "official_train_before": len(official_train),
        "official_test_videos": len(official_test),
        "cross_split_exact_duplicate_groups": len(duplicate_groups),
        "cross_split_train_members": train_members,
        "cross_split_test_members": test_members,
        "dropped_train_videos": 0,
        "official_train_after": len(official_train),
        "dropped_train_paths": [],
        "overlapping_test_paths": [],
        "examples": examples,
    }

    if duplicate_groups and policy == "error":
        first = duplicate_groups[0]
        first_train = next(sample for owner, sample in first if owner == "train")
        first_test = next(sample for owner, sample in first if owner == "test")
        raise RuntimeError(
            "Exact-content duplicate crosses the official train/test boundary: "
            f"train={first_train[0]} (label {first_train[1]}) and "
            f"test={first_test[0]} (label {first_test[1]}). "
            f"Found {len(duplicate_groups)} cross-boundary exact-duplicate group(s). "
            "Use --official_duplicate_policy drop_train for a leakage-safe "
            "official-derived split, or --official_duplicate_policy allow to "
            "preserve the untouched official benchmark and report the overlap."
        )

    cleaned_train = list(official_train)
    if policy == "drop_train" and train_paths_to_drop:
        cleaned_train = [
            sample
            for sample in official_train
            if str(Path(sample[0]).resolve()) not in train_paths_to_drop
        ]
        report["dropped_train_videos"] = len(official_train) - len(cleaned_train)
        report["official_train_after"] = len(cleaned_train)

    report["dropped_train_paths"] = sorted(train_paths_to_drop)
    report["overlapping_test_paths"] = sorted(overlapping_test_paths)

    return cleaned_train, list(official_test), report


def validate_split_integrity(
    train_split: Sequence[Sample],
    validation_split: Sequence[Sample],
    test_split: Sequence[Sample],
    *,
    check_fingerprints: bool = True,
    cross_split_duplicate_policy: Literal["error", "report"] = "error",
) -> dict[str, int]:
    """Reject path overlap and audit exact extracted-frame duplicates.

    Candidate groups are found using representative frames, then confirmed by
    hashing every extracted frame. ``cross_split_duplicate_policy='report'`` is
    intended only for explicitly requested untouched benchmark protocols.
    """
    if cross_split_duplicate_policy not in {"error", "report"}:
        raise ValueError("cross_split_duplicate_policy must be error or report")

    split_map = {
        "train": list(train_split),
        "val": list(validation_split),
        "test": list(test_split),
    }
    normalized_paths: dict[str, dict[str, int]] = {}

    for split_name, samples in split_map.items():
        if not samples:
            raise RuntimeError(f"{split_name} split is empty")
        paths: dict[str, int] = {}
        for video_dir, label in samples:
            resolved = str(Path(video_dir).resolve())
            if resolved in paths:
                raise RuntimeError(
                    f"Duplicate video path inside {split_name}: {resolved}"
                )
            directory = Path(resolved)
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Video directory in {split_name} does not exist: {directory}"
                )
            if not list_frame_paths(directory):
                raise RuntimeError(
                    f"Video directory in {split_name} contains no supported frames: "
                    f"{directory}"
                )
            paths[resolved] = int(label)
        normalized_paths[split_name] = paths

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left_name, right_name in pairs:
        overlap = set(normalized_paths[left_name]) & set(normalized_paths[right_name])
        if overlap:
            example = sorted(overlap)[0]
            raise RuntimeError(
                f"Path overlap between {left_name} and {right_name}: {example}"
            )

    within_split_groups = 0
    within_split_members = 0
    cross_split_groups = 0
    cross_split_members = 0
    if check_fingerprints:
        tagged = [
            (split_name, sample)
            for split_name, samples in split_map.items()
            for sample in samples
        ]
        for group in _group_tagged_samples_by_exact_content(tagged):
            if len(group) <= 1:
                continue

            labels = {int(sample[1]) for _, sample in group}
            if len(labels) > 1:
                first_owner, first_sample = group[0]
                second_owner, second_sample = next(
                    (owner, sample)
                    for owner, sample in group[1:]
                    if int(sample[1]) != int(first_sample[1])
                )
                raise RuntimeError(
                    "Exact-content duplicate has conflicting class labels: "
                    f"{first_owner}={first_sample[0]} (label {first_sample[1]}) and "
                    f"{second_owner}={second_sample[0]} (label {second_sample[1]})"
                )

            owners = {split_name for split_name, _ in group}
            if len(owners) > 1:
                cross_split_groups += 1
                cross_split_members += len(group)
                if cross_split_duplicate_policy == "error":
                    first_owner, first_sample = group[0]
                    second_owner, second_sample = next(
                        (owner, sample)
                        for owner, sample in group[1:]
                        if owner != first_owner
                    )
                    raise RuntimeError(
                        "Exact-content duplicate crosses split boundaries: "
                        f"{first_owner}={first_sample[0]} "
                        f"(label {first_sample[1]}) and "
                        f"{second_owner}={second_sample[0]} "
                        f"(label {second_sample[1]})"
                    )
            else:
                within_split_groups += 1
                within_split_members += len(group) - 1

    return {
        "train_videos": len(split_map["train"]),
        "val_videos": len(split_map["val"]),
        "test_videos": len(split_map["test"]),
        "within_split_exact_duplicate_groups": within_split_groups,
        "within_split_exact_duplicate_members": within_split_members,
        "cross_split_exact_duplicate_groups": cross_split_groups,
        "cross_split_exact_duplicate_members": cross_split_members,
    }

def _expand_groups(groups: list[list[Sample]], chosen: np.ndarray) -> list[Sample]:
    return [item for group_index in chosen for item in groups[int(group_index)]]


def train_validation_split(
    samples: Sequence[Sample],
    validation_ratio: float,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Create duplicate-aware stratified train/validation subsets."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")

    groups = _group_by_fingerprint(samples)
    representatives = [group[0] for group in groups]
    labels = np.array([sample[1] for sample in representatives])
    indices = np.arange(len(representatives))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_ratio,
        random_state=seed,
    )
    train_idx, validation_idx = next(splitter.split(indices, labels))
    return _expand_groups(groups, train_idx), _expand_groups(groups, validation_idx)


def dataset_split(
    samples: Sequence[Sample],
    train_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Create duplicate-aware stratified random train/validation/test splits."""
    if train_ratio <= 0 or test_ratio <= 0 or train_ratio + test_ratio >= 1:
        raise ValueError("train_ratio and test_ratio must be positive and sum to < 1")

    groups = _group_by_fingerprint(samples)
    representatives = [group[0] for group in groups]
    labels = np.array([sample[1] for sample in representatives])
    indices = np.arange(len(representatives))

    first_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=seed,
    )
    train_val_idx, test_idx = next(first_split.split(indices, labels))

    validation_ratio = 1.0 - train_ratio - test_ratio
    relative_validation = validation_ratio / (train_ratio + validation_ratio)
    second_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_validation,
        random_state=seed,
    )
    train_local, validation_local = next(
        second_split.split(train_val_idx, labels[train_val_idx])
    )
    train_idx = train_val_idx[train_local]
    validation_idx = train_val_idx[validation_local]

    return (
        _expand_groups(groups, train_idx),
        _expand_groups(groups, validation_idx),
        _expand_groups(groups, test_idx),
    )


def resolve_official_split_dir(
    split_dir: str | Path,
    class_names: Sequence[str],
    split_number: int = 1,
) -> Path:
    """Locate the directory that directly contains all official split files.

    ``split_dir`` may point either to the split-file directory itself or to a
    parent directory created by extracting an archive with an extra top-level
    folder. The search stays inside ``split_dir`` and rejects ambiguous matches.
    """
    if split_number not in {1, 2, 3}:
        raise ValueError("split_number must be 1, 2, or 3")

    root = Path(split_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Official split path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(
            "Official split path must be an extracted directory, not an archive: "
            f"{root}"
        )

    expected_names = [
        f"{class_name}_test_split{split_number}.txt"
        for class_name in sorted(class_names)
    ]

    def is_complete(candidate: Path) -> bool:
        return all((candidate / name).is_file() for name in expected_names)

    if is_complete(root):
        return root.resolve()

    candidate_dirs = {
        path.parent.resolve()
        for path in root.rglob(f"*_test_split{split_number}.txt")
    }
    complete_dirs = sorted(
        (candidate for candidate in candidate_dirs if is_complete(candidate)),
        key=lambda candidate: (len(candidate.parts), str(candidate)),
    )

    if len(complete_dirs) == 1:
        return complete_dirs[0]
    if len(complete_dirs) > 1:
        formatted = "\n  - ".join(str(path) for path in complete_dirs[:10])
        raise RuntimeError(
            "Multiple complete official split directories were found below "
            f"{root}. Pass the intended directory explicitly:\n  - {formatted}"
        )

    present_files = sorted(root.rglob(f"*_test_split{split_number}.txt"))
    present_names = {path.name for path in present_files}
    missing_names = [name for name in expected_names if name not in present_names]
    sample_missing = ", ".join(missing_names[:8])
    nearby = "\n  - ".join(str(path) for path in present_files[:10])
    details = (
        f"Found {len(present_files)} files matching *_test_split{split_number}.txt "
        f"below {root}; expected {len(expected_names)} in one directory."
    )
    if sample_missing:
        details += f" Example missing files: {sample_missing}."
    if nearby:
        details += f"\nFirst matching files found:\n  - {nearby}"

    raise FileNotFoundError(
        "Could not locate a complete extracted HMDB51 official split directory. "
        f"{details} The directory must directly or recursively contain files such "
        f"as {expected_names[0]}."
    )


def official_dataset_split(
    samples: Sequence[Sample],
    label_dict: dict[str, int],
    split_dir: str | Path,
    split_number: int = 1,
    validation_ratio: float = 0.15,
    seed: int = 42,
    duplicate_policy: Literal["error", "drop_train", "allow"] = "error",
    return_audit: bool = False,
) -> (
    tuple[list[Sample], list[Sample], list[Sample]]
    | tuple[list[Sample], list[Sample], list[Sample], dict[str, Any]]
):
    """Use an official HMDB51 train/test split and carve validation from train.

    Official split rows use ``1`` for train, ``2`` for test, and ``0`` for unused.
    The extracted frame directory may retain or omit the original video extension;
    both forms are matched. Exact copies crossing the official train/test boundary
    are handled explicitly by ``duplicate_policy`` before validation is carved out.
    """
    if split_number not in {1, 2, 3}:
        raise ValueError("split_number must be 1, 2, or 3")

    requested_root = Path(split_dir).expanduser()
    root = resolve_official_split_dir(
        requested_root,
        label_dict.keys(),
        split_number=split_number,
    )
    try:
        requested_resolved = requested_root.resolve()
    except OSError:
        requested_resolved = requested_root
    if root != requested_resolved:
        print(f"Resolved nested official split directory: {root}")

    lookup: dict[tuple[int, str], list[Sample]] = {}
    for sample in samples:
        video_path = Path(sample[0])
        label = sample[1]
        aliases = {video_path.name, video_path.stem}
        for alias in aliases:
            lookup.setdefault((label, alias), []).append(sample)

    official_train: list[Sample] = []
    official_test: list[Sample] = []
    unmatched: list[str] = []
    ambiguous: list[tuple[str, list[str]]] = []
    assigned_paths: dict[str, tuple[str, str]] = {}

    for class_name, label in sorted(label_dict.items(), key=lambda item: item[1]):
        split_file = root / f"{class_name}_test_split{split_number}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing official split file: {split_file}")

        for raw_line in split_file.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            filename, split_code_text = line.rsplit(maxsplit=1)
            split_code = int(split_code_text)
            if split_code not in {0, 1, 2}:
                raise ValueError(
                    f"Invalid split code {split_code} in {split_file}: {line!r}"
                )
            if split_code == 0:
                continue

            candidates: dict[str, Sample] = {}
            for alias in {filename, Path(filename).stem}:
                for candidate in lookup.get((label, alias), []):
                    candidates[str(Path(candidate[0]).resolve())] = candidate

            if not candidates:
                unmatched.append(f"{class_name}/{filename}")
                continue

            if len(candidates) > 1:
                ambiguous.append(
                    (
                        f"{class_name}/{filename}",
                        sorted(candidates),
                    )
                )
                continue

            sample = next(iter(candidates.values()))
            resolved_sample = str(Path(sample[0]).resolve())
            split_name = "train" if split_code == 1 else "test"
            previous_assignment = assigned_paths.get(resolved_sample)
            if previous_assignment is not None:
                previous_split, previous_entry = previous_assignment
                raise RuntimeError(
                    "One extracted video directory matched multiple official split "
                    f"entries: {resolved_sample} matched {previous_entry} "
                    f"({previous_split}) and {class_name}/{filename} ({split_name})."
                )
            assigned_paths[resolved_sample] = (
                split_name,
                f"{class_name}/{filename}",
            )

            if split_code == 1:
                official_train.append(sample)
            elif split_code == 2:
                official_test.append(sample)

    if unmatched:
        examples = "\n  - ".join(unmatched[:10])
        raise RuntimeError(
            f"Could not match {len(unmatched)} official split entries to frame "
            f"directories. First unmatched entries:\n  - {examples}"
        )
    if ambiguous:
        first_entry, first_candidates = ambiguous[0]
        candidates_text = "\n  - ".join(first_candidates[:10])
        raise RuntimeError(
            f"Could not uniquely match {len(ambiguous)} official split entries to "
            "frame directories. The first ambiguous entry was "
            f"{first_entry}, which matched:\n  - {candidates_text}\n"
            "Remove or rename duplicate extraction directories before training."
        )
    if not official_train or not official_test:
        raise RuntimeError("Official split produced an empty train or test set")

    cleaned_train, cleaned_test, duplicate_audit = decontaminate_official_train_test(
        official_train,
        official_test,
        policy=duplicate_policy,
    )
    if duplicate_audit["cross_split_exact_duplicate_groups"]:
        print(
            "Official train/test exact-duplicate audit: "
            f"groups={duplicate_audit['cross_split_exact_duplicate_groups']}, "
            f"train_members={duplicate_audit['cross_split_train_members']}, "
            f"test_members={duplicate_audit['cross_split_test_members']}, "
            f"policy={duplicate_policy}, "
            f"dropped_train={duplicate_audit['dropped_train_videos']}"
        )
        if duplicate_policy == "drop_train":
            print(
                "The official test set is preserved, but this is now a "
                "leakage-safe official-derived protocol rather than the untouched "
                "official benchmark."
            )
        elif duplicate_policy == "allow":
            print(
                "WARNING: exact-content overlap is being retained to preserve the "
                "untouched official benchmark."
            )
    else:
        print("Official train/test exact-duplicate audit: no overlap found")

    train_split, validation_split = train_validation_split(
        cleaned_train,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    if return_audit:
        return train_split, validation_split, cleaned_test, duplicate_audit
    return train_split, validation_split, cleaned_test

def dataset_audit(
    samples: Sequence[Sample],
    frames_per_video: int,
    temporal_stride: int,
) -> dict[str, float | int]:
    """Summarize frame counts and how often uniform short-video sampling is used."""
    frame_counts = np.array(
        [len(list_frame_paths(video_dir)) for video_dir, _ in samples],
        dtype=int,
    )
    if frame_counts.size == 0:
        raise ValueError("Cannot audit an empty sample collection")

    required_span = 1 + (frames_per_video - 1) * temporal_stride
    short_count = int((frame_counts < required_span).sum())
    fewer_than_output = int((frame_counts < frames_per_video).sum())
    return {
        "videos": int(frame_counts.size),
        "minimum_frames": int(frame_counts.min()),
        "median_frames": float(np.median(frame_counts)),
        "maximum_frames": int(frame_counts.max()),
        "required_span": int(required_span),
        "videos_using_uniform_sampling": short_count,
        "uniform_sampling_fraction": float(short_count / frame_counts.size),
        "videos_with_fewer_frames_than_output": fewer_than_output,
    }


def collate_fn_video(batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor, Tensor]:
    """Stack fixed-size clips and return compatibility sequence lengths."""
    videos, labels = zip(*batch)
    stacked = torch.stack(videos)  # [B,K,T,C,H,W]
    lengths = torch.full((len(videos),), stacked.shape[2], dtype=torch.long)
    return stacked, torch.tensor(labels, dtype=torch.long), lengths


# Backward-compatible import name used by older code.
collate_fn_rnn = collate_fn_video
