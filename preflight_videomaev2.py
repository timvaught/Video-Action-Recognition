"""Download, verify, and structurally validate VideoMAE V2 source weights."""

from __future__ import annotations

import argparse
import json

from videomaev2 import preflight_videomaev2_checkpoint


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
    """Parse preflight arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify a pinned VideoMAE V2 checkpoint and require a structural "
            "backbone match before long-running training."
        )
    )
    parser.add_argument(
        "--architecture",
        choices=(
            "videomaev2_vit_s_distilled",
            "videomaev2_vit_b_distilled",
        ),
        default="videomaev2_vit_s_distilled",
    )
    parser.add_argument("--cache_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--verify_sha256", type=parse_bool, default=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the CPU checkpoint preflight and print a JSON report."""
    report = preflight_videomaev2_checkpoint(
        args.architecture,
        cache_dir=args.cache_dir,
        verify_sha256=args.verify_sha256,
        pretrained_checkpoint=args.checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(args_parser())
