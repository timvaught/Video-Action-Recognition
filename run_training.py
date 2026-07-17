"""Backward-compatible alias for the unified run.py entry point."""

from run import args_parser, main


if __name__ == "__main__":
    main(args_parser())
