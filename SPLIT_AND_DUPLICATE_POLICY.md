# Official split setup and exact-duplicate policy

## Split directory resolution

`--official_split_dir` may point either to the directory that directly contains
all 51 `*_test_splitN.txt` files or to a parent directory created by extracting
the archive with extra nesting. The resolver:

- searches only below the supplied path;
- requires all 51 files for the selected split in one directory;
- rejects multiple complete candidates rather than selecting silently; and
- reports matching-file counts and examples of missing files when setup is
  incomplete.

The core training command requires extracted text files. The v7 Colab notebook
can download, checksum-verify, extract, and validate the small official split
archive automatically.

## Exact-content duplicates in an official split

The audit first hashes the frame count and seven representative frames to find
candidates. It then hashes every ordered extracted frame before declaring two
videos byte-identical. A candidate match alone is never used to remove data.

Use one of these policies:

- `drop_train` (v7 default): preserve the complete official test assignment and
  remove exact training copies that overlap it before carving validation. This
  is leakage-safe, but the result must be described as an **official-derived
  decontaminated split**, not the untouched historical benchmark.
- `allow`: preserve the historical assignment and report the known overlap. This
  is useful only for reproduction and is not leakage-free.
- `error`: abort when a confirmed exact train/test overlap is found.

The selected policy, every affected group, and all removed training paths are
saved in `splits.json` and `official_duplicate_audit.json`.

The byte-level audit still does not detect arbitrary re-encoding, resizing,
cropping, temporal shifts, or other perceptual near-duplicates.
