# Official HMDB51 exact-duplicate policy

Use one of the following values with `--official_duplicate_policy`:

- `drop_train` (recommended for leakage-safe experiments): preserve every official
  test item, remove training copies whose extracted frames are byte-identical to a
  test item, and then create the validation subset from the cleaned official train
  portion.
- `allow` (historical benchmark reproduction): keep the untouched official
  assignments and report the exact overlap. This is not leakage-free.
- `error`: stop on the first confirmed train/test overlap.

The `drop_train` result is an **official-derived decontaminated protocol**. It is not
strictly comparable with papers that trained on the untouched official 70-video
per-class training assignment. The test portion remains unchanged at 30 videos per
class; only contaminated training copies are removed.
