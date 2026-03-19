# Data preparation

GeoWorld uses the original CrossTask and COIN video annotations and RGB videos. The dataset action space is fixed by those annotations and is shared by preparation, training, evaluation, and planning. Downloaded metadata, videos, extracted features, and checkpoints belong under ignored data/checkpoint directories; they are not distributed with this repository.

## Download annotations and obtain videos

Run these commands from the repository root after installing the environment:

```bash
python scripts/download_metadata.py --dataset all --output data/metadata
```

The downloader retrieves the complete public annotation releases, verifies their SHA256 checksums, and records source URLs in `provenance.json`. Existing downloads are reused after checksum verification.

| Dataset | Official annotation source | Local location |
| --- | --- | --- |
| CrossTask | [Original release archive](https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip), linked by the [CrossTask repository](https://github.com/DmZhukov/CrossTask) | `data/metadata/crosstask/crosstask_release/` |
| COIN | [COIN.json](https://github.com/coin-dataset/annotations/blob/67e31260e9c25951e27cda8d4336a23eacf62a80/COIN.json), pinned to annotation commit `67e31260e9c25951e27cda8d4336a23eacf62a80` | `data/metadata/coin/COIN.json` |

Original video URLs are in CrossTask's `videos.csv` and COIN's `database[video_id].video_url`. Obtain the videos through the [CrossTask release instructions](https://github.com/DmZhukov/CrossTask) and [COIN annotation repository](https://github.com/coin-dataset/annotations). Retain each original video identifier as its filename. Supported extensions are `.mp4`, `.webm`, `.mkv`, `.avi`, `.mov`, and `.m4v`; subdirectories are searched recursively.

```text
data/crosstask/videos/<video_id>.mp4
data/coin/videos/<video_id>.mp4
```

> [!NOTE]
> Some CrossTask and COIN video links are no longer available. Contact the authors of the respective dataset papers to request the original videos for those entries. The downloader retrieves annotations only.

## Prepare full-dataset manifests

```bash
python scripts/prepare_data.py --dataset crosstask \
  --metadata data/metadata/crosstask --videos data/crosstask/videos \
  --output data/crosstask/prepared

python scripts/prepare_data.py --dataset coin \
  --metadata data/metadata/coin --videos data/coin/videos \
  --output data/coin/prepared
```

Preparation creates `vocabulary.json`, `preparation_report.json`, and `train.jsonl`, `val.jsonl`, `test.jsonl`. The report records missing videos, duplicate filenames, overlap counts, the split protocol, and a semantic hash of the shared vocabulary. Missing original videos cause the default command to fail. Multiple files with the same video identifier also cause failure.

`--video-policy available` explicitly prepares only videos that exist locally and reports omissions. Its vocabulary remains the complete dataset vocabulary. `--video-policy metadata` performs an annotation audit while retaining expected paths for missing videos; those manifests cannot be used to extract features until the original videos exist. Missing videos are recorded in the preparation report and must be accounted for in evaluation. A file's presence is checked during preparation; actual decoding is validated during feature extraction.

The public metadata contains:

| Dataset | Annotated videos | Tasks | Fixed actions | Default source split |
| --- | ---: | ---: | ---: | --- |
| CrossTask primary tasks | 2,750 | 18 | 105 | 2,390 train / 360 validation |
| COIN | 11,827 | 180 | 778 | 9,030 train / 2,797 test |

CrossTask's release lists 4,700 videos across 83 tasks, but its 65 related tasks do not provide the action annotations needed here. Only the 18 annotated primary tasks enter the supervised manifests. The COIN count comes directly from the pinned annotation JSON.

## Fixed action vocabulary

CrossTask task cards contain 133 task-local steps. Their exact label strings contain 105 unique actions. IDs are assigned in first-appearance order in `tasks_primary.txt`; equal strings share an ID. The 133 task-local assignments match the released [PDPP `actions_one_hot.npy` mapping](https://github.com/MCG-NJU/PDPP/blob/5c888c24368da694043f9aad89b8483e05c65467/dataset/crosstask/actions_one_hot.npy). Label order is preserved when assigning IDs.

COIN action IDs are the official annotation IDs minus one, giving zero-based IDs `0..777`. Equal text labels with different official IDs stay distinct. Action names do not create a new vocabulary at inference.

Each dataset's `vocabulary.json` records all labels, task memberships, the mapping rule, and hashes of the source label files. The same vocabulary is preserved even when preparing a partial set of available videos or custom splits. Each manifest row carries its `action_count` and `vocabulary_sha256`; loaders reject inconsistencies. This SHA256 is computed over canonical JSON via `geoworld.data.metadata.vocabulary_hash`, not over the formatting of the JSON file.

Task memberships are available as `task_action_ids` and a fixed-width Boolean `task_action_mask`. Task-constrained experiments can use this mask; the default planner searches the entire dataset action space.

## Splits and benchmark comparability

The default CrossTask split uses the original release's `videos_val.csv`, which defines an action-localization validation split. COIN preserves the `training` and `testing` fields in its original annotations. These source splits have not been verified against the GeoWorld planning benchmark splits. The default CrossTask test manifest and COIN validation manifest are empty.

To select benchmark planning splits, pass their exact video ID lists with `--split-file path/to/planning_splits.json`. That JSON must map the keys `train`, `val`, and/or `test` to lists of original video ID strings. A video may appear only once across all lists. Unlisted videos are deliberately excluded, unknown IDs are rejected, and all trajectories from a video stay together.

For development, `--holdout-fraction 0.2 --seed 42` uses a deterministic hash of each video ID to assign train/test membership. It overrides the source split and is explicitly recorded as nonofficial. Its size is determined by the hash threshold rather than an exact rounded fraction. Feature windows are formed after the video split, preventing windows from the same video from crossing train/test boundaries.

## Timestamps and RGB clips

Each manifest row describes one annotated video trajectory. Required preparation fields include `id`, `video_id`, `task`, `dataset`, `split`, the original `video` path, global integer `actions`, annotation `segments` in seconds, `state_times`, `task_action_ids`, and the fixed action count/hash. Feature extraction adds a `features` path. Relative video and feature paths are resolved against the manifest directory; preparation itself writes absolute original-video paths.

Annotations are ordered by their start times, then end times. Repeated action IDs and overlapping intervals are retained. There are 744 overlapping annotated CrossTask videos in the verified release. The state sequence is the start timestamp of each ordered action followed by the maximum annotation end timestamp, yielding exactly `T+1` states for `T` actions. Using the maximum final end preserves monotonic timestamps when intervals are nested.

A state observation is a causal clip ending at its state timestamp. Frames are sampled uniformly across the preceding configured duration; the decoder selects the last presentation timestamp at or before each query. Negative timestamps at the beginning repeat the first frame. Image observations use the endpoint frame. Endpoints up to `max(0.25 seconds, one frame interval)` after the last decoded frame repeat that frame to accommodate small annotation/encoding discrepancies; larger truncation raises an error. Frames use actual encoded presentation timestamps rather than assuming a fixed frame rate.

The initial observation contains only frames at or before the first action's start. Overlapping annotations are retained in the trajectory.

`VideoTrajectoryDataset` and `load_video_clip` return RGB float tensors in `[0,1]`. A trajectory has shape `[T+1,3,F,H,W]`; one clip has shape `[3,F,H,W]`. The frozen V-JEPA 2 encoder owns shortest-side resizing, center cropping, and ImageNet normalization. An optional decoder spatial-size cap limits memory use before encoder preprocessing.

## Cache real encoder features

```bash
for split in train val; do
  python -m geoworld.extract --config configs/crosstask.yaml \
    --manifest "data/crosstask/prepared/${split}.jsonl" \
    --output "data/crosstask/features/${split}"
done
```

For COIN, use `configs/coin.yaml`, the `data/coin/` directories, and splits `train test`. The configured pretrained frozen encoder must be available. `--observation-mode image` selects endpoint images. Training uses the extracted V-JEPA 2 features; the original CrossTask precomputed I3D features have a different representation.

Extraction saves one finite `[T+1,D]` array per video trajectory and a manifest with the original annotations and cache location. Hashed trajectory IDs form filenames; dataset identifiers are never interpreted as output paths. Cached features and their sibling `vocabulary.json` should be moved together. Extraction fails on missing or unreadable videos by default; `--skip-errors` explicitly permits omissions and writes an error report.

`CachedTrajectoryDataset` validates action IDs, vocabulary consistency, feature shape, finite values, and feature dimension. A fixed horizon enumerates contiguous action windows and their corresponding `T+1` feature states; trajectories shorter than that horizon contribute no windows. Without a horizon, full trajectories are returned. Batching pads actions with `-1` and supplies separate valid-action and valid-state masks. Padded positions are excluded by the training loss.

## Local verification

```bash
python -m pytest tests/test_data.py -q
```

Tests cover vocabulary identity, split isolation, action ordering, missing-video behavior, feature windowing and padding, causal frame selection, beginning padding, and truncation errors. Original training videos must be obtained separately.
