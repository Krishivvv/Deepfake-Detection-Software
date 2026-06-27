"""
create_splits.py — Create train/val/test CSV splits at the VIDEO level.
========================================================================

This script scans the processed data directory for video folders (i.e.,
folders that contain at least one .jpg face image) and creates three CSV
files (train.csv, val.csv, test.csv) that map each video to its label.

Split ratio: 70% train / 15% validation / 15% test
Labels:      0 = real, 1 = fake
Columns:     video_path, label

The CSVs contain PATHS ONLY — no files are duplicated or moved.

Usage:
    python src/preprocessing/create_splits.py
    python -m src.preprocessing.create_splits
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")

RANDOM_STATE = 42


def discover_video_folders(base_dir, label_name):
    """
    Walk a base directory and find all leaf folders that contain .jpg files.
    Returns a list of relative paths (relative to PROCESSED_DATA_DIR) using
    forward slashes for cross-platform compatibility.

    Parameters
    ----------
    base_dir : str
        Absolute path to search (e.g., .../data/processed/real)
    label_name : str
        Top-level label folder name ("real" or "fake")

    Returns
    -------
    list[str]
        Relative paths like "real/004", "fake/deepfakes/004_982", etc.
    """
    video_folders = []
    if not os.path.exists(base_dir):
        print(f"  WARNING: Directory not found: {base_dir}")
        return video_folders

    for root, dirs, files in os.walk(base_dir):
        # A folder is a "video folder" if it contains at least one .jpg
        has_images = any(f.lower().endswith(".jpg") for f in files)
        if has_images:
            # Get path relative to PROCESSED_DATA_DIR
            rel = os.path.relpath(root, PROCESSED_DATA_DIR).replace("\\", "/")
            video_folders.append(rel)

    return video_folders


def main():
    print("=" * 60)
    print("  Generating Train/Val/Test CSV Splits (70/15/15)")
    print("  Split level: VIDEO (no data leakage)")
    print("=" * 60)

    os.makedirs(SPLITS_DIR, exist_ok=True)

    real_dir = os.path.join(PROCESSED_DATA_DIR, "real")
    fake_dir = os.path.join(PROCESSED_DATA_DIR, "fake")

    # ── Discover video folders ───────────────────────────────────────────
    print(f"\nScanning: {PROCESSED_DATA_DIR}")
    real_videos = discover_video_folders(real_dir, "real")
    fake_videos = discover_video_folders(fake_dir, "fake")

    print(f"  Found {len(real_videos)} REAL video folders")
    print(f"  Found {len(fake_videos)} FAKE video folders")
    total = len(real_videos) + len(fake_videos)
    print(f"  Total: {total} videos")

    if total == 0:
        print("\nERROR: No processed video folders found!")
        print("       Run the preprocessing pipeline first:")
        print("       python -m src.preprocessing.run_pipeline")
        sys.exit(1)

    # ── Build a single DataFrame ─────────────────────────────────────────
    records = []
    for v in real_videos:
        records.append({"video_path": v, "label": 0})
    for v in fake_videos:
        records.append({"video_path": v, "label": 1})

    df = pd.DataFrame(records)

    # ── Stratified split at VIDEO level ──────────────────────────────────
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Second split: 50/50 of the 30% → 15% val + 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    # ── Save CSVs ────────────────────────────────────────────────────────
    splits = {"train": train_df, "val": val_df, "test": test_df}

    print(f"\nSaving splits to: {SPLITS_DIR}")
    for name, split_df in splits.items():
        out_path = os.path.join(SPLITS_DIR, f"{name}.csv")
        split_df.to_csv(out_path, index=False)

        n_real = (split_df["label"] == 0).sum()
        n_fake = (split_df["label"] == 1).sum()
        print(f"  {name}.csv: {len(split_df)} videos  "
              f"(real={n_real}, fake={n_fake})")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'-' * 60}")
    print("  SPLIT SUMMARY")
    print(f"{'-' * 60}")
    print(f"  Train : {len(train_df):>5} videos  ({len(train_df)/total*100:.1f}%)")
    print(f"  Val   : {len(val_df):>5} videos  ({len(val_df)/total*100:.1f}%)")
    print(f"  Test  : {len(test_df):>5} videos  ({len(test_df)/total*100:.1f}%)")
    print(f"  Total : {total:>5} videos")
    print(f"{'-' * 60}")
    print("  Labels: 0 = real, 1 = fake")
    print("  Columns: video_path, label")
    print("  NO files duplicated -- CSV mapping only [OK]")
    print(f"{'-' * 60}")


if __name__ == "__main__":
    main()
