"""
extract_frames.py — Extract evenly-spaced frames from a video file.
====================================================================

Usage (standalone):
    python -m src.preprocessing.extract_frames <video_path> <output_dir> [--n_frames 32]

Public API:
    extract_frames(video_path, output_dir, n_frames=32) -> list[str]
"""

import os
from typing import List

import cv2
import numpy as np


def extract_frames(
    video_path: str,
    output_dir: str,
    n_frames: int = 32,
    save: bool = True,
) -> List[np.ndarray]:
    """
    Extract *n_frames* evenly-spaced frames from *video_path*.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to an MP4 video file.
    output_dir : str
        Directory where extracted frames will be saved (as JPGs).
        Created automatically if it does not exist.
    n_frames : int, default 32
        How many frames to extract from the video.
    save : bool, default True
        If True, save each frame as ``frame_XX.jpg`` inside *output_dir*.

    Returns
    -------
    frames : list[np.ndarray]
        List of BGR frames (OpenCV format).  Empty list on failure.
    """

    # ── 1. Open the video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 1:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")

    # ── 2. Compute evenly-spaced frame indices ───────────────────────────
    # If the video has fewer frames than requested, take every frame.
    actual_n = min(n_frames, total_frames)
    # np.linspace gives *actual_n* evenly-spaced points in [0, total_frames-1]
    frame_indices = np.linspace(0, total_frames - 1, num=actual_n, dtype=int)

    # ── 3. Read the selected frames ──────────────────────────────────────
    frames: List[np.ndarray] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)   # seek to the target frame
        ret, frame = cap.read()
        if not ret:
            continue  # skip unreadable frames silently
        frames.append(frame)

    cap.release()

    # ── 4. Optionally save each frame as a JPG ───────────────────────────
    if save and frames:
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            filename = f"frame_{i:02d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)

    return frames


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("video_path", help="Path to the input video (.mp4)")
    parser.add_argument("output_dir", help="Directory to save extracted frames")
    parser.add_argument(
        "--n_frames", type=int, default=32,
        help="Number of evenly-spaced frames to extract (default: 32)",
    )
    args = parser.parse_args()

    extracted = extract_frames(args.video_path, args.output_dir, args.n_frames)
    print(f"Extracted {len(extracted)} frames → {args.output_dir}")
