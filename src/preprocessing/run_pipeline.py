"""
run_pipeline.py — End-to-end preprocessing pipeline for the dataset.
=====================================================================

Usage:
    python -m src.preprocessing.run_pipeline
"""

import datetime
import glob
import os
import traceback

import cv2
from tqdm import tqdm

from .detect_faces import FaceDetector
from .extract_frames import extract_frames

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOG_FILE = os.path.join(PROJECT_ROOT, "outputs", "preprocessing.log")

def setup_directories():
    """Ensure raw data folders and log directory exist."""
    # Create the expected nested input structure
    expected_dirs = [
        "real",
        "fake/deepfakes",
        "fake/face2face",
        "fake/faceswap",
        "fake/neuraltextures",
        "kaggle/real",
        "kaggle/fake"
    ]
    for d in expected_dirs:
        os.makedirs(os.path.join(RAW_DATA_DIR, d), exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_error(msg: str):
    """Append error message to the log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")

def process_video(
    video_path: str,
    output_dir: str,
    detector: FaceDetector,
    n_frames: int = 32
) -> tuple[int, int]:
    """
    Process a single video: extract frames, crop faces, save to processed dir.

    Returns:
        tuple: (extracted_count, faces_detected_count)
    """
    # Create the specific output directory for this video's frames
    os.makedirs(output_dir, exist_ok=True)

    extracted_count = 0
    faces_detected_count = 0

    try:
        # Step 1: Extract frames (don't save original frames, we only want faces)
        frames = extract_frames(video_path, output_dir=None, n_frames=n_frames, save=False)
        extracted_count = len(frames)

        if extracted_count == 0:
            log_error(f"WARNING: No frames extracted from {video_path}")
            return (0, 0)

        # Step 2: Detect and crop face in each frame
        for i, frame in enumerate(frames):
            face_img = detector.detect_and_crop(frame)

            if face_img is not None:
                # Save cropped face
                out_path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
                cv2.imwrite(out_path, face_img)
                faces_detected_count += 1
            else:
                log_error(f"WARNING: No face detected in frame {i} of {video_path}")

    except Exception:
        log_error(f"ERROR processing video {video_path}:\n{traceback.format_exc()}")

    return (extracted_count, faces_detected_count)


def main():
    print("=" * 60)
    print(" Deepfake Detection — Data Preprocessing Pipeline")
    print("=" * 60)

    setup_directories()

    # Initialize detector once to load model into memory
    print("\nLoading MTCNN face detector...")
    detector = FaceDetector(target_size=(224, 224))

    # Find all videos
    # We look for mp4 files recursively in RAW_DATA_DIR
    search_pattern = os.path.join(RAW_DATA_DIR, "**", "*.mp4")
    video_files = glob.glob(search_pattern, recursive=True)

    if not video_files:
        print(f"\nNo .mp4 videos found in {RAW_DATA_DIR}/")
        print("Please place your videos in the correct nested directories and run again.")
        return

    print(f"\nFound {len(video_files)} videos to process.")

    # Statistics
    stats = {
        "videos_processed": 0,
        "videos_failed": 0,
        "total_frames_extracted": 0,
        "total_faces_detected": 0
    }

    log_error("--- Started Preprocessing Run ---")

    # Process each video with a tqdm progress bar
    progress_bar = tqdm(video_files, desc="Processing videos", unit="vid")

    for video_path in progress_bar:
        # Figure out the relative path to maintain the folder structure
        # e.g., "fake/deepfakes/video_001.mp4"
        rel_path = os.path.relpath(video_path, RAW_DATA_DIR)
        rel_dir = os.path.dirname(rel_path)  # e.g., "fake/deepfakes"

        filename = os.path.basename(rel_path)
        video_name = os.path.splitext(filename)[0]

        # Mirror the directory structure in processed data
        output_dir = os.path.join(PROCESSED_DATA_DIR, rel_dir, video_name)

        frames, faces = process_video(video_path, output_dir, detector)

        if frames > 0:
            stats["videos_processed"] += 1
            stats["total_frames_extracted"] += frames
            stats["total_faces_detected"] += faces
        else:
            stats["videos_failed"] += 1

        progress_bar.set_postfix({
            "faces": stats["total_faces_detected"],
            "fails": stats["videos_failed"]
        })

    log_error("--- Finished Preprocessing Run ---")

    # Print summary
    print("\n" + "=" * 60)
    print(" PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Videos successfully processed : {stats['videos_processed']}")
    print(f"Videos failed to process      : {stats['videos_failed']}")
    print(f"Total frames extracted        : {stats['total_frames_extracted']}")
    print(f"Total faces detected & saved  : {stats['total_faces_detected']}")
    if stats['total_frames_extracted'] > 0:
        face_rate = (stats['total_faces_detected'] / stats['total_frames_extracted']) * 100
        print(f"Face detection successful rate: {face_rate:.2f}%")
    print(f"\nOutput saved to: {PROCESSED_DATA_DIR}")
    print(f"Error logs saved to: {LOG_FILE}\n")


if __name__ == "__main__":
    main()
