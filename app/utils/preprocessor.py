"""
Video → (1, T, 3, H, W) tensor pipeline for inference.

Reuses the existing training preprocessing utilities so the inference
distribution matches the training distribution exactly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.detect_faces import FaceDetector  # noqa: E402
from src.preprocessing.extract_frames import extract_frames  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

log = logging.getLogger("app.preprocessor")


class PreprocessingError(Exception):
    """User-presentable error raised when a video cannot be turned into a clip."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class VideoPreprocessor:
    """One-shot preprocessor that turns a video file into a model-ready tensor.

    Holds the (slow-to-init) MTCNN detector so subsequent uploads do not
    re-instantiate it.
    """

    def __init__(
        self,
        num_frames: int = 32,
        image_size: int = 224,
        face_margin: int = 20,
        device: Optional[str] = None,
    ) -> None:
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.transform = _eval_transform(image_size=image_size)
        self.detector = FaceDetector(
            target_size=(image_size, image_size),
            margin=face_margin,
            device=device,
        )

    def preprocess(self, video_path: str | Path) -> tuple[torch.Tensor, dict]:
        """Return ``(clip_tensor, info)`` where clip_tensor is (1, T, 3, H, W).

        ``info`` contains diagnostics: total_frames_in_video, frames_with_faces,
        sampled_frame_indices.
        """
        video_path = str(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise PreprocessingError(
                "video_unreadable",
                "Unable to read the uploaded video. The file may be corrupted "
                "or use an unsupported codec.",
            )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()

        if total_frames < 1:
            raise PreprocessingError(
                "video_empty",
                "The uploaded video contains zero frames.",
            )
        if fps > 0 and total_frames / fps < 0.5:
            raise PreprocessingError(
                "video_too_short",
                "Video is too short. Please upload a clip at least 1 second long.",
            )

        try:
            frames = extract_frames(
                video_path=video_path,
                output_dir="",  # save=False => not used
                n_frames=self.num_frames,
                save=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("extract_frames failed for %s", video_path)
            raise PreprocessingError(
                "frame_extraction_failed",
                "Error processing video. Please try again with a different file.",
            ) from exc

        if not frames:
            raise PreprocessingError(
                "frame_extraction_failed",
                "Could not read any frames from the video.",
            )

        face_crops_bgr: list[np.ndarray] = []
        for frame in frames:
            crop = self.detector.detect_and_crop(frame)
            if crop is not None:
                face_crops_bgr.append(crop)

        n_faces = len(face_crops_bgr)
        if n_faces == 0:
            raise PreprocessingError(
                "no_faces",
                "No faces found in the video. Please upload a video with "
                "visible faces.",
            )

        # Pad by repeating the last face if MTCNN missed some frames; this
        # mirrors the dataset's "repeat last frame" padding behaviour.
        while len(face_crops_bgr) < self.num_frames:
            face_crops_bgr.append(face_crops_bgr[-1])

        # Truncate to num_frames just in case more were collected.
        face_crops_bgr = face_crops_bgr[: self.num_frames]

        tensors: list[torch.Tensor] = []
        for bgr in face_crops_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self.transform(pil))

        clip = torch.stack(tensors, dim=0)            # (T, 3, H, W)
        clip = clip.unsqueeze(0)                      # (1, T, 3, H, W)

        info = {
            "total_video_frames": int(total_frames),
            "video_fps": fps,
            "frames_sampled": len(frames),
            "frames_with_faces": int(n_faces),
            "frames_padded": int(self.num_frames - n_faces) if n_faces < self.num_frames else 0,
        }
        return clip, info

    def thumbnails_from_video(self, video_path: str | Path, n: int = 6) -> list[bytes]:
        """Return up to ``n`` JPEG-encoded face thumbnails for the result page.

        On any error returns []; thumbnails are a UI nicety, not load-bearing.
        """
        try:
            frames = extract_frames(
                video_path=str(video_path),
                output_dir="",
                n_frames=max(n, 1),
                save=False,
            )
            crops: list[np.ndarray] = []
            for f in frames:
                c = self.detector.detect_and_crop(f)
                if c is not None:
                    crops.append(c)
                if len(crops) >= n:
                    break
            out: list[bytes] = []
            for c in crops:
                ok, buf = cv2.imencode(".jpg", c, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    out.append(bytes(buf))
            return out
        except Exception:  # noqa: BLE001
            log.exception("thumbnails_from_video failed")
            return []
