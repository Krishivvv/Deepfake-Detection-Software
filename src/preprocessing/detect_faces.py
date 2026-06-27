"""
detect_faces.py — Face detection and cropping using MTCNN.
==========================================================

Usage:
    from detect_faces import FaceDetector
    detector = FaceDetector()
    face_img = detector.detect_and_crop(frame)
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


class FaceDetector:
    """
    Wrapper around facenet-pytorch's MTCNN for detecting and cropping faces.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224), margin: int = 20, device: Optional[str] = None):
        """
        Initialize the MTCNN face detector.

        Parameters
        ----------
        target_size : tuple[int, int], default (224, 224)
            The final size (width, height) of the cropped face image.
        margin : int, default 20
            Pixels to add around the detected face bounding box before cropping.
        device : str, optional
            'cuda' or 'cpu'. If None, autodetects available GPU.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.target_size = target_size
        self.margin = margin

        # Initialize MTCNN
        # keep_all=False ensures we only get the most prominent face
        self.mtcnn = MTCNN(
            image_size=max(target_size),
            margin=margin,
            keep_all=False,
            device=self.device
        )

    def detect_and_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect a face in a BGR frame, crop it, and resize to target_size.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV BGR image array.

        Returns
        -------
        cropped_face : np.ndarray or None
            The cropped and resized face as a BGR numpy array.
            Returns None if no face is detected.
        """
        try:
            # facenet_pytorch expects RGB PIL Images or numpy arrays
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # MTCNN returns the cropped face as a PyTorch tensor by default,
            # but we can get bounding boxes instead to crop manually and securely resize.
            boxes, probs = self.mtcnn.detect(pil_img)

            if boxes is None or len(boxes) == 0:
                return None  # No face detected

            # Use the first (most prominent) face detected
            box = boxes[0]

            # Apply margin (MTCNN does this automatically if returning a tensor,
            # but since we process boxes manually for full control, we apply it here)
            x_left = int(max(0, box[0] - self.margin))
            y_top = int(max(0, box[1] - self.margin))
            x_right = int(min(frame.shape[1], box[2] + self.margin))
            y_bottom = int(min(frame.shape[0], box[3] + self.margin))

            # Crop the face from the original BGR frame
            face_crop = frame[y_top:y_bottom, x_left:x_right]

            # In case the crop is empty due to weird box coordinates
            if face_crop.size == 0:
                return None

            # Resize to the exact target size (224x224)
            face_resized = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_AREA)

            return face_resized

        except Exception:
            # We catch any unexpected exceptions (e.g., malformed image)
            # returning None allows the pipeline to log the error and continue
            return None


def detect_and_crop_face(frame: np.ndarray, target_size=(224, 224)) -> Optional[np.ndarray]:
    """
    Convenience function to detect and crop a face without manually instantiating
    the FaceDetector class. Note: instantiating MTCNN every time is slow,
    so for bulk processing, use the FaceDetector class directly.
    """
    detector = FaceDetector(target_size=target_size)
    return detector.detect_and_crop(frame)
