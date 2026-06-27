"""
Veridex — public inference demo (Hugging Face Spaces, Gradio SDK).

Single-file demo: upload a short video → the exact training preprocessing
(MTCNN face crop + ImageNet normalization) → the deployed ``hybrid_v3`` model
→ REAL / FAKE verdict, confidence, class probabilities, and a Grad-CAM heatmap
showing where the CNN backbone focused.

Preprocessing and inference reuse the production modules
(``app/utils/preprocessor.py`` and ``app/utils/predictor.py``) so the demo
distribution matches training exactly — no drift.

Weights are NOT bundled in git. They are pulled at startup from a Hugging Face
Hub model repo (or found locally under ``models/``). Configure via env vars:

    VERIDEX_HF_REPO       e.g. "krishivvv/veridex-deepfake"  (model repo)
    VERIDEX_BACKBONE_FILE default "cnn_baseline_best.pth"
    VERIDEX_HEAD_FILE     default "hybrid_v3_head.pth"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402

CFG = load_config()
DEVICE = "cpu"  # Spaces free tier is CPU; inference is fast enough.
MODELS_DIR = CFG.dir("models_dir")
THRESHOLD = float(CFG.model("hybrid_v3")["threshold"])


# --------------------------------------------------------------------------- #
# Weight resolution: local first, then Hugging Face Hub.
# --------------------------------------------------------------------------- #
def _resolve_weight(filename: str) -> Path:
    """Return a local path to ``filename``, downloading from the Hub if needed."""
    local = MODELS_DIR / filename
    if local.exists():
        return local
    repo = os.environ.get("VERIDEX_HF_REPO")
    if not repo:
        raise FileNotFoundError(
            f"{local} not found and VERIDEX_HF_REPO is unset. Either place the "
            f"weights under models/ or set VERIDEX_HF_REPO to a Hub model repo."
        )
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo, filename=filename)
    return Path(cached)


def _build():
    """Construct the predictor, preprocessor, and a Grad-CAM helper once."""
    from app.utils.predictor import build_predictor
    from app.utils.preprocessor import VideoPreprocessor

    backbone_file = os.environ.get("VERIDEX_BACKBONE_FILE", "cnn_baseline_best.pth")
    head_file = os.environ.get("VERIDEX_HEAD_FILE", "hybrid_v3_head.pth")
    backbone_path = _resolve_weight(backbone_file)
    head_path = _resolve_weight(head_file)

    predictor = build_predictor(
        "hybrid_v3",
        cnn_backbone_checkpoint=backbone_path,
        head_checkpoint=head_path,
        cnn_dropout=float(CFG.model("cnn")["dropout"]),
        cnn_trainable_backbone_layers=int(CFG.model("cnn")["trainable_backbone_layers"]),
        device=DEVICE,
    )
    preprocessor = VideoPreprocessor(
        num_frames=CFG.num_frames, image_size=CFG.image_size, device=DEVICE,
    )
    cam = GradCAM(backbone_path)
    return predictor, preprocessor, cam


# --------------------------------------------------------------------------- #
# Grad-CAM on the CNN-baseline backbone (layer4) for a single face frame.
# --------------------------------------------------------------------------- #
class GradCAM:
    """Grad-CAM for the frame-level CNN classifier (ResNet-50 ``layer4``).

    Uses the standalone CNN baseline (fc intact) so we have a per-frame fake
    logit to backprop. This visualises which facial region drove the
    frame-level fake score; it is an interpretability aid, not the deployed
    decision path (which is the LSTM over backbone features).
    """

    def __init__(self, backbone_checkpoint: Path) -> None:
        from src.models.resnet_classifier import DeepfakeClassifier

        self.model = DeepfakeClassifier(
            pretrained=False,
            dropout=float(CFG.model("cnn")["dropout"]),
            trainable_backbone_layers=1,
        ).to(DEVICE)
        ck = torch.load(backbone_checkpoint, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
        self.model.eval()
        self.target_layer = self.model.backbone.layer4
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self.target_layer.register_forward_hook(self._fwd_hook)
        self.target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _m, _i, output):
        self._activations = output.detach()

    def _bwd_hook(self, _m, _gi, grad_output):
        self._gradients = grad_output[0].detach()

    def heatmap(self, frame: torch.Tensor) -> np.ndarray:
        """``frame``: (3, H, W) normalized tensor. Returns HxW float map in [0, 1]."""
        x = frame.unsqueeze(0).to(DEVICE).requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).view(-1)[0]  # fake logit
        logit.backward()
        # weights = global-avg-pooled gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)      # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)[0, 0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def _overlay_cam(frame_tensor: torch.Tensor, cam: np.ndarray) -> np.ndarray:
    """Denormalize a frame and overlay the CAM as a heatmap; returns HxWx3 uint8 RGB."""
    import cv2

    mean = np.array(CFG.imagenet_mean).reshape(3, 1, 1)
    std = np.array(CFG.imagenet_std).reshape(3, 1, 1)
    img = (frame_tensor.cpu().numpy() * std + mean).clip(0, 1)  # (3, H, W)
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)       # (H, W, 3) RGB
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8)
    return overlay


# Lazy globals so import is cheap (Spaces imports the module to read config).
_STATE: dict = {}


def _ensure_loaded():
    if "predictor" not in _STATE:
        predictor, preprocessor, cam = _build()
        _STATE.update(predictor=predictor, preprocessor=preprocessor, cam=cam)
    return _STATE["predictor"], _STATE["preprocessor"], _STATE["cam"]


def analyze(video_path: str):
    """Gradio callback. Returns (label_markdown, prob_dict, gradcam_image)."""
    if not video_path:
        return "### Please upload a video.", {}, None
    predictor, preprocessor, cam = _ensure_loaded()

    try:
        clip, info = preprocessor.preprocess(video_path)
    except Exception as exc:  # noqa: BLE001 — surface a friendly message
        return f"### ⚠️ {exc}", {}, None

    result = predictor.predict(clip, threshold=THRESHOLD)
    label = result["label"]
    emoji = "🔴" if label == "FAKE" else "🟢"
    md = (
        f"## {emoji} {label}\n"
        f"**Confidence:** {result['confidence'] * 100:.1f}%  \n"
        f"**P(fake):** {result['probability_fake'] * 100:.1f}% · "
        f"**P(real):** {result['probability_real'] * 100:.1f}%  \n"
        f"_Threshold {result['threshold']:.3f} · {info['frames_with_faces']} "
        f"face frames · {result['inference_seconds']:.2f}s on CPU_"
    )
    probs = {"FAKE": result["probability_fake"], "REAL": result["probability_real"]}

    # Grad-CAM on the most-fake frame (best-effort; never breaks the verdict).
    gradcam_img = None
    try:
        frames = clip[0]  # (T, 3, H, W)
        with torch.no_grad():
            logits = cam.model(frames.to(DEVICE))
            idx = int(torch.sigmoid(logits).argmax().item())
        heat = cam.heatmap(frames[idx])
        gradcam_img = _overlay_cam(frames[idx], heat)
    except Exception:  # noqa: BLE001
        gradcam_img = None

    return md, probs, gradcam_img


def build_demo():
    import gradio as gr

    with gr.Blocks(title="Veridex — Deepfake Video Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Veridex — Deepfake Video Detector\n"
            "Upload a short clip with a visible face. Veridex samples frames, "
            "crops faces (MTCNN), and runs a ResNet-50 + BiLSTM model "
            "(`hybrid_v3`, test acc 0.80 / ROC-AUC 0.87). "
            "**Probabilistic aid — not for forensic use.**"
        )
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Video (mp4/mov/webm, ≥1s, one face)")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                probs = gr.Label(label="Class probabilities", num_top_classes=2)
                cam_out = gr.Image(label="Grad-CAM (where the model looked)")
        btn.click(analyze, inputs=video, outputs=[verdict, probs, cam_out])
        gr.Markdown(
            "Model card: [MODEL_CARD.md](https://github.com/Krishivvv/"
            "Deepfake-Detection-Software/blob/main/MODEL_CARD.md)"
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
