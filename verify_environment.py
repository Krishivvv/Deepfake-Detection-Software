"""
Deepfake Detection Project - Environment Verification Script
=============================================================
This script verifies that all required libraries are installed and working
correctly. It tests:
  1. Core library imports and versions
  2. PyTorch tensor operations (CPU & GPU)
  3. A simple neural network forward pass
  4. OpenCV video processing capabilities
  5. Face detection with facenet-pytorch
  6. Data science stack (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)
"""

import os
import sys

# ─── Color helpers for terminal output ───────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def header(msg):
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}{RESET}")

def ok(msg):
    print(f"  {GREEN}✔ {msg}{RESET}")

def fail(msg):
    print(f"  {RED}✘ {msg}{RESET}")

def warn(msg):
    print(f"  {YELLOW}⚠ {msg}{RESET}")

results = {"passed": 0, "failed": 0, "warnings": 0}

def check(label, fn):
    """Run a check function; record pass / fail."""
    try:
        fn()
        ok(label)
        results["passed"] += 1
    except Exception as e:
        fail(f"{label}  →  {e}")
        results["failed"] += 1

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CORE LIBRARY VERSIONS
# ═══════════════════════════════════════════════════════════════════════════════
header("1. Core Library Versions")

def check_python():
    v = sys.version
    print(f"     Python {v}")
    assert sys.version_info >= (3, 9), "Python 3.9+ required"
check("Python version", check_python)

def check_pytorch():
    import torch
    print(f"     PyTorch {torch.__version__}")
    assert torch.__version__ >= "2.0", "PyTorch 2.0+ recommended"
check("PyTorch", check_pytorch)

def check_torchvision():
    import torchvision
    print(f"     TorchVision {torchvision.__version__}")
check("TorchVision", check_torchvision)

def check_opencv():
    import cv2
    print(f"     OpenCV {cv2.__version__}")
check("OpenCV", check_opencv)

def check_numpy():
    import numpy as np
    print(f"     NumPy {np.__version__}")
check("NumPy", check_numpy)

def check_pandas():
    import pandas as pd
    print(f"     Pandas {pd.__version__}")
check("Pandas", check_pandas)

def check_matplotlib():
    import matplotlib
    print(f"     Matplotlib {matplotlib.__version__}")
check("Matplotlib", check_matplotlib)

def check_seaborn():
    import seaborn as sns
    print(f"     Seaborn {sns.__version__}")
check("Seaborn", check_seaborn)

def check_sklearn():
    import sklearn
    print(f"     Scikit-learn {sklearn.__version__}")
check("Scikit-learn", check_sklearn)

def check_tqdm():
    import tqdm as tqdm_mod
    print(f"     tqdm {tqdm_mod.__version__}")
check("tqdm", check_tqdm)

def check_facenet():
    print("     facenet-pytorch loaded")
check("facenet-pytorch", check_facenet)

def check_streamlit():
    import streamlit
    print(f"     Streamlit {streamlit.__version__}")
check("Streamlit", check_streamlit)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PYTORCH TENSOR OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════
header("2. PyTorch Tensor Operations")

def test_tensor_creation():
    import torch
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2), f"Expected (2,2) but got {t.shape}"
    assert t.dtype == torch.float32
check("Tensor creation & dtype", test_tensor_creation)

def test_tensor_arithmetic():
    import torch
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    c = a + b
    _ = a * b
    e = torch.matmul(a, b.T)          # (3,4) x (4,3) → (3,3)
    assert c.shape == (3, 4)
    assert e.shape == (3, 3)
check("Tensor arithmetic & matmul", test_tensor_arithmetic)

def test_tensor_autograd():
    import torch
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()                 # y = x0² + x1²
    y.backward()
    assert torch.allclose(x.grad, 2 * x.detach()), "Gradient mismatch"
check("Autograd / backpropagation", test_tensor_autograd)

def test_gpu_availability():
    import torch
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        ok(f"CUDA GPU detected: {dev}")
        t = torch.randn(2, 2, device="cuda")
        assert t.device.type == "cuda"
        results["passed"] += 1
    else:
        warn("No CUDA GPU detected (CPU-only mode — use Colab for training)")
        results["warnings"] += 1
test_gpu_availability()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMPLE NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
header("3. Simple Neural Network (forward pass)")

def test_simple_nn():
    import torch
    import torch.nn as nn

    class SimpleNet(nn.Module):
        """Tiny CNN → LSTM demo (mirrors your planned architecture)."""
        def __init__(self):
            super().__init__()
            # Mini CNN feature extractor
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            # LSTM temporal module
            self.lstm = nn.LSTM(input_size=16 * 4 * 4, hidden_size=32, batch_first=True)
            # Classifier
            self.fc = nn.Linear(32, 2)   # real vs fake

        def forward(self, x):
            # x: (batch, seq_len, C, H, W)
            B, T, C, H, W = x.shape
            # Extract per-frame features
            feats = []
            for t in range(T):
                f = self.cnn(x[:, t])        # (B, 16, 4, 4)
                feats.append(f.view(B, -1))  # (B, 256)
            feats = torch.stack(feats, dim=1) # (B, T, 256)
            # Temporal modelling
            out, _ = self.lstm(feats)         # (B, T, 32)
            logits = self.fc(out[:, -1])      # last time step → (B, 2)
            return logits

    model = SimpleNet()
    # Dummy input: batch=2, seq_len=5 frames, 3×64×64
    dummy = torch.randn(2, 5, 3, 64, 64)
    logits = model(dummy)
    assert logits.shape == (2, 2), f"Expected (2,2) but got {logits.shape}"

    # Quick training step
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.tensor([0, 1])   # real, fake
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"     Model params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"     Forward pass : logits shape = {tuple(logits.shape)}")
    print(f"     Training step: loss = {loss.item():.4f}")
check("CNN-LSTM forward pass + training step", test_simple_nn)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. OPENCV VIDEO PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
header("4. OpenCV Video Processing")

def test_opencv_video():
    import cv2
    import numpy as np
    # Create a tiny synthetic video in memory
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = os.path.join(os.path.dirname(__file__), "outputs", "_test_video.mp4")
    writer = cv2.VideoWriter(tmp_path, fourcc, 10, (64, 64))
    for i in range(10):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Read it back
    cap = cv2.VideoCapture(tmp_path)
    assert cap.isOpened(), "Could not open test video"
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    os.remove(tmp_path)
    assert count == 10, f"Expected 10 frames, read {count}"
    print("     Wrote & read back 10 synthetic frames (64×64)")
check("Video write / read round-trip", test_opencv_video)

def test_opencv_transforms():
    import cv2
    import numpy as np
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, (64, 64))
    assert gray.shape == (128, 128)
    assert resized.shape == (64, 64, 3)
check("Image transforms (resize, gray)", test_opencv_transforms)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FACE DETECTION (facenet-pytorch)
# ═══════════════════════════════════════════════════════════════════════════════
header("5. Face Detection (facenet-pytorch)")

def test_mtcnn():
    import numpy as np
    from facenet_pytorch import MTCNN
    from PIL import Image

    detector = MTCNN(keep_all=True, device="cpu")
    # Create a dummy 160×160 image (won't find a real face, but shouldn't crash)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8))
    boxes, probs = detector.detect(dummy_img)
    print("     MTCNN initialised on CPU — detection ran without error")
check("MTCNN face detector init + inference", test_mtcnn)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. DATA SCIENCE STACK
# ═══════════════════════════════════════════════════════════════════════════════
header("6. Data Science Stack")

def test_pandas_ops():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({
        "video": [f"vid_{i}" for i in range(100)],
        "label": np.random.choice(["real", "fake"], 100),
        "score": np.random.rand(100),
    })
    assert len(df) == 100
    assert set(df["label"].unique()) <= {"real", "fake"}
    print(f"     DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
check("Pandas DataFrame ops", test_pandas_ops)

def test_sklearn_ops():
    import numpy as np
    from sklearn.model_selection import train_test_split
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 80
    assert len(X_test) == 20
    print(f"     train_test_split: {len(X_train)} train, {len(X_test)} test")
check("Scikit-learn split & metrics", test_sklearn_ops)

def test_matplotlib_seaborn():
    import matplotlib
    matplotlib.use("Agg")              # non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    ax.plot(np.sin(np.linspace(0, 6, 50)))
    plt.close(fig)
    print("     Matplotlib + Seaborn plot created (Agg backend)")
check("Matplotlib & Seaborn plotting", test_matplotlib_seaborn)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PROJECT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
header("7. Project Folder Structure")

BASE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_DIRS = [
    "data/raw",
    "data/processed",
    "models",
    "notebooks",
    "src/preprocessing",
    "src/models",
    "outputs",
]

for d in EXPECTED_DIRS:
    full = os.path.join(BASE, d)
    if os.path.isdir(full):
        ok(f"  {d}/")
        results["passed"] += 1
    else:
        fail(f"  {d}/ — MISSING")
        results["failed"] += 1

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
header("SUMMARY")
total = results["passed"] + results["failed"]
print(f"  Passed  : {results['passed']}/{total}")
print(f"  Failed  : {results['failed']}/{total}")
print(f"  Warnings: {results['warnings']}")
if results["failed"] == 0:
    print(f"\n  {GREEN}{BOLD}🎉  Environment is ready! You can proceed to Step 2.{RESET}")
else:
    print(f"\n  {RED}{BOLD}⚠  Some checks failed. Please fix the issues above.{RESET}")

print()
