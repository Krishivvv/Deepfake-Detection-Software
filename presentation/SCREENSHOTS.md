# Screenshots — capture instructions

The presentation references three screenshots that need to be captured
on your machine. They go in `presentation/screenshots/`. The PPTX
generator picks them up automatically — re-run
`python presentation/generate_pptx.py` after you save them.

## File targets

| File | Slide it appears on |
|---|---|
| `presentation/screenshots/app_index.png` | Slide 12 — "Web demo — upload page" |
| `presentation/screenshots/app_result.png` | Slide 13 — "Web demo — prediction result" |

The PPTX already references these paths. If a file is missing, the
generator inserts a "screenshot placeholder" textbox in its place — so
the slide isn't broken until you fill it in.

## How to capture them

### 1. Start the app

```powershell
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\run_app.py
```

Wait for `Running on http://127.0.0.1:5000`.

### 2. Open the page in a clean browser

- Use a fresh window — close other tabs so the title bar is uncluttered.
- Zoom level 100 % (Ctrl+0).
- Resize the window to roughly 1280 × 900. The app is responsive but
  this size matches the 16:9 slide aspect.

Open <http://127.0.0.1:5000>.

### 3. Capture `app_index.png`

The empty upload page. **Snipping Tool** (Win+Shift+S) → Rectangular
snip → drag a rectangle that captures the navy header through the
"Analyze video" button.

Save to:
```
F:\ml_project\deepfake-detection\presentation\screenshots\app_index.png
```

### 4. Run a prediction (so the result card is visible)

In the same browser tab:

1. Click the upload box; pick a known-fake test video, e.g.
   `F:\ml_project\deepfake-detection\data\raw\fake\deepfakes\670_661.mp4`.
2. Click "Analyze video".
3. Wait ~15 seconds for the result.

You'll see a **FAKE** label with a high confidence bar (around 95 %),
the per-class probabilities, and the timing diagnostics.

### 5. Capture `app_result.png`

Same Snipping Tool flow. Crop a rectangle that includes the navy
header, the upload box (now showing the filename), and the full
result card. Save to:

```
F:\ml_project\deepfake-detection\presentation\screenshots\app_result.png
```

### 6. Regenerate the deck

```powershell
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\presentation\generate_pptx.py
```

Open `presentation/Final_Presentation.pptx` and confirm slides 12 and
13 now show the screenshots.

## Troubleshooting

- **Result card is empty after upload**: face-detection failed on the
  video. Try a different test video — `data/raw/fake/deepfakes/670_661.mp4`
  is one the model handles cleanly. If MTCNN returns no faces, the
  app shows a typed error message rather than a result card.
- **`presentation/screenshots/` doesn't exist**: create it manually:
  ```powershell
  New-Item -ItemType Directory -Path F:\ml_project\deepfake-detection\presentation\screenshots
  ```
- **Screenshot looks tiny in the slide**: PPTX scales the image to
  width 11 inches. If your capture is small, the deck will upscale it
  and look pixelated. Capture at native browser resolution and let
  PPTX downscale.

## Bonus: confusion-matrix and threshold-sweep images

These are auto-saved by the evaluation scripts and the PPTX already
references them:

- `outputs/hybrid_confusion_matrix.png` — slide 10
- `outputs/hybrid_threshold_sweep.png` — slide 11

If either is missing, regenerate them with:

```powershell
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\evaluate_hybrid_cached.py `
  --features-dir F:\ml_project\deepfake-detection\data\features_cnn `
  --head-checkpoint F:\ml_project\deepfake-detection\models\hybrid_v3_head.pth
```
