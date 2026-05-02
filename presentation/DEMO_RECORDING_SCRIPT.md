# Demo Recording Script

A 3-minute end-to-end walkthrough you can record yourself with any
screen recorder (OBS, Windows `Win+G` Game Bar, Loom, etc.). The Flask
demo, the model load, and the inference all happen on this laptop —
no internet required during recording.

**Target length: 3 minutes.** Save as `presentation/Demo_Video.mp4`.

## Before you record

1. **Close everything else.** Browsers, VS Code, Slack — anything that
   could ping the network or pop a notification mid-recording. The
   first prediction takes ~15 s on this laptop; CPU contention will
   make it longer.
2. **Open a Powershell terminal** in `F:\ml_project\deepfake-detection`.
3. **Pre-warm the venv** so the activation doesn't add delay during recording:
   ```powershell
   F:\ml_project\.venv\Scripts\Activate.ps1
   python -c "import torch; print(torch.__version__)"
   ```
4. **Pick two test videos** beforehand and copy their paths to a notepad:
   - One real: `F:\ml_project\deepfake-detection\data\raw\real\672.mp4`
     (this one the model gets right)
   - One fake: `F:\ml_project\deepfake-detection\data\raw\fake\deepfakes\670_661.mp4`
     (this one the model gets right with very high confidence)
5. **Set screen recorder to 1080p, 30 fps** (or whatever your preset
   for clean text capture is).

## Recording — what to do, what to say

> **0:00 — Title card / intro**
>
> *On screen:* terminal window open in the project directory.
>
> *Say:* "This is the deepfake detection web demo. It uses a frame-level
> ResNet-50 CNN trained on a subset of FaceForensics++ to classify a
> short video as real or fake."

> **0:10 — Start the server**
>
> *Type and run:*
> ```powershell
> python run_app.py
> ```
>
> *Say while it's loading:* "On startup the app loads the trained CNN
> baseline and the MTCNN face detector. The model itself is loaded
> once, so subsequent predictions don't pay model-load latency."
>
> Wait for the line `Running on http://127.0.0.1:5000`.

> **0:30 — Open the browser**
>
> *On screen:* open `http://127.0.0.1:5000` in a fresh browser tab.
>
> *Say:* "The upload page accepts mp4, avi, mov, mkv and webm, capped
> at 50 megabytes. The model expects 32 face frames at 224 by 224, so
> very short clips without visible faces are rejected with a clear
> error message."

> **0:45 — Predict the first (real) video**
>
> *On screen:* drag-and-drop `data/raw/real/672.mp4` into the dropzone
> (or click and select).
>
> *Click* "Analyze video".
>
> *Say while waiting (~15 s):* "Behind the scenes the app extracts 32
> evenly spaced frames, runs each through MTCNN face detection, crops
> and normalises them, then runs the CNN once per frame and averages
> the per-frame fake probabilities into a single video-level decision."
>
> When the result appears: "Result: REAL with around 50 percent
> confidence — the model is closer to its decision boundary on this
> one. Total round-trip about 15 seconds on this laptop."

> **1:30 — Predict the fake video**
>
> *On screen:* upload `data/raw/fake/deepfakes/670_661.mp4`.
>
> *Click* "Analyze video".
>
> *Say:* "On a clear deepfake the model is much more confident."
>
> When the result appears: "FAKE at over 99 percent confidence — every
> frame agreed."

> **2:10 — Show the About page**
>
> *On screen:* click "About" in the top nav.
>
> *Say:* "The About page documents the deployed architecture, the
> calibrated decision threshold of 0.75 (chosen by sweeping
> thresholds on the validation set for macro-F1), and the actual test
> metrics — 77.58 percent accuracy, ROC-AUC 0.7657, with the standing
> limitation that real-class recall is around 42 percent because of
> the dataset's 4-to-1 fake-real skew."

> **2:40 — Show error handling (optional)**
>
> *On screen:* upload a non-video file (e.g. a `.txt` or `.zip`).
>
> *Say:* "Bad uploads are caught at the form level and surfaced inline
> — invalid extension, file too large, no faces detected, video too
> short, model failure all return typed error codes the frontend
> renders without a page reload."

> **2:55 — Outro**
>
> *Say:* "Source code, training scripts, evaluation reports and full
> documentation are in the project repository. The CPU training path
> is reproducible end-to-end on this laptop in about two hours;
> details are in TRAINING_GUIDE."

> **3:00 — End recording.**

## Cuts to make in post

- Trim the model-load wait at 0:30 if it's longer than ~5 s of dead air.
- If the first prediction takes >20 s, **cut the silent wait** — speed
  it up 2x or jump-cut to "Result:".
- Mute any system notification dings.

## Verifying the videos play correctly before recording

Test both videos at the command line first. The 9 out of 10 times
something goes wrong on stage, it's because the test videos themselves
aren't where you expected them.

```powershell
# Confirm files exist
Test-Path F:\ml_project\deepfake-detection\data\raw\real\672.mp4
Test-Path F:\ml_project\deepfake-detection\data\raw\fake\deepfakes\670_661.mp4

# Confirm the model is up
curl http://127.0.0.1:5000/health
# Should return {"model_loaded":true,"ok":true}
```

## If the predictions don't match the script

The model's behaviour is well-defined but not perfect. The two videos
chosen above are stable — but if for any reason the predictions
differ from what's described:

- **Real video predicted FAKE**: this happens to ~58 % of real videos
  (real recall is 42 %). Pick `data/raw/real/671.mp4` instead — the
  model also gets this one right.
- **Fake video predicted REAL**: rarer (fake recall is 87 %), but try
  `data/raw/fake/deepfakes/261_254.mp4` — it predicts FAKE at 97 %.

Always **rehearse the recording end-to-end at least once** before the
final take.
