# Veridex — Deepfake Video Detection

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Next.js](https://img.shields.io/badge/next.js-14-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![Tailwind](https://img.shields.io/badge/tailwind-3.4-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A full-stack deepfake video detector. Drop a clip into the web UI; the
hybrid CNN-LSTM model returns a calibrated **REAL** / **FAKE** verdict
in ~15 seconds, all on your machine — no third-party APIs.

> **Headline numbers** (held-out 150-video test set):
> **82.0 % accuracy · ROC-AUC 0.870 · macro-F1 0.751 · **
> Architecture, training procedure and limitations are documented in
> [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) and
> [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

## Tech stack

**Frontend (`frontend/`)**
- Next.js 14 (App Router) + TypeScript 5
- Tailwind CSS 3 (custom emerald-on-charcoal theme)
- Vanilla React, no UI library — every component is hand-rolled

**Backend (`app/`)**
- Flask 3 + flask-cors
- SQLAlchemy 2 + SQLite (user accounts)
- flask-bcrypt (password hashing)
- PyJWT (HS256, 7-day tokens)

**Model (`src/`, `models/`)**
- PyTorch 2.x
- ResNet-50 backbone (last block fine-tuned on FaceForensics++)
- BiLSTM(128, 1, dropout 0.5) temporal head
- facenet-pytorch (MTCNN) face detection
- OpenCV frame extraction

## Quick start (5 minutes)

### Prereqs
- Python 3.10 / 3.11 / 3.12
- Node 18+ (tested on Node 24)
- ~2 GB free RAM, 4 GB free disk

### 1. Backend (Flask + model + auth)

```powershell
# from the project root
F:\ml_project\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_app.py
# Listens on http://127.0.0.1:5000
```

On startup you should see:
```
INFO app.auth: Auth initialised (sqlite at .../app/users.db)
INFO app: Model kind: hybrid_v3 | decision threshold: 0.575
INFO app.predictor: Loaded HybridV3Predictor (device=cpu) ...
 * Running on http://127.0.0.1:5000
```

### 2. Frontend (Next.js)

In a second terminal:

```powershell
cd frontend
copy .env.local.example .env.local      # macOS/Linux: cp
npm install
npm run dev
# Listens on http://127.0.0.1:3000
```

Open <http://127.0.0.1:3000> in any browser. You can:
- **Sign up** at `/signup` (creates a real account in `app/users.db`).
- **Sign in** at `/login`.
- **Detect** at `/detect` — upload an `.mp4` and get a real prediction.
- Read the model story on `/about`.

For a production-style run instead of `npm run dev`:
```powershell
npm run build
npm start
```

## Environment variables

### Frontend (`frontend/.env.local`)
| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:5000` | Base URL of the Flask backend |

### Backend (env vars override `app/config.py`)
| Variable | Default | Description |
|---|---|---|
| `APP_HOST` | `127.0.0.1` | Bind address |
| `APP_PORT` | `5000` | Bind port |
| `APP_DEBUG` | `0` | `1` enables Flask debug mode |
| `APP_MODEL_KIND` | `hybrid_v3` | `hybrid_v3`, `cnn`, or `hybrid` |
| `APP_SECRET_KEY` | dev placeholder | **Change before any non-local use.** Used to sign JWTs. |
| `APP_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated list of frontend origins allowed to call `/api/*`, `/predict`, `/health` |

## How the frontend talks to the backend

Single `lib/api.ts` client, base URL from `NEXT_PUBLIC_API_URL`. JWT is
held in `localStorage["veridex_token"]` and attached as `Authorization:
Bearer <token>` to every authenticated request. `lib/auth-context.tsx`
exposes `useAuth()` for components to read `user` / call
`signIn` / `signUp` / `signOut`. On boot, the context calls `/api/auth/me`
to refresh state from a persisted token.

### Endpoints used

| Method | Path | Body / Headers | Returns |
|---|---|---|---|
| POST | `/api/auth/signup` | `{email, password, name?}` | `{ok, token, user}` |
| POST | `/api/auth/login` | `{email, password}` | `{ok, token, user}` |
| GET | `/api/auth/me` | `Authorization: Bearer <token>` | `{ok, user}` |
| POST | `/api/auth/logout` | — | `{ok}` (stateless; client discards token) |
| POST | `/predict` | `multipart/form-data video=<file>`, optional `Authorization` | `PredictionResult` (see below) |
| GET | `/health` | — | `{ok, model_loaded}` |

### `PredictionResult` shape
```json
{
  "ok": true,
  "request_id": "85337c39",
  "label": "FAKE",
  "confidence_pct": 95.57,
  "probability_fake_pct": 95.57,
  "probability_real_pct": 4.43,
  "threshold": 0.575,
  "device": "cpu",
  "inference_seconds": 3.42,
  "total_seconds": 13.56,
  "video_info": {
    "total_video_frames": 714,
    "video_fps": 25.0,
    "frames_sampled": 32,
    "frames_with_faces": 32,
    "frames_padded": 0
  }
}
```

Errors always come back as
```json
{ "ok": false, "error": { "code": "no_faces", "message": "..." } }
```

## Folder structure

```
deepfake-detection/
├── frontend/                          Next.js 14 app
│   ├── app/                           App-router pages
│   │   ├── page.tsx                     Landing
│   │   ├── detect/page.tsx              Detector
│   │   ├── about/page.tsx               About
│   │   ├── login/page.tsx               Sign in
│   │   ├── signup/page.tsx              Sign up
│   │   ├── layout.tsx                   Root layout (Navbar, Footer, AuthProvider)
│   │   └── globals.css                  Tailwind layer + custom utilities
│   ├── components/                    Hand-rolled UI components
│   │   ├── Navbar.tsx                   Sticky, mobile menu, auth-aware
│   │   ├── Footer.tsx
│   │   ├── Hero.tsx                     Animated hero + preview card
│   │   ├── StatBar.tsx                  Animated counters
│   │   ├── UseCaseCards.tsx             Score-ring use cases
│   │   ├── HowItWorks.tsx               Numbered steps + trace mock
│   │   ├── FeatureGrid.tsx
│   │   ├── FAQAccordion.tsx
│   │   ├── FileUploader.tsx             Drag/drop + validation
│   │   └── ResultCard.tsx               Confidence gauge + breakdown
│   ├── lib/
│   │   ├── api.ts                       Centralised API client
│   │   └── auth-context.tsx             React auth context
│   ├── tailwind.config.ts
│   ├── package.json
│   └── .env.local.example
│
├── app/                               Flask backend
│   ├── app.py                           App factory + CORS + auth wiring
│   ├── auth.py                          User model + JWT signup/login/me
│   ├── config.py                        Settings + threshold loaders
│   ├── utils/
│   │   ├── preprocessor.py              Video → tensor pipeline
│   │   └── predictor.py                 CNN / Hybrid / Hybrid-v3 wrappers
│   ├── templates/                       Legacy Jinja templates (still served at /)
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js
│   │   ├── uploads/                     temp upload dir (gitignored)
│   │   └── ...
│   ├── logs/                            error.log, predictions.log (gitignored)
│   └── users.db                         SQLite user store (gitignored)
│
├── src/                               Core ML library
│   ├── data/                            Datasets (frame + video)
│   ├── models/                          ResNet, LSTM, Hybrid
│   └── preprocessing/                   Frame extraction, MTCNN, splits
│
├── data/                              splits/ + processed/ + features_cnn/
├── models/                            *.pth checkpoints
├── outputs/                           Evaluation reports, training history
├── presentation/                      .pptx + scripts + screenshots
├── requirements.txt
├── README.md                          ← this file
├── CODE_DOCUMENTATION.md
├── TRAINING_GUIDE.md
├── DEPLOYMENT_GUIDE.md
└── LICENSE
```

## Available scripts

### Frontend (`cd frontend`)
| Command | What it does |
|---|---|
| `npm run dev` | Start Next.js dev server with hot reload on `:3000` |
| `npm run build` | Production build (Tailwind tree-shake + type check) |
| `npm start` | Serve the production build on `:3000` |
| `npm run lint` | Next.js ESLint |

### Backend (project root)
| Command | What it does |
|---|---|
| `python run_app.py` | Start the Flask server on `:5000` |
| `python evaluate_hybrid_cached.py --features-dir data/features_cnn --head-checkpoint models/hybrid_v3_head.pth` | Re-run the test eval (~30 s) |
| `python presentation/generate_pptx.py` | Re-generate the slide deck from the latest evaluation outputs |

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the full training pipeline.

## Deployment

### Local (development)
Two processes on the same machine, as in Quick Start above.

### Local (production-style, single host)
```bash
cd frontend && npm run build && npm start &       # :3000
python run_app.py &                                # :5000
```

### Docker
A minimal Dockerfile lives in [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

### Vercel (frontend) + your own host (backend)
1. Deploy `frontend/` to Vercel; set `NEXT_PUBLIC_API_URL` in the project
   settings to the public URL of your backend.
2. On the backend host, set `APP_CORS_ORIGINS` to include your Vercel
   domain (e.g. `APP_CORS_ORIGINS=https://veridex.vercel.app`).
3. Run the backend behind a real WSGI server (e.g. `waitress` on
   Windows, `gunicorn` on Linux) and reverse-proxy through nginx /
   Caddy with HTTPS. Detailed checklist:
   [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#production-checklist).

## Screenshots

Capture two screenshots of the running app and drop them at the paths
below. The presentation generator and this README will pick them up
automatically.

```
docs/screenshots/landing.png       (the marketing landing)
docs/screenshots/detect-result.png (a finished prediction)
```

Step-by-step capture instructions:
[presentation/SCREENSHOTS.md](presentation/SCREENSHOTS.md).

## Limitations

- **Real-class recall is 73 %** — the model still mislabels a meaningful
  share of authentic videos as fake.
- Trained on a 700-video subset of FaceForensics++. Published numbers
  in the 90 %+ range fine-tune end-to-end on a GPU.
- Single-modality, single-face: no audio, no body, no compression-domain
  features.
- Not for forensic use. Predictions are probabilistic.

## Contributing

PRs welcome. The key paths:
- New model architecture → `src/models/` + a new branch in
  `app/utils/predictor.py` + `app/config.py`.
- New auth provider → extend `app/auth.py`; preserve the `/api/auth/*`
  contract documented above so the frontend keeps working.
- New page → add `frontend/app/<slug>/page.tsx` and a link in
  `frontend/components/Navbar.tsx`.

Run `python -m compileall .` and `cd frontend && npm run build` before
opening a PR.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Rössler et al., ICCV 2019
- [PyTorch](https://pytorch.org/) and the `torchvision` ResNet implementation
- [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) for MTCNN
- Frontend palette inspired by deepfakedetection.io's information
  density (but with our own emerald-on-charcoal identity)
