# Deepfake-Detection-Software

## Deepfake Detection - Week 3 CNN Baseline

This week implements a frame-level **ResNet-50 transfer learning baseline** for binary deepfake detection (`0=real`, `1=fake`) using PyTorch.

## Added Components

- `src/data/dataset.py`: CSV-driven dataset loader with ImageNet normalization and train-time augmentation.
- `src/models/resnet_classifier.py`: ResNet-50 classifier with frozen early layers and dropout head.
- `train_cnn.py`: Local quick training script (small subset, default 3 epochs).
- `evaluate_cnn.py`: Test evaluation script with Accuracy, Precision, Recall, F1, and confusion matrix.
- `notebooks/Train_CNN_Baseline.ipynb`: Colab notebook for full GPU training with early stopping and scheduler.

## Dataset Assumptions

Expected project structure:

```text
F:\ml_project\deepfake-detection\
├── data\
│   ├── processed\
│   └── splits\train.csv, val.csv, test.csv
```

CSV path rows can be either:
- frame-level (e.g., `data/processed/real/video_001/frame_05.jpg`)
- video-level folder paths (e.g., `fake/deepfakes/241_210`) that are expanded into frame images

## Local Quick Training (Week 3 Local Verification)

Use your requested Python interpreter:

```powershell
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\train_cnn.py --project-root F:\ml_project\deepfake-detection
```

Default local behavior:
- batch size `32`
- epochs `3`
- small subset for fast verification
- saves best checkpoint to `models\cnn_baseline_best.pth`
- saves curves to `outputs\training_curves.png`

Useful overrides:

```powershell
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\train_cnn.py --project-root F:\ml_project\deepfake-detection --epochs 3 --batch-size 32 --num-workers 4
```

For full-data local run:

```powershell
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\train_cnn.py --project-root F:\ml_project\deepfake-detection --full-data --epochs 20
```

## Evaluation on Test Split

```powershell
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\evaluate_cnn.py --project-root F:\ml_project\deepfake-detection
```

Outputs:
- `outputs\cnn_evaluation.txt`
- `outputs\cnn_confusion_matrix.png`

## Colab Training (Production)

1. Open `notebooks/Train_CNN_Baseline.ipynb` in Google Colab.
2. Set `PROJECT_ROOT` in notebook to your Drive folder containing this project.
3. Run all cells.
4. Notebook will:
   - train for up to 20 epochs
   - apply early stopping (`patience=3`) on validation loss
   - apply LR scheduling (`ReduceLROnPlateau`)
   - save best model to `models/cnn_baseline_best.pth`
   - generate `outputs/training_curves.png`
   - evaluate on test split and save `outputs/cnn_evaluation.txt`
