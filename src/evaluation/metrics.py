"""
Centralised binary-classification metrics + report logging.

One implementation used by every evaluation path (CNN, hybrid, hybrid_v3) so
reports are consistent. Produces:

* a :class:`BinaryMetrics` dataclass (accuracy / precision / recall / F1 per
  class + macro, ROC-AUC, confusion matrix),
* a JSON dump,
* a human-readable ``.txt`` report (sklearn classification report),
* a confusion-matrix ``.png``.

The label convention is fixed across the project: ``real = 0``, ``fake = 1``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ("real", "fake")  # index 0 = real, index 1 = fake


@dataclass
class BinaryMetrics:
    """Container for a full binary-classification evaluation at one threshold."""

    threshold: float
    accuracy: float
    precision_real: float
    recall_real: float
    f1_real: float
    precision_fake: float
    recall_fake: float
    f1_fake: float
    macro_f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]
    support: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"acc={self.accuracy:.4f} macroF1={self.macro_f1:.4f} "
            f"ROC-AUC={self.roc_auc:.4f} "
            f"F1(real)={self.f1_real:.4f} F1(fake)={self.f1_fake:.4f} "
            f"@thr={self.threshold:.3f}"
        )


def compute_binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute all binary metrics from ground-truth labels and fake-class probabilities.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0 = real, 1 = fake).
    y_prob:
        Predicted probability of the fake (positive) class.
    threshold:
        Decision threshold applied to ``y_prob``.
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_prob, dtype=float)
    y_pred = (yp >= threshold).astype(int)

    try:
        roc_auc = float(roc_auc_score(yt, yp))
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(yt, y_pred, labels=[0, 1])
    return BinaryMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(yt, y_pred)),
        precision_real=float(precision_score(yt, y_pred, pos_label=0, zero_division=0)),
        recall_real=float(recall_score(yt, y_pred, pos_label=0, zero_division=0)),
        f1_real=float(f1_score(yt, y_pred, pos_label=0, zero_division=0)),
        precision_fake=float(precision_score(yt, y_pred, pos_label=1, zero_division=0)),
        recall_fake=float(recall_score(yt, y_pred, pos_label=1, zero_division=0)),
        f1_fake=float(f1_score(yt, y_pred, pos_label=1, zero_division=0)),
        macro_f1=float(f1_score(yt, y_pred, average="macro", zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=cm.tolist(),
        support={"real": int((yt == 0).sum()), "fake": int((yt == 1).sum())},
    )


def save_confusion_matrix_png(cm: Sequence[Sequence[int]], out_path: Path, title: str) -> None:
    """Render a 2x2 confusion matrix to ``out_path`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xticks([0, 1], ["Pred Real", "Pred Fake"])
    ax.set_yticks([0, 1], ["True Real", "True Fake"])
    color_thr = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(int(cm_arr[i, j]), "d"), ha="center", va="center",
                    color="white" if cm_arr[i, j] > color_thr else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics(
    metrics: BinaryMetrics,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    outputs_dir: Path,
    name: str,
    title: str,
    extra_header: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Persist a metrics bundle to ``outputs/`` and return the written paths.

    Writes ``<name>_metrics.json``, ``<name>_evaluation.txt`` and
    ``<name>_confusion_matrix.png``.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob, dtype=float) >= metrics.threshold).astype(int)

    json_path = outputs_dir / f"{name}_metrics.json"
    txt_path = outputs_dir / f"{name}_evaluation.txt"
    png_path = outputs_dir / f"{name}_confusion_matrix.png"

    json_path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")

    header_lines = [title, "=" * max(len(title), 40)]
    for k, v in (extra_header or {}).items():
        header_lines.append(f"{k}: {v}")
    body = [
        "",
        f"Threshold : {metrics.threshold:.3f}",
        f"Accuracy  : {metrics.accuracy:.4f}",
        f"Macro F1  : {metrics.macro_f1:.4f}",
        f"ROC-AUC   : {metrics.roc_auc:.4f}",
        f"F1 real   : {metrics.f1_real:.4f}   F1 fake   : {metrics.f1_fake:.4f}",
        f"Rec real  : {metrics.recall_real:.4f}   Rec fake  : {metrics.recall_fake:.4f}",
        "",
        "Confusion Matrix [[TN, FP], [FN, TP]]:",
        str(metrics.confusion_matrix),
        "",
        classification_report(yt, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES),
                              digits=4, zero_division=0),
    ]
    txt_path.write_text("\n".join(header_lines + body), encoding="utf-8")
    save_confusion_matrix_png(metrics.confusion_matrix, png_path, title)

    return {"json": json_path, "txt": txt_path, "png": png_path}
