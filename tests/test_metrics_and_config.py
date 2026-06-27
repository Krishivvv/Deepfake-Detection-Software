"""Metrics computation and config loader sanity."""

from __future__ import annotations

from src.config import load_config
from src.evaluation.metrics import compute_binary_metrics, save_metrics


def test_perfect_predictions_score_one():
    y_true = [0, 0, 1, 1]
    y_prob = [0.01, 0.2, 0.8, 0.99]
    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.confusion_matrix == [[2, 0], [0, 2]]


def test_threshold_changes_decision():
    y_true = [0, 1]
    y_prob = [0.4, 0.6]
    assert compute_binary_metrics(y_true, y_prob, 0.5).accuracy == 1.0
    # Raise threshold above 0.6 -> the fake sample is now predicted real.
    assert compute_binary_metrics(y_true, y_prob, 0.7).accuracy == 0.5


def test_save_metrics_writes_files(tmp_path):
    y_true = [0, 1, 1]
    y_prob = [0.1, 0.7, 0.9]
    m = compute_binary_metrics(y_true, y_prob, 0.5)
    paths = save_metrics(m, y_true, y_prob, tmp_path, "unit", "UNIT TEST REPORT")
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0


def test_config_resolves_paths():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.dir("models_dir").is_absolute()
    hv3 = cfg.model("hybrid_v3")
    assert hv3["threshold"] == 0.575
    assert cfg.resolve(hv3["head_checkpoint"]).name == "hybrid_v3_head.pth"
