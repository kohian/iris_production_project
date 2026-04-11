import io
import json
from contextlib import contextmanager

import pandas as pd

from iris_production_project import log_metrics as log_metrics_module


class DummyFS:
    def __init__(self, buffer):
        self.buffer = buffer

    @contextmanager
    def open(self, path, mode):
        yield self.buffer


def test_log_metrics_writes_summary_json(monkeypatch):
    scores = {
        "fit_time": [0.1, 0.2],
        "score_time": [0.01, 0.02],
        "test_accuracy": [0.9, 1.0],
        "test_f1_macro": [0.8, 0.9],
    }

    csv_calls = {}

    def fake_to_csv(self, path, index=False):
        csv_calls["path"] = path
        csv_calls["index"] = index
        csv_calls["columns"] = list(self.columns)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

    json_buffer = io.StringIO()

    monkeypatch.setattr(
        log_metrics_module.gcsfs,
        "GCSFileSystem",
        lambda: DummyFS(json_buffer),
    )

    log_metrics_module.log_metrics(scores, "logreg", "test_run")

    assert csv_calls["path"].endswith("/logreg/test_run_cv_scores.csv")
    assert csv_calls["index"] is False
    assert "test_accuracy" in csv_calls["columns"]
    assert "test_f1_macro" in csv_calls["columns"]

    json_buffer.seek(0)
    summary = json.load(json_buffer)

    assert summary["model"] == "logreg"
    assert summary["run_name"] == "test_run"
    assert "test_accuracy" in summary["metrics"]
    assert "test_f1_macro" in summary["metrics"]
    assert summary["metrics"]["test_accuracy"]["mean"] == 0.95
    assert summary["metrics"]["test_accuracy"]["min"] == 0.9
    assert summary["metrics"]["test_accuracy"]["max"] == 1.0