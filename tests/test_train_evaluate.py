import pandas as pd
from iris_production_project import train_evaluate as te


class DummyFS:
    def open(self, path, mode):
        class DummyFile:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, data):
                pass

        return DummyFile()


class DummyModel:
    def predict(self, X):
        return [0] * len(X)


def test_model_registry_contains_expected_models():
    assert "logreg" in te.MODEL_REGISTRY
    assert "xgb" in te.MODEL_REGISTRY


def test_main_runs_training_and_logging_for_logreg(monkeypatch):
    df = pd.DataFrame(
        {
            "f1": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            "f2": [0.1, 0.2, 1.0, 1.1, 2.0, 2.1],
            "target": [0, 0, 1, 1, 2, 2],
            "species_name": ["a", "a", "b", "b", "c", "c"],
        }
    )

    saved = {}
    logged = {}

    monkeypatch.setattr(te, "load_processed_data", lambda: df)

    monkeypatch.setattr(
        te,
        "cross_validate",
        lambda model, X, y, cv, scoring: {
            "test_accuracy": [0.9, 0.95, 1.0, 0.85, 0.9],
            "test_precision_macro": [0.9] * 5,
            "test_recall_macro": [0.9] * 5,
            "test_f1_macro": [0.9] * 5,
        },
    )

    monkeypatch.setattr(
        te,
        "log_metrics",
        lambda scores, model_name: logged.update(
            {"scores": scores, "model_name": model_name}
        ),
    )

    monkeypatch.setattr(te.gcsfs, "GCSFileSystem", lambda: DummyFS())
    monkeypatch.setattr(
        te.joblib,
        "dump",
        lambda model, file_obj: saved.update({"model": model}),
    )

    original_registry = te.MODEL_REGISTRY.copy()
    te.MODEL_REGISTRY["logreg"] = lambda X, y: DummyModel()

    monkeypatch.setattr(
        te.argparse.ArgumentParser,
        "parse_args",
        lambda self: type("Args", (), {"model": "logreg"})(),
    )

    try:
        te.main()
    finally:
        te.MODEL_REGISTRY.clear()
        te.MODEL_REGISTRY.update(original_registry)

    assert "model" in saved
    assert logged["model_name"] == "logreg"
    assert "test_accuracy" in logged["scores"]