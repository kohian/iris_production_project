from src.config import BUCKET, RAW_PATH, PROCESSED_PATH, MODEL_DIR, REPORTS_DIR


def test_bucket_path_prefix():
    assert BUCKET.startswith("gs://")


def test_config_paths_are_built_from_bucket():
    assert RAW_PATH == f"{BUCKET}/data/iris.csv"
    assert PROCESSED_PATH == f"{BUCKET}/data/iris_processed.csv"
    assert MODEL_DIR == f"{BUCKET}/model_artifacts"
    assert REPORTS_DIR == f"{BUCKET}/reports"