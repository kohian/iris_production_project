import argparse
from datetime import datetime

from google.cloud import aiplatform

PROJECT_ID = "iris-ml-production"
LOCATION = "us-central1"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

CONTAINER_IMAGE_URI = (
    "us-central1-docker.pkg.dev/"
    "iris-ml-production/ian-docker-artifacts/iris-ml:latest"
)


def build_hpt_job_dict(model: str, display_name: str) -> dict:
    """
    Build a Vertex AI HyperparameterTuningJob payload.

    Notes:
    - parameter_id values must match argparse argument names in train_tune.py
    - the training code must report a scalar metric named "accuracy"
    """

    if model == "xgb":
        parameters = [
            {
                "parameter_id": "learning_rate",
                "double_value_spec": {"min_value": 0.01, "max_value": 0.3},
                "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE,
            },
            {
                "parameter_id": "max_depth",
                "integer_value_spec": {"min_value": 3, "max_value": 10},
                "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
            },
            {
                "parameter_id": "n_estimators",
                "integer_value_spec": {"min_value": 50, "max_value": 300},
                "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
            },
            {
                "parameter_id": "subsample",
                "double_value_spec": {"min_value": 0.6, "max_value": 1.0},
                "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
            },
            {
                "parameter_id": "colsample_bytree",
                "double_value_spec": {"min_value": 0.6, "max_value": 1.0},
                "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
            },
        ]
        base_args = ["--model", "xgb"]

    elif model == "logreg":
        parameters = [
            {
                "parameter_id": "C",
                "double_value_spec": {"min_value": 0.01, "max_value": 10.0},
                "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE,
            }
        ]
        base_args = ["--model", "logreg"]

    else:
        raise ValueError("model must be one of: xgb, logreg")

    job = {
        "display_name": display_name,
        "max_trial_count": 12,
        "parallel_trial_count": 3,
        "max_failed_trial_count": 3,
        "study_spec": {
            "metrics": [
                {
                    "metric_id": "accuracy",
                    "goal": aiplatform.gapic.StudySpec.MetricSpec.GoalType.MAXIMIZE,
                }
            ],
            "parameters": parameters,
        },
        "trial_job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-4",
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": CONTAINER_IMAGE_URI,
                        "command": [
                            "python",
                            "-m",
                            "iris_production_project.tuning.train_tune",
                        ],
                        "args": base_args,
                    },
                }
            ]
        },
    }

    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb", "logreg"], required=True)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--parallel-trials", type=int, default=3)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"iris-{args.model}-hpt-{timestamp}"

    client_options = {"api_endpoint": API_ENDPOINT}
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)

    job = build_hpt_job_dict(model=args.model, display_name=display_name)
    job["max_trial_count"] = args.max_trials
    job["parallel_trial_count"] = args.parallel_trials

    parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    response = client.create_hyperparameter_tuning_job(
        parent=parent,
        hyperparameter_tuning_job=job,
    )

    print("Created HyperparameterTuningJob")
    print(f"Display name: {response.display_name}")
    print(f"Resource name: {response.name}")
    print(
        f"State: {response.state.name if hasattr(response.state, 'name') else response.state}"
    )


if __name__ == "__main__":
    main()

# example: python -m iris_production_project.tuning.submit_hpt_job --model xgb --max-trials 4 --parallel-trials 2
