import argparse
import json

import gcsfs
from google.cloud import aiplatform

PROJECT = "iris-ml-production"
REGION = "us-central1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True, help="Full Vertex HPT job resource name")
    parser.add_argument("--output-path", required=True, help="GCS path to save best params JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    aiplatform.init(project=PROJECT, location=REGION)

    job = aiplatform.HyperparameterTuningJob.get(args.job_name)

    best_trial = max(
        job.trials,
        key=lambda t: next(m.value for m in t.final_measurement.metrics if m.metric_id == "accuracy"),
    )

    best_metric = next(
        m.value for m in best_trial.final_measurement.metrics if m.metric_id == "accuracy"
    )

    best_params = {
        p.parameter_id: p.value
        for p in best_trial.parameters
    }

    result = {
        "job_name": args.job_name,
        "best_metric_name": "accuracy",
        "best_metric_value": best_metric,
        "best_params": best_params,
    }

    print("Best trial metric:", best_metric)
    print("Best params:", best_params)

    fs = gcsfs.GCSFileSystem()
    with fs.open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved best params to: {args.output_path}")


if __name__ == "__main__":
    main()

# example usage
#     python extract_best_params.py \
#   --job-name "projects/iris-ml-production/locations/us-central1/hyperparameterTuningJobs/1234567890123456789" \
#   --output-path "gs://my-bucket/best_params/xgb/2026-04-10/best_params.json"