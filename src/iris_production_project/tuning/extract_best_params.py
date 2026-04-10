import argparse
import json

import gcsfs
from google.cloud import aiplatform

PROJECT = "iris-ml-production"
REGION = "us-central1"

# standard prefixes
JOB_PREFIX = f"projects/{PROJECT}/locations/{REGION}/hyperparameterTuningJobs/"
GCS_BASE_PATH = "gs://iris-csv/tuning/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="Only the numeric HPT job ID")
    parser.add_argument("--filename", required=True, help="Output JSON filename")
    return parser.parse_args()


def main():
    args = parse_args()

    # construct full values
    job_name = JOB_PREFIX + args.job_id
    output_path = GCS_BASE_PATH + args.filename

    print(f"Resolved job_name: {job_name}")
    print(f"Resolved output_path: {output_path}")

    aiplatform.init(project=PROJECT, location=REGION)

    job = aiplatform.HyperparameterTuningJob.get(job_name)

    # get best trial by accuracy
    best_trial = max(
        job.trials,
        key=lambda t: next(
            m.value for m in t.final_measurement.metrics if m.metric_id == "accuracy"
        ),
    )

    best_metric = next(
        m.value for m in best_trial.final_measurement.metrics if m.metric_id == "accuracy"
    )

    best_params = {
        p.parameter_id: p.value
        for p in best_trial.parameters
    }

    result = {
        "job_name": job_name,
        "best_metric_name": "accuracy",
        "best_metric_value": best_metric,
        "best_params": best_params,
    }

    print("Best trial metric:", best_metric)
    print("Best params:", best_params)

    fs = gcsfs.GCSFileSystem()
    with fs.open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved best params to: {output_path}")


if __name__ == "__main__":
    main()

# example usage
# python -m iris_production_project.tuning.extract_best_params --job-id 7333581324193628160 --filename xgb/20260410-151741.json