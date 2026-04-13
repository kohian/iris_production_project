# MLOps Production Pipeline

This project demonstrates a production-style machine learning pipeline, deliberately built around the simple Iris dataset.

Note: This project is intentionally over-engineered for a basic dataset.
The aim is not to showcase model complexity, but to demonstrate real-world MLOps and ML engineering practices such as modular pipeline design, packaging, testing, containerization, CI/CD, and cloud-based training workflows.

---

## Objectives

This project showcases an end-to-end ML pipeline with emphasis on:

* Structured and reproducible data processing
* Modular pipeline design
* Python package-based project organization
* Cross-validation based evaluation
* Hyperparameter tuning workflow
* Metrics logging and report generation
* Containerized execution with multi-stage Docker builds
* Automated linting and testing in CI
* Cloud container registry integration
* Cloud training workflow using Vertex AI

---

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── build_docker.yml
│
├── data/
│   ├── raw/
│   │   └── iris.csv
│   └── processed/
│       └── iris_processed.csv
│
├── model_artifacts/              # Not tracked in Git
├── reports/                      # Not tracked in Git
├── notebooks/
│
├── src/
│   ├── iris_production_project/
│   │   ├── config.py
│   │   ├── log_metrics.py
│   │   ├── make_dataset.py
│   │   ├── preprocess_funcs.py
│   │   ├── train_evaluate.py
│   │   │
│   │   ├── models/
│   │   │   ├── logistic_regression_model.py
│   │   │   └── xgboost_model.py
│   │   │
│   │   ├── tuning/
│   │   │   ├── build_model.py
│   │   │   ├── extract_best_params.py
│   │   │   ├── parse_args.py
│   │   │   ├── submit_hpt_job.py
│   │   │   └── train_tune.py
│   │
│   └── iris_production_project.egg-info/
│
├── tests/
│   ├── test_config.py
│   ├── test_log_metrics.py
│   ├── test_models.py
│   ├── test_preprocess_funcs.py
│   └── test_train_evaluate.py
│
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── README.md
├── requirements.txt
├── requirements_dev.txt
```

---

## Models

The project currently implements and compares:

### Logistic Regression

* Scikit-learn implementation
* Serves as a baseline model

### XGBoost

* Gradient boosting decision trees
* Strong performance for tabular classification problems

---

## Evaluation

Models are evaluated using Stratified K-Fold Cross Validation.

Metrics tracked include:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1 score (macro)

Results are exported as structured reports, while trained models are saved as artifacts outside version control.

---

## Hyperparameter Tuning (Vertex AI)

This project includes a hyperparameter tuning workflow using Vertex AI Hyperparameter Tuning Jobs.

### Tuning Components

Located under:

```bash
src/iris_production_project/tuning/
```

* `submit_hpt_job.py` – submits Vertex AI HPT job
* `train_tune.py` – training entry point for each trial
* `parse_args.py` – handles parameter parsing
* `build_model.py` – constructs model with given params
* `extract_best_params.py` – extracts best params to GCS

---

### Tuning Workflow

1. Submit tuning job:

```bash
python -m iris_production_project.tuning.submit_hpt_job
```

2. Vertex AI executes multiple trials:

* Each trial uses different hyperparameters
* Each trial performs cross-validation

3. Best parameters are extracted and saved to GCS:

```
gs://iris-csv/tuning/xgb/<timestamp>.json
```

---

## Final Model Training

The training pipeline has been updated to support retraining using tuned parameters.

### Command

```bash
python -m iris_production_project.train_evaluate \
    --model=xgb \
    --params-file=20260410-151741.json
```

### Behavior

* With `--params-file`:

  * Loads best parameters from GCS
  * Trains final model on full dataset

* Without:

  * Runs baseline model

---

## Important Concept

Hyperparameter tuning does NOT produce a final deployable model.

* 1 HPT trial = multiple models (cross-validation)
* HPT identifies best parameters only
* Final retraining is required for production

---

## Run Naming Strategy

* Tuned runs:

```
<params_filename>
```

* Baseline runs:

```
baseline_<timestamp>
```

---

## Updated Pipeline Flow

```
Hyperparameter Tuning (Vertex AI)
        ↓
Best Parameters (GCS JSON)
        ↓
Final Training (train_evaluate.py)
        ↓
Model Artifact + Metrics Logging
```

---

## Python Packaging

The project is structured as an installable Python package using `pyproject.toml`.

Run pipeline:

```bash
python -m iris_production_project.train_evaluate
```

---

## Docker Design

Multi-stage Dockerfile:

### Base

* Installs dependencies
* Copies source code
* Installs package

### Prod

* Runs training pipeline

### Dev

* Includes tests + linting

---

## Testing and Linting

Tools used:

* pytest
* ruff

Executed locally and in CI.

---

## CI/CD with GitHub Actions

Two-stage workflow:

### 1. Lint & Test

* Build dev container
* Run ruff
* Run pytest

### 2. Build & Push

* Authenticate via Workload Identity Federation
* Build Docker image
* Push to Artifact Registry

---

## Google Cloud Integration

Workflow:

1. GitHub Actions builds container
2. Push to Artifact Registry
3. Vertex AI pulls image
4. Training runs using custom container

---

## Artifacts and Reports

Excluded from Git:

* model_artifacts/
* reports/

Typically stored externally in production systems.

---

## Design Philosophy

This project focuses on engineering practices rather than model performance.

Core principles:

* separation of concerns
* reproducibility
* testability
* modular design
* container-first workflow
* package-based execution
* cloud compatibility

---

## Tech Stack

* Python
* scikit-learn
* XGBoost
* pandas
* numpy
* pytest
* ruff
* Docker
* GitHub Actions
* Google Artifact Registry
* Vertex AI

---

## Summary

Although this project uses the Iris dataset, it reflects how machine learning systems are built in production.

It demonstrates a complete workflow including:

* training and evaluation
* hyperparameter tuning
* retraining with best parameters
* containerization
* CI/CD
* cloud-based execution

This is a production-oriented MLOps project focused on system design rather than model complexity.
