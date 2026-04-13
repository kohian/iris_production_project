# MLOps Production Pipeline

This project demonstrates a production-style machine learning pipeline, deliberately built around the simple Iris dataset.

Note: This project is intentionally over-engineered for a basic dataset.
The aim is not to showcase model complexity, but to demonstrate real-world MLOps and ML engineering practices such as modular pipeline design, packaging, testing, containerization, CI/CD, and cloud-based training workflows.

The Iris dataset is used as a controlled example so that the focus stays on engineering, reproducibility, and deployment-oriented design rather than on solving a difficult ML problem.

---

## Objectives

This project showcases an end-to-end ML pipeline with emphasis on:

* Structured and reproducible data processing
* Modular pipeline design
* Python package-based project organization
* Cross-validation based evaluation
* Hyperparameter tuning workflow (Vertex AI)
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
├── pip_freeze_dump.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── requirements_dev.txt
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

Evaluation results are exported as structured reports, while trained models are saved as artifacts outside version control.

---

## Hyperparameter Tuning (Vertex AI)

This project includes a hyperparameter tuning workflow using Vertex AI Hyperparameter Tuning Jobs.

### Tuning Components

Located under:

```
src/iris_production_project/tuning/
```

* `submit_hpt_job.py`
  Submits a Vertex AI Hyperparameter Tuning job

* `train_tune.py`
  Entry point executed by each Vertex AI trial

* `parse_args.py`
  Handles parameter parsing for tuning runs

* `build_model.py`
  Constructs model instances based on provided hyperparameters

* `extract_best_params.py`
  Extracts the best-performing parameters and stores them in GCS

---

### Tuning Workflow

1. Submit HPT job:

```bash
python -m iris_production_project.tuning.submit_hpt_job
```

2. Vertex AI runs multiple trials:

* Each trial uses a different hyperparameter configuration
* Each trial performs cross-validation

3. Best parameters are extracted and saved to GCS:

```
gs://iris-csv/tuning/xgb/<timestamp>.json
```

---

## Final Model Training with Tuned Parameters

The training pipeline has been extended to support retraining using optimal hyperparameters from tuning.

### Command

```bash
python -m iris_production_project.train_evaluate \
    --model=xgb \
    --params-file=20260410-151741.json
```

### Behavior

* If `--params-file` is provided:

  * Loads hyperparameters from GCS
  * Trains model on full dataset using tuned parameters

* If not provided:

  * Runs baseline training using default parameters

---

## Important Concept

Hyperparameter tuning does **not** produce a final deployable model.

* Each HPT trial performs cross-validation
* Each trial produces multiple models
* The output of HPT is **best parameters**, not a trained model

Therefore:

* A final training step is required using the best parameters on the full dataset

---

## Run Naming Strategy

To ensure traceability:

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

This allows execution via:

```bash
python -m iris_production_project.train_evaluate
```

Inside Docker:

```bash
pip install --no-cache-dir --no-deps .
```

---

## Local Setup

Create virtual environment:

```bash
python -m venv .venv
```

Activate (Windows):

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install package:

```bash
pip install --no-deps .
```

Install dev dependencies:

```bash
pip install -r requirements_dev.txt
```

---

## Running the Pipeline Locally

Prepare dataset:

```bash
python -m iris_production_project.make_dataset
```

Train and evaluate:

```bash
python -m iris_production_project.train_evaluate
```

---

## Docker Design

Multi-stage Dockerfile:

### Base Stage

* Installs dependencies
* Copies source code
* Installs package

### Prod Stage

* Runs training pipeline

### Dev Stage

* Includes tests and linting

Build dev image:

```bash
docker build --target dev -t iris-dev .
```

Run tests:

```bash
docker run --rm iris-dev
```

Run lint:

```bash
docker run --rm iris-dev ruff check src tests
```

Build prod image:

```bash
docker build --target prod -t iris-ml-pipeline .
```

Run pipeline:

```bash
docker run --rm iris-ml-pipeline
```

---

## Testing and Linting

Tools used:

* pytest
* ruff

Test coverage includes:

* configuration
* preprocessing
* model definitions
* training pipeline
* metrics logging

---

## CI/CD with GitHub Actions

Two-stage workflow:

### 1. Lint and Test

* Build dev Docker image
* Run ruff
* Run pytest

### 2. Build and Push

* Authenticate via Workload Identity Federation
* Build Docker image
* Push to Artifact Registry

Images are tagged using commit SHA and `latest`.

---

## Google Cloud Integration

Workflow:

1. GitHub Actions builds container
2. Push to Artifact Registry
3. Vertex AI pulls container
4. Training runs using custom container

---

## Artifacts and Reports

Excluded from version control:

* model_artifacts/
* reports/

These are typically stored externally in production systems.

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

It demonstrates:

* cross-validation based evaluation
* hyperparameter tuning workflows
* retraining using optimal parameters
* containerization and CI/CD
* cloud-based training pipelines

This is a production-oriented MLOps project focused on system design rather than model complexity.
