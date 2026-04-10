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
├── notebooks/
├── reports/                      # Not tracked in Git
│
├── src/
│   ├── iris_production_project/
│   │   ├── config.py
│   │   ├── log_metrics.py
│   │   ├── make_dataset.py
│   │   ├── preprocess_funcs.py
│   │   ├── train_evaluate.py
│   │   └── models/
│   │       ├── logistic_regression_model.py
│   │       └── xgboost_model.py
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

Logistic Regression

* Scikit-learn implementation
* Serves as a baseline model

XGBoost

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

## Python Packaging

The project is structured as an installable Python package using `pyproject.toml`.

This allows the pipeline to be executed as:

```bash
python -m iris_production_project.train_evaluate
```

Inside the Docker image, the package is installed using:

```bash
pip install --no-cache-dir --no-deps .
```

---

## Local Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the project package:

```bash
pip install --no-deps .
```

Install development dependencies:

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

This performs data loading, preprocessing, model training, cross-validation, metric logging, and artifact generation.

---

## Docker Design

This project uses a single multi-stage Dockerfile.

Stages:

Base stage

* Installs runtime dependencies
* Copies source code
* Installs the project as a package

Prod stage

* Runs the training pipeline
* Entry point:
  `python -m iris_production_project.train_evaluate`

Dev stage

* Installs development dependencies
* Includes test suite
* Default command runs `pytest`

Build development image:

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

Build production image:

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

Testing and linting are executed both locally and within Docker, and are enforced in CI.

---

## CI/CD with GitHub Actions

The workflow consists of two jobs.

First job (lint and test):

* Builds the dev Docker image
* Runs ruff against src and tests
* Runs pytest inside the container

Second job (build and push):

* Authenticates to Google Cloud via Workload Identity Federation
* Builds the production Docker image
* Pushes the image to Google Artifact Registry

Images are tagged using both commit SHA and latest.

---

## Google Cloud Integration

The production container is pushed to Google Artifact Registry and used for Vertex AI custom training.

Workflow:

1. GitHub Actions builds and pushes the container
2. Vertex AI pulls the container image
3. Training runs using the packaged module entry point

---

## GitHub Actions Authentication and Permissions

Authentication is implemented using Workload Identity Federation.

Required permissions include:

* Artifact Registry Writer (to push images)
* Workload Identity User (to allow GitHub to impersonate the service account)

Additional IAM bindings are required to connect the GitHub identity provider to the service account.

---

## Artifacts and Reports

The following are excluded from version control:

* model_artifacts/
* reports/

These outputs are typically stored in external systems in production environments.

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

Although this project uses the Iris dataset, it is designed to reflect how machine learning systems are built in production.

It demonstrates a complete workflow including packaging, testing, containerization, CI/CD, and cloud-based training.

This is best understood as a deliberately over-engineered MLOps project focused on machine learning engineering practices rather than model complexity.
