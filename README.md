# MLOps Pipeline Production Project

This project demonstrates a **production-style machine learning pipeline**, deliberately built around the simple Iris dataset.

> ⚠️ **Note:** This project is intentionally **over-engineered** for a basic dataset.
> The goal is not model complexity, but to showcase **real-world MLOps practices**, including pipeline design, reproducibility, testing, and deployment readiness.

The Iris dataset is used purely as a controlled environment to focus on **machine learning engineering and system design**, rather than experimentation.

---

##  Objectives

This project showcases an end-to-end ML pipeline with emphasis on:

* Structured and reproducible data processing
* Modular pipeline design
* Cross-validation based evaluation
* Experiment logging and reporting
* Testable and maintainable codebase
* Containerized execution (Docker)
* Production-oriented project structure

---
##  Project Structure

```
.
├── data/
│   ├── raw/                      # Original dataset
│   │   └── iris.csv
│   └── processed/                # Processed dataset
│       └── iris_processed.csv
│
├── model_artifacts/              # (Not tracked in Git) saved models
├── reports/                      # (Not tracked in Git) evaluation outputs
│
├── notebooks/                    # Exploratory analysis
│
├── src/
│   └── iris_production_project/
│       ├── config.py
│       ├── make_dataset.py       # Data preparation
│       ├── preprocess_funcs.py   # Preprocessing logic
│       ├── train_evaluate.py     # Training + evaluation pipeline
│       ├── log_metrics.py        # Metrics logging
│       └── models/
│           ├── logistic_regression_model.py
│           └── xgboost_model.py
│
├── tests/                        # Unit tests
│   ├── test_config.py
│   ├── test_log_metrics.py
│   ├── test_models.py
│   ├── test_preprocess_funcs.py
│   └── test_train_evaluate.py
│
├── Dockerfile                    # Single container for pipeline execution
├── pyproject.toml                # Packaging + tooling config
├── requirements.txt
├── requirements_dev.txt
└── README.md
```

##  Models

The project currently implements and compares:

### Logistic Regression

* Scikit-learn implementation
* Serves as a baseline model

### XGBoost

* Gradient boosting decision trees
* Strong performance for tabular data

---

##  Evaluation

Models are evaluated using **Stratified K-Fold Cross Validation**.

Metrics tracked:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1 Score (macro)

Evaluation results are exported as structured reports (not committed to Git).

---

##  Setup (Local)

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

---

##  Running the Pipeline

### 1. Prepare Dataset

```bash
python -m iris_production_project.make_dataset
```

* Loads raw data
* Applies preprocessing
* Saves processed dataset

---

### 2. Train & Evaluate

```bash
python -m iris_production_project.train_evaluate
```

This step:

* Loads processed data
* Trains models
* Performs cross-validation
* Logs evaluation metrics
* Saves model artifacts (locally, not in repo)

---

##  Docker Usage

Build image:

```bash
docker build -t iris-ml-pipeline .
```

Run pipeline:

```bash
docker run --rm iris-ml-pipeline
```

The container executes:

```bash
python -m iris_production_project.train_evaluate
```

---

##  Testing & Code Quality

Development dependencies include:

* `pytest` for testing
* `ruff` for linting

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check src tests
```

---

##  Artifacts & Reports

The following are intentionally **excluded from version control**:

* `model_artifacts/` → trained models
* `reports/` → evaluation outputs

This mirrors production practice where:

* artifacts are stored in object storage (e.g., GCS, S3)
* reports are logged to tracking systems or dashboards

---

##  Design Philosophy

This project is not about achieving the best model performance.

Instead, it focuses on:

> “How would this look if it had to run in production?”

Key principles:

* Separation of concerns (data, models, pipeline)
* Reproducibility
* Testability
* Clear structure and extensibility
* Container-first execution

---

##  Tech Stack

* Python
* scikit-learn
* XGBoost
* pandas / numpy
* Docker
* pytest / ruff

---

##  Summary

This project demonstrates how to structure a **production-ready machine learning pipeline**, even for a simple dataset.

It is designed as a **portfolio project for MLOps / Machine Learning Engineering roles**, emphasizing system design and engineering practices over model complexity.
