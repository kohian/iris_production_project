# Iris Production ML Pipeline

A small end-to-end machine learning pipeline using the classic **Iris dataset**.  
The goal of this project is to demonstrate a **clean ML engineering workflow** including:

- dataset preparation
- preprocessing
- model training
- cross-validation evaluation
- experiment logging
- model artifact storage
- reproducible project structure

The project currently compares two models:

- Logistic Regression
- XGBoost

---

## Project Structure 

```
iris_production_project/
│
├── api/                         # FastAPI / inference service (future)
│
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Cleaned dataset
│
├── model_artifacts/             # Saved trained models (offline)
│   ├── logreg.joblib
│   └── xgb.joblib
│
├── notebooks/                   # Exploratory notebooks
│   ├── iris_to_csv.ipynb
│   ├── iris_training_logistic_regression.ipynb
│   └── iris_training_XGBoost.ipynb
│
├── reports/                     # Evaluation results
│   ├── logreg/                  # Logistic Regression CV metrics
│   └── xgb/                     # XGBoost CV metrics
│
├── src/
│   ├── models/                  # Model definitions
│   │   ├── logistic_regression_model.py
│   │   └── xgboost_model.py
│   │
│   ├── preprocess.py            # Data loading + preprocessing
│   ├── make_dataset.py          # Dataset preparation script
│   ├── train_evaluate.py        # Model training + cross validation
│   └── log_metrics.py           # Metric logging utilities
│
├── requirements.txt
├── Dockerfile
└── README.md
```


## Models

Currently implemented:

### Logistic Regression

* sklearn implementation
* baseline model

### XGBoost

* gradient boosting decision trees
* often strong for tabular data

Models are evaluated using **Stratified K-Fold cross validation** with the following metrics:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1 score (macro)

---

## Setup

Create a virtual environment:

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

## Running the Pipeline

### 1️⃣ Prepare the dataset

```bash
python -m src.make_dataset
```

This will create the processed dataset.

---

### 2️⃣ Train and evaluate models

```bash
python -m src.train_evaluate
```

This script:

* loads processed data
* saves trained models
* runs cross validation
* logs metrics

---

## Example Outputs

Models are saved to:

```
model_artifacts/
```

Evaluation reports are saved to:

```
reports/model_name
```

## Notebooks

The `notebooks` folder contains exploratory work:

* dataset preparation
* model experimentation
* baseline evaluation

---

## Future Improvements

Possible next steps:

* Add PyTorch neural network model
* Hyperparameter search
* Model comparison dashboard
* FastAPI inference service
* Dockerized deployment
* experiment tracking (MLflow / Weights & Biases)

---

## Technologies Used

* Python
* scikit-learn
* XGBoost
* pandas
* numpy
* joblib

---

## Purpose

This project demonstrates how to structure a **reproducible machine learning pipeline** suitable for production-style development.


