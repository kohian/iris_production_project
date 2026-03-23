import json

import gcsfs
import numpy as np
import pandas as pd

# from pathlib import Path
from src.config import REPORTS_DIR


def log_metrics(scores, model_name):
    # REPORTS_DIR = Path(f"reports/{model_name}") 
    # REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = f"{REPORTS_DIR}/{model_name}/cv_scores.csv"
    summary_path = f"{REPORTS_DIR}/{model_name}/cv_summary.json"


    # scores is the dict from cross_validate(...)
    # columns: fit_time, score_time, test_accuracy, ...
    df_scores = pd.DataFrame(scores)                 
    df_scores.to_csv(report_path, index=False)

    summary = {}
    for col in df_scores.columns:
        if col.startswith("test_"):
            summary[col] = {
                "mean": float(np.mean(df_scores[col])),
                "std": float(np.std(df_scores[col], ddof=1)),
                "min": float(np.min(df_scores[col])),
                "max": float(np.max(df_scores[col])),
            }

    # with open(summary_path, "w", encoding="utf-8") as f:
    #     json.dump(summary, f, indent=2)

    fs = gcsfs.GCSFileSystem()
    with fs.open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", summary_path, "and", summary_path)