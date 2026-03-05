import json
from pathlib import Path
import pandas as pd
import numpy as np

def log_metrics(scores, model_name):
    REPORTS_DIR = Path(f"reports/{model_name}") 
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # scores is the dict from cross_validate(...)
    df_scores = pd.DataFrame(scores)                 # columns: fit_time, score_time, test_accuracy, ...
    df_scores.to_csv(REPORTS_DIR / "cv_scores.csv", index=False)

    summary = {}
    for col in df_scores.columns:
        if col.startswith("test_"):
            summary[col] = {
                "mean": float(np.mean(df_scores[col])),
                "std": float(np.std(df_scores[col], ddof=1)),
                "min": float(np.min(df_scores[col])),
                "max": float(np.max(df_scores[col])),
            }

    with open(REPORTS_DIR / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", REPORTS_DIR / "cv_scores.csv", "and", REPORTS_DIR / "cv_summary.json")