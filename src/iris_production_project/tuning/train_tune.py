from sklearn.model_selection import StratifiedKFold, cross_validate

from iris_production_project.preprocess_funcs import load_processed_data, split_features_target
from iris_production_project.tuning.build_model import build_model
from iris_production_project.tuning.parse_args import parse_args

try:
    from hypertune import HyperTune
    HPT_AVAILABLE = True
except ImportError:
    HPT_AVAILABLE = False


def main():
    args = parse_args()

    print(f"Trial parameters: {vars(args)}")

    df = load_processed_data()
    X, y = split_features_target(df)

    model = build_model(args)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring=["accuracy"],
        return_train_score=False,
    )

    mean_accuracy = scores["test_accuracy"].mean()

    print(f"Mean CV Accuracy: {mean_accuracy}")

    if HPT_AVAILABLE:
        hpt = HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="accuracy",
            metric_value=mean_accuracy,
            global_step=1,
        )


if __name__ == "__main__":
    main()