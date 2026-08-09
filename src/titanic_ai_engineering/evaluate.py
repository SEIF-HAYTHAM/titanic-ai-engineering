from pathlib import Path
import json

import dagshub
import hydra
import joblib
import mlflow
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


TEST_DATA = Path("data/processed/test.csv")
MODEL_PATH = Path("models/titanic_model.pkl")
METRICS_PATH = Path("metrics/metrics.json")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def evaluate_model(cfg: DictConfig):

    # ============================================================
    # 1. Initialize DagsHub + MLflow
    # ============================================================

    dagshub.init(
        repo_owner="seifeldinhaytham",
        repo_name="titanic-ai-engineering",
        mlflow=True,
    )

    mlflow.set_experiment("Titanic Survival Prediction")

    # ============================================================
    # 2. Load test data
    # ============================================================

    test_df = pd.read_csv(TEST_DATA)

    X_test = test_df.drop(columns=["Survived"])
    y_test = test_df["Survived"]

    # ============================================================
    # 3. Load trained model
    # ============================================================

    model = joblib.load(MODEL_PATH)

    # ============================================================
    # 4. Make predictions
    # ============================================================

    predictions = model.predict(X_test)

    # ============================================================
    # 5. Calculate metrics
    # ============================================================

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # ============================================================
    # 6. Save metrics locally
    # ============================================================

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(METRICS_PATH, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ============================================================
    # 7. Log metrics to MLflow
    # ============================================================

    with mlflow.start_run(run_name="Titanic Evaluation"):

        # Log model configuration
        mlflow.log_param(
            "n_estimators",
            cfg.model.n_estimators,
        )

        mlflow.log_param(
            "max_depth",
            cfg.model.max_depth,
        )

        mlflow.log_param(
            "min_samples_split",
            cfg.model.min_samples_split,
        )

        mlflow.log_param(
            "random_state",
            cfg.model.random_state,
        )

        # Log evaluation metrics
        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        # Save metrics file as MLflow artifact
        mlflow.log_artifact(
            str(METRICS_PATH)
        )

    # ============================================================
    # 8. Print results
    # ============================================================

    print("\nEvaluation completed.")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print(
        f"Metrics saved to: {METRICS_PATH}"
    )

    print("Metrics logged to MLflow.")


if __name__ == "__main__":
    evaluate_model()