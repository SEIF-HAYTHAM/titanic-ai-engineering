from pathlib import Path

import dagshub
import joblib
import mlflow
import pandas as pd
from hydra import main
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def train(cfg: DictConfig):

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
    # 2. Load processed dataset
    # ============================================================

    input_path = Path("data/processed/titanic_processed.csv")

    print(f"Loading dataset from: {input_path}")

    df = pd.read_csv(input_path)

    print(f"Dataset shape: {df.shape}")

    # ============================================================
    # 3. Separate features and target
    # ============================================================

    target_column = "Survived"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ============================================================
    # 4. Train / test split
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=cfg.model.random_state,
        stratify=y,
    )

    # ============================================================
    # 5. Save train/test datasets
    # ============================================================

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    train_data[target_column] = y_train.values

    test_data = X_test.copy()
    test_data[target_column] = y_test.values

    train_data.to_csv(
        processed_dir / "train.csv",
        index=False,
    )

    test_data.to_csv(
        processed_dir / "test.csv",
        index=False,
    )

    print("Train/test datasets saved.")

    # ============================================================
    # 6. Create model
    # ============================================================

    model = RandomForestClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        min_samples_split=cfg.model.min_samples_split,
        random_state=cfg.model.random_state,
    )

    # ============================================================
    # 7. Start MLflow experiment
    # ============================================================

    with mlflow.start_run():

        # --------------------------------------------------------
        # Log experiment parameters
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Log preprocessing configuration
        # --------------------------------------------------------

        mlflow.log_param(
            "age_strategy",
            cfg.preprocessing.age_strategy,
        )

        mlflow.log_param(
            "fare_strategy",
            cfg.preprocessing.fare_strategy,
        )

        mlflow.log_param(
            "drop_missing_embarked",
            cfg.preprocessing.drop_missing_embarked,
        )

        # --------------------------------------------------------
        # Train model
        # --------------------------------------------------------

        print("Training Random Forest model...")

        model.fit(X_train, y_train)

        print("Model training completed.")

        # --------------------------------------------------------
        # Save model locally
        # --------------------------------------------------------

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "titanic_model.pkl"

        joblib.dump(model, model_path)

        print(f"Model saved to: {model_path}")

        # --------------------------------------------------------
        # Log model to MLflow
        # --------------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="titanic_model",
        )

        print("Model logged to MLflow.")


if __name__ == "__main__":
    train()