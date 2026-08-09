from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


RAW_DATA = Path("data/raw/titanic.csv")
PROCESSED_DATA = Path("data/processed/titanic_processed.csv")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def preprocess_data(cfg: DictConfig):
    df = pd.read_csv(RAW_DATA)

    df = df[
        [
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ]
    ]

    # Age missing-value strategy
    if cfg.preprocessing.age_strategy == "median":
        df["Age"] = df["Age"].fillna(df["Age"].median())

    elif cfg.preprocessing.age_strategy == "mean":
        df["Age"] = df["Age"].fillna(df["Age"].mean())

    elif cfg.preprocessing.age_strategy == "drop":
        df = df.dropna(subset=["Age"])

    else:
        raise ValueError(
            f"Unknown age strategy: {cfg.preprocessing.age_strategy}"
        )

    # Fare missing-value strategy
    if cfg.preprocessing.fare_strategy == "median":
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    elif cfg.preprocessing.fare_strategy == "mean":
        df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

    else:
        raise ValueError(
            f"Unknown fare strategy: {cfg.preprocessing.fare_strategy}"
        )

    # Embarked handling
    if cfg.preprocessing.drop_missing_embarked:
        df = df.dropna(subset=["Embarked"])
    else:
        df["Embarked"] = df["Embarked"].fillna(
            df["Embarked"].mode()[0]
        )

    # Convert categorical variables to numerical values
    df = pd.get_dummies(
        df,
        columns=["Sex", "Embarked"],
        dtype=int,
    )

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)

    print("Preprocessing configuration:")
    print(f"  Age strategy: {cfg.preprocessing.age_strategy}")
    print(f"  Fare strategy: {cfg.preprocessing.fare_strategy}")
    print(
        f"  Drop missing Embarked: "
        f"{cfg.preprocessing.drop_missing_embarked}"
    )

    print(f"Processed dataset saved to {PROCESSED_DATA}")


if __name__ == "__main__":
    preprocess_data()