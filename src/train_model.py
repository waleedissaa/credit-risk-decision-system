"""Train and save the credit-risk classification model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import CATEGORY_ENCODINGS, FEATURE_NAMES


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIRECTORY
    / "data"
    / "raw"
    / "train_u6lujuX_CVtuZ9i.csv"
)

MODEL_DIRECTORY = ROOT_DIRECTORY / "models"
MODEL_PATH = MODEL_DIRECTORY / "credit_risk_model.pkl"

REPORTS_DIRECTORY = ROOT_DIRECTORY / "reports"
METRICS_PATH = REPORTS_DIRECTORY / "training_metrics.json"

TARGET_COLUMN = "Loan_Status"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the loan application training dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset was not found at: {path}"
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("The training dataset is empty.")

    return data


def validate_columns(data: pd.DataFrame) -> None:
    """Confirm that all required feature and target columns are present."""

    required_columns = set(FEATURE_NAMES + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing_list}"
        )


def encode_categorical_column(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """Encode a categorical training column using the application mappings."""

    mapping = CATEGORY_ENCODINGS[column_name]

    observed_values = set(values.dropna().unique())
    allowed_values = set(mapping)

    unexpected_values = observed_values.difference(allowed_values)

    if unexpected_values:
        unexpected_list = ", ".join(
            sorted(str(value) for value in unexpected_values)
        )
        raise ValueError(
            f"Unexpected values in {column_name}: {unexpected_list}"
        )

    return values.map(mapping)


def prepare_training_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert the raw dataset into model-ready features and labels."""

    validate_columns(data)

    features = data[FEATURE_NAMES].copy()

    for column_name in CATEGORY_ENCODINGS:
        features[column_name] = encode_categorical_column(
            features[column_name],
            column_name,
        )

    allowed_dependents = {"0", "1", "2", "3", "3+"}

    observed_dependents = set(
        features["Dependents"]
        .dropna()
        .astype(str)
        .unique()
    )

    unexpected_dependents = observed_dependents.difference(
        allowed_dependents
    )

    if unexpected_dependents:
        unexpected_list = ", ".join(
            sorted(unexpected_dependents)
        )
        raise ValueError(
            f"Unexpected Dependents values: {unexpected_list}"
        )

    features["Dependents"] = (
        features["Dependents"]
        .replace({"3+": 3})
    )

    numeric_columns = [
        "Dependents",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]

    for column_name in numeric_columns:
        features[column_name] = pd.to_numeric(
            features[column_name],
            errors="coerce",
        )

    target_mapping = {
        "N": 0,
        "Y": 1,
    }

    unexpected_targets = set(
        data[TARGET_COLUMN].dropna().unique()
    ).difference(target_mapping)

    if unexpected_targets:
        unexpected_list = ", ".join(
            sorted(str(value) for value in unexpected_targets)
        )
        raise ValueError(
            f"Unexpected target values: {unexpected_list}"
        )

    target = data[TARGET_COLUMN].map(target_mapping)

    if target.isna().any():
        raise ValueError(
            "The target column contains missing or invalid values."
        )

    return features[FEATURE_NAMES], target.astype(int)


def calculate_metrics(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Calculate validation metrics for the trained model."""

    predictions = model.predict(features)
    approval_probabilities = model.predict_proba(features)[:, 1]

    return {
        "accuracy": float(
            accuracy_score(target, predictions)
        ),
        "precision": float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                target,
                approval_probabilities,
            )
        ),
    }


def train_model() -> tuple[
    RandomForestClassifier,
    dict[str, object],
]:
    """Train the model and return it with its training metadata."""

    data = load_training_data()
    features, target = prepare_training_data(data)

    (
        training_features,
        validation_features,
        training_target,
        validation_target,
    ) = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    # Calculate missing-value replacements using only the training split.
    # This prevents information from the validation set leaking into training.
    imputation_values = training_features.median()

    training_features = training_features.fillna(
        imputation_values
    )

    validation_features = validation_features.fillna(
        imputation_values
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        training_features,
        training_target,
    )

    metrics = calculate_metrics(
        model,
        validation_features,
        validation_target,
    )

    feature_importances = {
        feature_name: float(importance)
        for feature_name, importance in zip(
            FEATURE_NAMES,
            model.feature_importances_,
        )
    }

    metadata: dict[str, object] = {
        "model_type": "RandomForestClassifier",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": str(DATA_PATH.relative_to(ROOT_DIRECTORY)),
        "total_rows": int(len(data)),
        "training_rows": int(len(training_features)),
        "validation_rows": int(len(validation_features)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "imputation_values": {
            feature_name: float(value)
            for feature_name, value in imputation_values.items()
        },
    }

    return model, metadata


def save_training_outputs(
    model: RandomForestClassifier,
    metadata: dict[str, object],
) -> None:
    """Save the trained model and its metrics report."""

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metadata,
            metrics_file,
            indent=2,
        )


def main() -> None:
    """Run the complete model-training workflow."""

    model, metadata = train_model()
    save_training_outputs(model, metadata)

    metrics = metadata["metrics"]

    print("Training completed successfully.")
    print(
        f"Training rows: {metadata['training_rows']}"
    )
    print(
        f"Validation rows: {metadata['validation_rows']}"
    )

    print("\nValidation metrics:")

    for metric_name, metric_value in metrics.items():
        readable_name = metric_name.replace("_", " ").title()
        print(
            f"  {readable_name}: {metric_value:.3f}"
        )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()