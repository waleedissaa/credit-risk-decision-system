"""Evaluate the trained credit-risk model and save performance reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import FEATURE_NAMES
from src.train_model import (
    DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    load_training_data,
    prepare_training_data,
)


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIRECTORY / "models" / "credit_risk_model.pkl"

TRAINING_METRICS_PATH = (
    ROOT_DIRECTORY
    / "reports"
    / "training_metrics.json"
)

EVALUATION_DIRECTORY = (
    ROOT_DIRECTORY
    / "reports"
    / "evaluation"
)

EVALUATION_REPORT_PATH = (
    EVALUATION_DIRECTORY
    / "evaluation_report.json"
)

CONFUSION_MATRIX_PATH = (
    EVALUATION_DIRECTORY
    / "confusion_matrix.png"
)

ROC_CURVE_PATH = (
    EVALUATION_DIRECTORY
    / "roc_curve.png"
)

FEATURE_IMPORTANCE_PATH = (
    EVALUATION_DIRECTORY
    / "feature_importance.png"
)


def load_model() -> Any:
    """Load the previously trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_training_metadata() -> dict[str, Any]:
    """Load metadata produced during model training."""

    if not TRAINING_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Training metrics were not found. "
            "Run python3 -m src.train_model first."
        )

    with TRAINING_METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        return json.load(metadata_file)


def create_validation_set(
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Recreate the validation set used during training."""

    data = load_training_data(DATA_PATH)
    features, target = prepare_training_data(data)

    (
        _,
        validation_features,
        _,
        validation_target,
    ) = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    saved_imputation_values = metadata.get(
        "imputation_values"
    )

    if not saved_imputation_values:
        raise ValueError(
            "Training metadata does not contain "
            "imputation values."
        )

    imputation_values = pd.Series(
        saved_imputation_values,
        dtype=float,
    ).reindex(FEATURE_NAMES)

    if imputation_values.isna().any():
        missing_features = imputation_values[
            imputation_values.isna()
        ].index.tolist()

        raise ValueError(
            "Missing imputation values for: "
            + ", ".join(missing_features)
        )

    validation_features = validation_features.fillna(
        imputation_values
    )

    if validation_features.isna().any().any():
        raise ValueError(
            "Validation data still contains missing values."
        )

    return validation_features, validation_target


def calculate_evaluation_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    dict[str, float],
    dict[str, Any],
    pd.Series,
    pd.Series,
]:
    """Calculate predictions, probabilities, and performance metrics."""

    predictions = pd.Series(
        model.predict(features),
        index=target.index,
    )

    approval_probabilities = pd.Series(
        model.predict_proba(features)[:, 1],
        index=target.index,
    )

    metrics = {
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

    report = classification_report(
        target,
        predictions,
        target_names=[
            "Rejected",
            "Approved",
        ],
        output_dict=True,
        zero_division=0,
    )

    return (
        metrics,
        report,
        predictions,
        approval_probabilities,
    )


def save_confusion_matrix(
    target: pd.Series,
    predictions: pd.Series,
) -> None:
    """Create and save a confusion-matrix chart."""

    matrix = confusion_matrix(
        target,
        predictions,
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Rejected",
            "Approved",
        ],
    )

    display.plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title(
        "Credit Risk Model Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_roc_curve(
    target: pd.Series,
    probabilities: pd.Series,
) -> None:
    """Create and save a receiver operating characteristic curve."""

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    RocCurveDisplay.from_predictions(
        target,
        probabilities,
        name="Random Forest",
        ax=axis,
    )

    axis.set_title(
        "Credit Risk Model ROC Curve"
    )

    figure.tight_layout()

    figure.savefig(
        ROC_CURVE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_feature_importance(
    model: Any,
) -> None:
    """Create and save a model feature-importance chart."""

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "The trained model does not provide "
            "feature importances."
        )

    importance_data = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        by="Importance",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.barh(
        importance_data["Feature"],
        importance_data["Importance"],
    )

    axis.set_title(
        "Credit Risk Model Feature Importance"
    )

    axis.set_xlabel(
        "Importance"
    )

    axis.set_ylabel(
        "Feature"
    )

    figure.tight_layout()

    figure.savefig(
        FEATURE_IMPORTANCE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_evaluation_report(
    metrics: dict[str, float],
    classification_results: dict[str, Any],
    validation_rows: int,
) -> None:
    """Save the complete evaluation report as JSON."""

    report = {
        "model": "RandomForestClassifier",
        "validation_rows": validation_rows,
        "metrics": metrics,
        "classification_report": classification_results,
        "generated_files": {
            "confusion_matrix": str(
                CONFUSION_MATRIX_PATH.relative_to(
                    ROOT_DIRECTORY
                )
            ),
            "roc_curve": str(
                ROC_CURVE_PATH.relative_to(
                    ROOT_DIRECTORY
                )
            ),
            "feature_importance": str(
                FEATURE_IMPORTANCE_PATH.relative_to(
                    ROOT_DIRECTORY
                )
            ),
        },
    }

    with EVALUATION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=2,
        )


def main() -> None:
    """Run the full model-evaluation workflow."""

    EVALUATION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model()
    metadata = load_training_metadata()

    validation_features, validation_target = (
        create_validation_set(metadata)
    )

    (
        metrics,
        classification_results,
        predictions,
        approval_probabilities,
    ) = calculate_evaluation_metrics(
        model,
        validation_features,
        validation_target,
    )

    save_confusion_matrix(
        validation_target,
        predictions,
    )

    save_roc_curve(
        validation_target,
        approval_probabilities,
    )

    save_feature_importance(model)

    save_evaluation_report(
        metrics,
        classification_results,
        len(validation_features),
    )

    print("Evaluation completed successfully.")
    print(
        f"Validation rows: {len(validation_features)}"
    )

    print("\nEvaluation metrics:")

    for metric_name, metric_value in metrics.items():
        readable_name = (
            metric_name
            .replace("_", " ")
            .title()
        )

        print(
            f"  {readable_name}: {metric_value:.3f}"
        )

    print("\nSaved files:")
    print(f"  {EVALUATION_REPORT_PATH}")
    print(f"  {CONFUSION_MATRIX_PATH}")
    print(f"  {ROC_CURVE_PATH}")
    print(f"  {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()