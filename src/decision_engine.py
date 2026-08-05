"""Convert model probabilities into clear loan recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import pandas as pd


APPROVE_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.40

DISCLAIMER = (
    "This recommendation is produced by an educational machine-learning "
    "prototype and should not be used as the sole basis for a real lending "
    "decision."
)


@dataclass(frozen=True)
class LoanDecision:
    """Structured result returned by the credit-risk decision engine."""

    recommendation: str
    approval_probability: float
    rejection_probability: float
    confidence: str
    risk_level: str
    explanation: str
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        """Return the decision as a dictionary."""

        return asdict(self)


def validate_thresholds(
    approve_threshold: float,
    review_threshold: float,
) -> None:
    """Validate the thresholds used for decision classification."""

    if not 0 <= review_threshold < approve_threshold <= 1:
        raise ValueError(
            "Thresholds must satisfy: "
            "0 <= review threshold < approve threshold <= 1."
        )


def validate_probability(probability: float) -> float:
    """Validate and return a model probability."""

    probability = float(probability)

    if not isfinite(probability):
        raise ValueError("Probability must be a finite number.")

    if not 0 <= probability <= 1:
        raise ValueError(
            "Probability must be between 0 and 1."
        )

    return probability


def determine_confidence(
    approval_probability: float,
) -> str:
    """Estimate confidence using distance from the 50% boundary."""

    distance_from_boundary = abs(
        approval_probability - 0.50
    )

    if distance_from_boundary >= 0.30:
        return "High"

    if distance_from_boundary >= 0.15:
        return "Moderate"

    return "Low"


def determine_risk_level(
    approval_probability: float,
    *,
    approve_threshold: float = APPROVE_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> str:
    """Convert approval probability into a simple risk level."""

    validate_thresholds(
        approve_threshold,
        review_threshold,
    )

    if approval_probability >= approve_threshold:
        return "Low"

    if approval_probability >= review_threshold:
        return "Moderate"

    return "High"


def create_decision(
    approval_probability: float,
    *,
    approve_threshold: float = APPROVE_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> LoanDecision:
    """Create a recommendation from an approval probability."""

    validate_thresholds(
        approve_threshold,
        review_threshold,
    )

    approval_probability = validate_probability(
        approval_probability
    )

    rejection_probability = 1 - approval_probability

    if approval_probability >= approve_threshold:
        recommendation = "Approve"
        explanation = (
            "The predicted approval probability is above "
            "the approval threshold."
        )

    elif approval_probability >= review_threshold:
        recommendation = "Manual Review"
        explanation = (
            "The predicted approval probability falls within "
            "the review range, so additional assessment is recommended."
        )

    else:
        recommendation = "Reject"
        explanation = (
            "The predicted approval probability is below "
            "the review threshold."
        )

    return LoanDecision(
        recommendation=recommendation,
        approval_probability=approval_probability,
        rejection_probability=rejection_probability,
        confidence=determine_confidence(
            approval_probability
        ),
        risk_level=determine_risk_level(
            approval_probability,
            approve_threshold=approve_threshold,
            review_threshold=review_threshold,
        ),
        explanation=explanation,
    )


def predict_decision(
    model: Any,
    applicant_data: pd.DataFrame,
    *,
    approve_threshold: float = APPROVE_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> LoanDecision:
    """Generate a decision for one prepared applicant record."""

    if not isinstance(applicant_data, pd.DataFrame):
        raise TypeError(
            "Applicant data must be a pandas DataFrame."
        )

    if len(applicant_data) != 1:
        raise ValueError(
            "Applicant data must contain exactly one row."
        )

    if applicant_data.isna().any().any():
        raise ValueError(
            "Applicant data cannot contain missing values."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The model must provide a predict_proba method."
        )

    if not hasattr(model, "classes_"):
        raise TypeError(
            "The trained model does not contain class labels."
        )

    model_classes = list(model.classes_)

    if 1 not in model_classes:
        raise ValueError(
            "The model does not contain approval class 1."
        )

    approval_class_index = model_classes.index(1)

    probabilities = model.predict_proba(
        applicant_data
    )

    if len(probabilities) != 1:
        raise ValueError(
            "The model returned an unexpected number of predictions."
        )

    approval_probability = float(
        probabilities[0][approval_class_index]
    )

    return create_decision(
        approval_probability,
        approve_threshold=approve_threshold,
        review_threshold=review_threshold,
    )