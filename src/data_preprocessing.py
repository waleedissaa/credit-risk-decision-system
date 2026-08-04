"""Utilities for preparing loan applicant data for model predictions."""

from __future__ import annotations

from typing import Any

import pandas as pd


FEATURE_NAMES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


CATEGORY_ENCODINGS = {
    "Gender": {
        "Female": 0,
        "Male": 1,
    },
    "Married": {
        "No": 0,
        "Yes": 1,
    },
    "Education": {
        "Graduate": 0,
        "Not Graduate": 1,
    },
    "Self_Employed": {
        "No": 0,
        "Yes": 1,
    },
    "Property_Area": {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2,
    },
}


def encode_category(feature_name: str, value: Any) -> int:
    """Convert a categorical value into the number expected by the model."""

    if feature_name not in CATEGORY_ENCODINGS:
        raise ValueError(f"Unknown categorical feature: {feature_name}")

    encoding = CATEGORY_ENCODINGS[feature_name]

    if value not in encoding:
        allowed_values = ", ".join(str(option) for option in encoding)
        raise ValueError(
            f"Invalid value for {feature_name}: {value}. "
            f"Expected one of: {allowed_values}"
        )

    return encoding[value]


def validate_non_negative(field_name: str, value: int | float) -> float:
    """Validate that a numerical applicant value is not negative."""

    numeric_value = float(value)

    if numeric_value < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return numeric_value


def prepare_applicant_data(
    *,
    gender: str,
    married: str,
    dependents: int,
    education: str,
    self_employed: str,
    applicant_income: int | float,
    coapplicant_income: int | float,
    loan_amount: int | float,
    loan_amount_term: int | float,
    credit_history: int | float,
    property_area: str,
) -> pd.DataFrame:
    """Prepare one applicant record in the format expected by the model."""

    if dependents not in {0, 1, 2, 3}:
        raise ValueError("Dependents must be 0, 1, 2, or 3.")

    if credit_history not in {0, 1, 0.0, 1.0}:
        raise ValueError("Credit history must be either 0 or 1.")

    validated_loan_term = validate_non_negative(
        "Loan amount term",
        loan_amount_term,
    )

    if validated_loan_term == 0:
        raise ValueError("Loan amount term must be greater than zero.")

    applicant = {
        "Gender": encode_category("Gender", gender),
        "Married": encode_category("Married", married),
        "Dependents": dependents,
        "Education": encode_category("Education", education),
        "Self_Employed": encode_category("Self_Employed", self_employed),
        "ApplicantIncome": validate_non_negative(
            "Applicant income",
            applicant_income,
        ),
        "CoapplicantIncome": validate_non_negative(
            "Coapplicant income",
            coapplicant_income,
        ),
        "LoanAmount": validate_non_negative(
            "Loan amount",
            loan_amount,
        ),
        "Loan_Amount_Term": validated_loan_term,
        "Credit_History": float(credit_history),
        "Property_Area": encode_category(
            "Property_Area",
            property_area,
        ),
    }

    return pd.DataFrame([applicant], columns=FEATURE_NAMES)