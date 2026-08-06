"""Tests for applicant data preprocessing."""

import pytest

from src.data_preprocessing import (
    FEATURE_NAMES,
    encode_category,
    prepare_applicant_data,
)


def create_valid_applicant():
    """Return a valid prepared applicant for testing."""

    return prepare_applicant_data(
        gender="Male",
        married="Yes",
        dependents=1,
        education="Graduate",
        self_employed="No",
        applicant_income=5000,
        coapplicant_income=1500,
        loan_amount=150,
        loan_amount_term=360,
        credit_history=1,
        property_area="Urban",
    )


def test_prepare_applicant_data_returns_expected_columns():
    """Prepared data should contain one row and all model features."""

    applicant = create_valid_applicant()

    assert applicant.shape == (1, len(FEATURE_NAMES))
    assert applicant.columns.tolist() == FEATURE_NAMES


def test_prepare_applicant_data_encodes_categories_correctly():
    """Categorical applicant values should use the expected encodings."""

    applicant = create_valid_applicant()
    row = applicant.iloc[0]

    assert row["Gender"] == 1
    assert row["Married"] == 1
    assert row["Education"] == 0
    assert row["Self_Employed"] == 0
    assert row["Property_Area"] == 2


def test_encode_category_rejects_invalid_value():
    """Unknown category values should raise a clear error."""

    with pytest.raises(ValueError, match="Invalid value for Gender"):
        encode_category("Gender", "Unknown")


def test_negative_income_is_rejected():
    """Applicant income cannot be negative."""

    with pytest.raises(
        ValueError,
        match="Applicant income cannot be negative",
    ):
        prepare_applicant_data(
            gender="Male",
            married="Yes",
            dependents=1,
            education="Graduate",
            self_employed="No",
            applicant_income=-100,
            coapplicant_income=1500,
            loan_amount=150,
            loan_amount_term=360,
            credit_history=1,
            property_area="Urban",
        )


def test_invalid_dependents_value_is_rejected():
    """Dependents must use one of the supported values."""

    with pytest.raises(
        ValueError,
        match="Dependents must be 0, 1, 2, or 3",
    ):
        prepare_applicant_data(
            gender="Male",
            married="Yes",
            dependents=5,
            education="Graduate",
            self_employed="No",
            applicant_income=5000,
            coapplicant_income=1500,
            loan_amount=150,
            loan_amount_term=360,
            credit_history=1,
            property_area="Urban",
        )


def test_zero_loan_term_is_rejected():
    """Loan terms must be greater than zero."""

    with pytest.raises(
        ValueError,
        match="Loan amount term must be greater than zero",
    ):
        prepare_applicant_data(
            gender="Male",
            married="Yes",
            dependents=1,
            education="Graduate",
            self_employed="No",
            applicant_income=5000,
            coapplicant_income=1500,
            loan_amount=150,
            loan_amount_term=0,
            credit_history=1,
            property_area="Urban",
        )