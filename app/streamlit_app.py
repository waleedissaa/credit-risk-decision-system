"""Streamlit interface for the credit risk decision system."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


# Add the project root to Python's import path.
ROOT_DIRECTORY = Path(__file__).resolve().parents[1]

if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))


from src.data_preprocessing import (  # noqa: E402
    FEATURE_NAMES,
    prepare_applicant_data,
)
from src.decision_engine import (  # noqa: E402
    LoanDecision,
    predict_decision,
)


MODEL_PATH = (
    ROOT_DIRECTORY
    / "models"
    / "credit_risk_model.pkl"
)


st.set_page_config(
    page_title="Credit Risk Decision System",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: Path) -> Any:
    """Load and cache the trained credit-risk model."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {model_path}"
        )

    return joblib.load(model_path)


def display_decision(decision: LoanDecision) -> None:
    """Display a formatted loan recommendation."""

    st.divider()
    st.subheader("Loan Decision Result")

    if decision.recommendation == "Approve":
        st.success(
            "Recommendation: Approve"
        )
    elif decision.recommendation == "Manual Review":
        st.warning(
            "Recommendation: Manual Review"
        )
    else:
        st.error(
            "Recommendation: Reject"
        )

    probability_column, confidence_column, risk_column = (
        st.columns(3)
    )

    probability_column.metric(
        "Approval Probability",
        f"{decision.approval_probability:.1%}",
    )

    confidence_column.metric(
        "Model Confidence",
        decision.confidence,
    )

    risk_column.metric(
        "Predicted Risk",
        decision.risk_level,
    )

    st.progress(
        decision.approval_probability,
        text=(
            "Predicted approval probability: "
            f"{decision.approval_probability:.1%}"
        ),
    )

    st.write(decision.explanation)

    with st.expander("Probability details"):
        probability_data = pd.DataFrame(
            {
                "Outcome": [
                    "Approval",
                    "Rejection",
                ],
                "Probability": [
                    decision.approval_probability,
                    decision.rejection_probability,
                ],
            }
        )

        st.dataframe(
            probability_data,
            hide_index=True,
            width="stretch",
        )

    st.info(decision.disclaimer)


def display_model_insights(model: Any) -> None:
    """Display model feature importance information."""

    st.subheader("Model Insights")

    if not hasattr(model, "feature_importances_"):
        st.warning(
            "Feature importance information is not "
            "available for this model."
        )
        return

    importance_data = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        by="Importance",
        ascending=False,
    )

    st.bar_chart(
        importance_data.set_index("Feature")
    )

    st.dataframe(
        importance_data,
        hide_index=True,
        width="stretch",
        column_config={
            "Feature": "Model Feature",
            "Importance": st.column_config.ProgressColumn(
                "Relative Importance",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
        },
    )

    st.caption(
        "Feature importance describes how strongly each "
        "input influenced the trained model overall. It does "
        "not explain the cause of an individual prediction."
    )


def main() -> None:
    """Run the Streamlit application."""

    st.title("Credit Risk Decision System")

    st.write(
        "Enter borrower information to receive a model-based "
        "loan recommendation and estimated approval probability."
    )

    st.caption(
        "Educational machine-learning project. This application "
        "is not intended for real lending decisions."
    )

    try:
        model = load_model(MODEL_PATH)
    except Exception as error:
        st.error(
            "The trained model could not be loaded."
        )

        st.exception(error)
        st.stop()

    decision_tab, insights_tab, information_tab = st.tabs(
        [
            "Loan Decision",
            "Model Insights",
            "About the Project",
        ]
    )

    with decision_tab:
        st.subheader("Borrower Information")

        with st.form("borrower_form"):
            personal_column, financial_column = st.columns(2)

            with personal_column:
                gender = st.selectbox(
                    "Gender",
                    options=[
                        "Male",
                        "Female",
                    ],
                )

                married = st.selectbox(
                    "Married",
                    options=[
                        "Yes",
                        "No",
                    ],
                )

                dependents = st.selectbox(
                    "Dependents",
                    options=[
                        0,
                        1,
                        2,
                        3,
                    ],
                    format_func=lambda value: (
                        "3+" if value == 3 else str(value)
                    ),
                )

                education = st.selectbox(
                    "Education",
                    options=[
                        "Graduate",
                        "Not Graduate",
                    ],
                )

                self_employed = st.selectbox(
                    "Self Employed",
                    options=[
                        "No",
                        "Yes",
                    ],
                )

                property_area = st.selectbox(
                    "Property Area",
                    options=[
                        "Urban",
                        "Semiurban",
                        "Rural",
                    ],
                )

            with financial_column:
                applicant_income = st.number_input(
                    "Applicant Income",
                    min_value=0.0,
                    value=5000.0,
                    step=500.0,
                )

                coapplicant_income = st.number_input(
                    "Coapplicant Income",
                    min_value=0.0,
                    value=0.0,
                    step=500.0,
                )

                loan_amount = st.number_input(
                    "Loan Amount",
                    min_value=0.0,
                    value=150.0,
                    step=10.0,
                )

                loan_amount_term = st.number_input(
                    "Loan Amount Term",
                    min_value=1.0,
                    value=360.0,
                    step=12.0,
                )

                credit_history = st.selectbox(
                    "Credit History",
                    options=[
                        1.0,
                        0.0,
                    ],
                    format_func=lambda value: (
                        "Meets credit history requirements"
                        if value == 1.0
                        else "Does not meet credit history requirements"
                    ),
                )

            submitted = st.form_submit_button(
                "Generate Loan Decision",
                type="primary",
                width="stretch",
            )

        if submitted:
            try:
                applicant_data = prepare_applicant_data(
                    gender=gender,
                    married=married,
                    dependents=dependents,
                    education=education,
                    self_employed=self_employed,
                    applicant_income=applicant_income,
                    coapplicant_income=coapplicant_income,
                    loan_amount=loan_amount,
                    loan_amount_term=loan_amount_term,
                    credit_history=credit_history,
                    property_area=property_area,
                )

                decision = predict_decision(
                    model,
                    applicant_data,
                )

                display_decision(decision)

                with st.expander(
                    "Prepared model input"
                ):
                    st.dataframe(
                        applicant_data,
                        hide_index=True,
                        width="stretch",
                    )

            except (TypeError, ValueError) as error:
                st.error(
                    f"Unable to generate a decision: {error}"
                )

            except Exception as error:
                st.error(
                    "An unexpected prediction error occurred."
                )

                st.exception(error)

    with insights_tab:
        display_model_insights(model)

    with information_tab:
        st.subheader("How the System Works")

        st.write(
            "The application prepares the borrower information, "
            "passes it to a trained Random Forest classifier, and "
            "converts the predicted approval probability into one "
            "of three recommendations: Approve, Manual Review, or "
            "Reject."
        )

        st.subheader("Project Limitations")

        st.write(
            "The model was trained on a small historical dataset. "
            "Its predictions may contain errors or reflect patterns "
            "and biases in the training data. Sensitive personal "
            "characteristics should not be used without appropriate "
            "legal, ethical, and fairness review."
        )

        st.subheader("Technology")

        st.write(
            "Python, pandas, scikit-learn, Joblib, and Streamlit."
        )


if __name__ == "__main__":
    main()