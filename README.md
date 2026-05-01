# Credit Risk Decision System

## Overview

This project builds a machine learning–based decision system to assess loan applications and predict approval likelihood. It combines predictive modeling with business decision rules and an interactive dashboard.

## Dashboard Preview
![Dashboard](reports/figures/dashboard.png)

## Business Insights
- Credit history was one of the most influential factors in loan approval predictions.
- The model combines machine learning outputs with decision logic to support loan assessment.
- The dashboard allows users to test borrower profiles and view approval probability in real time.

## Features

* Data cleaning and preprocessing pipeline
* Logistic Regression and Random Forest models
* Feature importance analysis
* Probability-based decision system
* Interactive Streamlit dashboard

## Tools & Technologies

* Python (Pandas, NumPy, Scikit-learn)
* Matplotlib
* Streamlit
* Joblib

## Results

* Achieved strong predictive performance on loan approval classification
* Identified key drivers of loan decisions (e.g., credit history, income, loan amount)
* Built a decision rule based on predicted probabilities

## How to Run

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
credit-risk-decision-system/
├── data/
├── notebooks/
├── src/
├── app/
├── models/
├── reports/
```
