# Project Title
Smart Loan Advisor: An End-to-End Machine Learning System for Loan Approval Prediction
## Project Overview
Traditional loan approval systems often operate like a “black box,” leaving applicants unsure why they were accepted or rejected.
At the same time, financial institutions need fast, consistent, and reliable assessment tools. Applicants deserve transparency; lenders need efficiency. 
This project delivers both through an interpretable machine‑learning pipeline that provides accurate predictions and clear, actionable explanations.
## From Raw Data to Real Insights
I worked with 4,269 loan applications and 12 financial indicators, uncovering patterns that shaped the model: Credit score strongly separated approvals from rejections
Asset values played a bigger role than expected
Negative asset values required careful cleaning
Each feature reflected a real human story, income, ambition, risk and opportunity
- This shifted my approach from purely technical to human‑aware modelling.

## Methodology
1. Data Preparation
Cleaned and standardised column names
Fixed negative asset values using median imputation
Mapped loan status to binary labels
Ensured no missing values remained

2. Exploratory Analysis
Balanced dataset with no major class imbalance
Strong correlations between credit score, income, assets and approval
Visual patterns that aligned with real‑world lending logic

3. Model Development
I tested several algorithms, treating each like a different “loan officer”:
| Model               | Accuracy | Strength                     |
|--------------------|---------|------------------------------|
| XGBoost (Tuned)     | 97.9%   | Excellent pattern recognition |
| Random Forest       | 97.8%   | Robust and stable            |
| Gradient Boosting   | 97.2%   | Strong sequential learning   |
| Logistic Regression | 96.0%   | Transparent and interpretable |

## Beyond Accuracy
Key performance metrics:
Precision: 97.9%
Recall: 98.7%
ROC‑AUC: 0.998
F1 Score: 0.983
- The model wasn’t just accurate — it showed balanced, trustworthy decision‑making.

## Feature Importance
Top predictors:
Credit Score (81.9%)
Loan Term (6.2%)
Loan Amount (2.9%)
Annual Income (1.8%)
- These aligned closely with traditional lending principles.

## Streamlit Application
To make the model usable, I built an interactive Streamlit app that demonstrates transparent, explainable loan assessment.
Key Features
Real‑time loan assessment
Matching score (0–100) for intuitive interpretation
Personalised financial recommendations
Visual breakdown of strengths and risks
Downloadable assessment reports

## Architecture
Feature Engineering Pipeline
XGBoost Model
Matching Score Algorithm
Streamlit Frontend

##  What I Learned
Technical
Clean data = reliable models
Tree‑based models excel in financial prediction
Feature engineering improves interpretability
Deployment requires production‑ready thinking

## Try the Live Application
Your loan advisory system is fully deployed and ready to explore. 
Test different financial profiles, see real‑time predictions, and understand exactly how the model makes its decisions.

[👉 Live Demo Here](https://smart-loan-advisor-fiappwkmt8s3stsq6a82j6c.streamlit.app/)

## What You Can Do in the App
Run instant loan assessments with your own inputs
View approval probabilities and matching scores
See transparent explanations of how each feature influences the outcome
Get personalised recommendations to improve financial standing
Download a full assessment report for later review
