# =========================================================
# MODULE 1 — IMPORTS, CONFIG, CSS, LOGGING
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime
from fpdf import FPDF
import io
import time
import os
import logging

# Required for Plotly → PNG conversion
# (Make sure kaleido==0.2.1 is in requirements.txt)
pio.kaleido.scope.default_format = "png"

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartLoanAdvisor")

# ---------------------------------------------------------
# PROFESSIONAL UI CSS
# ---------------------------------------------------------
st.markdown("""
<style>

    /* HEADERS */
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--primary-text-color);
    }

    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        color: var(--secondary-text-color);
    }

    /* INFO BOXES */
    .tip-box {
        background-color: var(--background-secondary);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }

    .success-box {
        background-color: rgba(16, 185, 129, 0.12);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin: 0.5rem 0;
    }

    .warning-box {
        background-color: rgba(245, 158, 11, 0.12);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 0.5rem 0;
    }

    .info-box {
        background-color: rgba(14, 165, 233, 0.12);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0EA5E9;
        margin: 0.5rem 0;
    }

    /* METRICS */
    .stMetric {
        min-height: 100px;
    }

    .stMetric > div {
        min-height: 85px;
    }

    .stMetric label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    /* THEME VARIABLES */
    :root {
        --primary-color: #3B82F6;
        --secondary-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --background-primary: #FFFFFF;
        --background-secondary: #F3F4F6;
        --primary-text-color: #111827;
        --secondary-text-color: #374151;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #60A5FA;
            --secondary-color: #34D399;
            --warning-color: #FBBF24;
            --danger-color: #F87171;
            --background-primary: #1F2937;
            --background-secondary: #374151;
            --primary-text-color: #F9FAFB;
            --secondary-text-color: #D1D5DB;
        }
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODULE 2 — MODEL LOADING + FEATURE ENGINEERING
# =========================================================

# ---------------------------------------------------------
# STRICT MODEL LOADING (NO HEURISTIC FALLBACK)
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    """Load trained model and feature columns. Hard-stop if missing."""
    model_path = "best_xgb_model.pkl"
    features_path = "feature_columns.pkl"

    if not os.path.exists(model_path):
        st.error("❌ Model file 'best_xgb_model.pkl' not found.")
        st.stop()

    if not os.path.exists(features_path):
        st.error("❌ Feature file 'feature_columns.pkl' not found.")
        st.stop()

    try:
        model = joblib.load(model_path)
        feature_columns = joblib.load(features_path)
        logger.info("Model + feature columns loaded successfully.")
        return model, feature_columns
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        logger.error(f"Model loading error: {e}")
        st.stop()


model, feature_columns = load_model_and_features()


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def create_credit_features(df):
    """Create engineered features for credit assessment."""
    df_engineered = df.copy()

    # Monthly income + loan payment
    monthly_income = df_engineered['income_annum'] / 12
    monthly_payment = (df_engineered['loan_amount'] / df_engineered['loan_term']) / 12

    df_engineered['monthly_payment_ratio'] = (monthly_payment / monthly_income) * 100
    df_engineered['asset_coverage'] = (df_engineered['total_assets'] / df_engineered['loan_amount']) * 100
    df_engineered['loan_to_income'] = (df_engineered['loan_amount'] / df_engineered['income_annum']) * 100

    # Credit score categories
    def categorize(score):
        if score >= 900: return "Excellent"
        if score >= 800: return "Very Good"
        if score >= 700: return "Good"
        if score >= 600: return "Fair"
        return "Needs Improvement"

    df_engineered['credit_category'] = df_engineered['credit_score'].apply(categorize)

    # Stability score
    df_engineered['stability_score'] = 0
    df_engineered.loc[df_engineered['self_employed'] == 'Employed', 'stability_score'] += 30
    df_engineered.loc[df_engineered['education'] == 'Graduate', 'stability_score'] += 20
    df_engineered.loc[df_engineered['no_of_dependents'] <= 2, 'stability_score'] += 10

    return df_engineered


# ---------------------------------------------------------
# MATCHING SCORE (0–100)
# ---------------------------------------------------------
def calculate_matching_score(row):
    """Calculate a 0–100 matching score based on financial profile."""
    score = 0

    # Credit score weight
    if row['credit_score'] >= 900: score += 40
    elif row['credit_score'] >= 800: score += 35
    elif row['credit_score'] >= 700: score += 30
    elif row['credit_score'] >= 600: score += 20
    else: score += 10

    # Affordability
    if row['monthly_payment_ratio'] <= 25: score += 25
    elif row['monthly_payment_ratio'] <= 35: score += 20
    elif row['monthly_payment_ratio'] <= 45: score += 15
    elif row['monthly_payment_ratio'] <= 55: score += 10
    else: score += 5

    # Asset coverage
    if row['asset_coverage'] >= 250: score += 20
    elif row['asset_coverage'] >= 175: score += 16
    elif row['asset_coverage'] >= 125: score += 12
    elif row['asset_coverage'] >= 75: score += 8
    else: score += 4

    # Stability
    score += min(row['stability_score'], 15)

    # Risk adjustments
    if row['self_employed'] == 'Self-Employed' and row['income_annum'] < 30000:
        score -= 10

    if row['no_of_dependents'] > 3:
        score -= 5

    if row['loan_term'] > 7 and row['loan_amount'] < 150000:
        score -= 3

    return max(0, min(score, 100))


# ---------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------
def generate_recommendations(score, features, applicant):
    """Generate personalised recommendations based on financial profile."""
    recs = []

    # CREDIT SCORE
    cs = features['credit_score']
    if cs >= 800:
        recs.append({
            "title": "Excellent Credit Standing",
            "message": f"Your credit score of {cs} places you in the top tier.",
            "actions": [
                "Negotiate for premium interest rates.",
                "Maintain low credit utilisation.",
                "Avoid unnecessary new credit applications."
            ]
        })
    elif cs >= 700:
        recs.append({
            "title": "Strong Credit Profile",
            "message": f"Your score of {cs} meets most lender requirements.",
            "actions": [
                "Aim for 750+ to unlock better rates.",
                "Check your credit report for minor issues.",
                "Avoid new credit for 3 months before applying."
            ]
        })
    elif cs >= 600:
        recs.append({
            "title": "Credit Improvement Opportunity",
            "message": f"Your score of {cs} is acceptable but could be improved.",
            "actions": [
                "Reduce credit utilisation below 30%.",
                "Register on the electoral roll.",
                "Use a credit builder card for 6 months."
            ]
        })
    else:
        recs.append({
            "title": "Credit Building Needed",
            "message": f"Your score of {cs} needs improvement before applying.",
            "actions": [
                "Obtain full credit reports from UK agencies.",
                "Dispute incorrect entries immediately.",
                "Build 12 months of clean credit history."
            ]
        })

    # AFFORDABILITY
    ratio = features['monthly_payment_ratio']
    monthly_payment = applicant['loan_amount'] / (applicant['loan_term'] * 12)

    if ratio <= 30:
        recs.append({
            "title": "Strong Payment Capacity",
            "message": f"Your monthly payment of £{monthly_payment:.0f} is only {ratio:.1f}% of income.",
            "actions": [
                "Consider shorter loan terms.",
                "You may qualify for better rates.",
                "You could borrow slightly more if needed."
            ]
        })
    elif ratio <= 40:
        recs.append({
            "title": "Manageable Payment Load",
            "message": f"Your payment ratio of {ratio:.1f}% is acceptable.",
            "actions": [
                "Maintain 3–6 months of emergency savings.",
                "Consider extending loan term slightly.",
                "Review monthly expenses for optimisation."
            ]
        })
    else:
        recs.append({
            "title": "High Payment Burden",
            "message": f"Your payment ratio of {ratio:.1f}% may strain your budget.",
            "actions": [
                "Reduce loan amount by 10–15%.",
                "Extend loan term to reduce monthly cost.",
                "Consider joint application to increase income."
            ]
        })

    # ASSET COVERAGE
    ac = features['asset_coverage']
    if ac >= 200:
        recs.append({
            "title": "Exceptional Asset Security",
            "message": f"Your assets cover {ac:.1f}% of the loan.",
            "actions": [
                "You may qualify for lower interest rates.",
                "Document assets clearly for lenders.",
                "Consider secured loan options."
            ]
        })
    elif ac >= 125:
        recs.append({
            "title": "Adequate Asset Coverage",
            "message": f"Your asset coverage of {ac:.1f}% meets standard requirements.",
            "actions": [
                "Include pension statements if applicable.",
                "Provide recent property valuations.",
                "Consolidate smaller assets for clarity."
            ]
        })
    else:
        recs.append({
            "title": "Asset Coverage Could Improve",
            "message": f"Your asset coverage of {ac:.1f}% is below ideal.",
            "actions": [
                "Increase savings for 6 months.",
                "Consider a guarantor.",
                "Build an emergency fund."
            ]
        })

    # OVERALL SCORE
    if score >= 85:
        recs.append({
            "title": "Premium Application Candidate",
            "message": f"Your matching score of {score}/100 places you in the top tier.",
            "actions": [
                "Apply to multiple lenders to compare offers.",
                "Expect fast approval turnaround.",
                "Negotiate for favourable terms."
            ]
        })
    elif score >= 70:
        recs.append({
            "title": "Strong Application Position",
            "message": f"Your score of {score}/100 indicates high approval likelihood.",
            "actions": [
                "Prepare full documentation.",
                "Apply to 2–3 preferred lenders.",
                "Expect 3–5 working days processing."
            ]
        })
    elif score >= 55:
        recs.append({
            "title": "Borderline Application",
            "message": f"Your score of {score}/100 requires careful preparation.",
            "actions": [
                "Provide detailed explanations for credit issues.",
                "Include employer income confirmation.",
                "Consider a co-signer."
            ]
        })
    else:
        recs.append({
            "title": "Profile Needs Strengthening",
            "message": f"Your score of {score}/100 suggests waiting before applying.",
            "actions": [
                "Improve credit score for 6–12 months.",
                "Increase savings by £5,000–£10,000.",
                "Consult a financial advisor."
            ]
        })

    return recs
# =========================================================
# MODULE 3 — CHARTS + SHAP + STRICT MODEL PREDICTION
# =========================================================

# ---------------------------------------------------------
# PLOTLY → PNG CONVERSION (for PDF thumbnails)
# ---------------------------------------------------------
def fig_to_png_bytes(fig, width=600, height=400, scale=2):
    """Convert a Plotly figure to PNG bytes using Kaleido."""
    try:
        return pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    except Exception as e:
        logger.error(f"PNG conversion failed: {e}")
        st.error("Chart rendering failed. Please try again.")
        st.stop()


# ---------------------------------------------------------
# GAUGE CHART — Matching Score
# ---------------------------------------------------------
def create_score_gauge(score):
    colors = ['#EF4444', '#F59E0B', '#10B981', '#047857']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Profile Match Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#3B82F6"},
            'steps': [
                {'range': [0, 50], 'color': colors[0]},
                {'range': [50, 70], 'color': colors[1]},
                {'range': [70, 85], 'color': colors[2]},
                {'range': [85, 100], 'color': colors[3]}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'value': 70
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ---------------------------------------------------------
# RADAR CHART — Financial Health Metrics
# ---------------------------------------------------------
def create_feature_radar(features):
    categories = ['Credit Score', 'Affordability', 'Asset Security', 'Stability']

    values = [
        min(100, features['credit_score'] / 9.99),
        100 - min(100, features['monthly_payment_ratio']),
        min(100, features['asset_coverage'] / 2.5),
        min(100, features['stability_score'] / 45 * 100)
    ]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line_color='rgb(59, 130, 246)',
        line_width=2
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=40, r=40, t=30, b=30)
    )

    return fig


# ---------------------------------------------------------
# SHAP BAR CHART — Top Decision Factors
# ---------------------------------------------------------
def create_shap_chart(shap_data):
    if shap_data is None or shap_data.empty:
        return None

    plot_data = shap_data.head(8).sort_values('Impact', ascending=True)

    fig = go.Figure(go.Bar(
        x=plot_data['Impact'],
        y=plot_data['Feature'].str.replace('_', ' ').str.title(),
        orientation='h',
        marker_color='#3B82F6',
        text=plot_data['Impact'].round(4),
        textposition='outside'
    ))

    fig.update_layout(
        title='Top Decision Factors',
        xaxis_title='Impact on Decision',
        yaxis_title='Feature',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=100, r=20, t=40, b=20)
    )

    return fig


# ---------------------------------------------------------
# SHAP EXPLANATION
# ---------------------------------------------------------
def generate_shap_explanation(model, X_input, feature_names):
    """Generate SHAP explanation for the model decision."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_importance = np.abs(shap_values).mean(0)

        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': feature_importance[0] if len(feature_importance.shape) > 1 else feature_importance
        }).sort_values('Impact', ascending=False)

        return feature_df

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return None


# ---------------------------------------------------------
# STRICT MODEL-ONLY PREDICTION
# ---------------------------------------------------------
def get_model_prediction(model, feature_columns, df_input, applicant_data):
    """Run prediction using the trained model only (no fallback)."""
    try:
        X_input = df_input.drop(columns=['total_assets'])
        X_input = pd.get_dummies(X_input)
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)

        approval_probability = model.predict_proba(X_input)[0, 1] * 100
        prediction = model.predict(X_input)[0]

        shap_data = generate_shap_explanation(model, X_input, feature_columns)

        logger.info(f"Prediction successful: {approval_probability:.1f}%")
        return approval_probability, prediction, shap_data

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error("Prediction failed. Please check your inputs or model files.")
        st.stop()
# =========================================================
# MODULE 4 — BANK-STYLE PDF REPORT GENERATOR
# =========================================================

class BankPDF(FPDF):
    """Custom PDF class with placeholder bank-style logo."""
    def header(self):
        # Placeholder geometric logo (blue square)
        self.set_fill_color(59, 130, 246)  # Tailwind blue-500
        self.rect(10, 8, 10, 10, 'F')

        # Title next to logo
        self.set_xy(25, 8)
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Smart Loan Advisor', ln=True)

        # Subtitle
        self.set_xy(25, 16)
        self.set_font('Arial', '', 10)
        self.cell(0, 6, 'UK Loan Pre-Approval Report', ln=True)

        # Divider line
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.5)
        self.line(10, 26, 200, 26)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align='C')


def create_pdf_report(applicant_data, results, features, shap_data,
                      gauge_fig, radar_fig, shap_fig):
    """Generate a professional bank-style PDF report with embedded charts."""

    pdf = BankPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---------------------------------------------------------
    # SECTION 1 — Applicant Profile
    # ---------------------------------------------------------
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, '1. Applicant Financial Profile', ln=True)
    pdf.set_font('Arial', '', 10)

    profile_fields = [
        ('Credit Score', applicant_data['credit_score']),
        ('Annual Income', f"£{applicant_data['income_annum']:,}"),
        ('Loan Amount Requested', f"£{applicant_data['loan_amount']:,}"),
        ('Loan Term', f"{applicant_data['loan_term']} years"),
        ('Employment Type', applicant_data['self_employed']),
        ('Education Level', applicant_data['education']),
        ('Dependents', applicant_data['no_of_dependents']),
        ('Total Assets', f"£{applicant_data['total_assets']:,}")
    ]

    for label, value in profile_fields:
        pdf.cell(60, 7, f"{label}:", 0, 0)
        pdf.cell(0, 7, str(value), 0, 1)

    pdf.ln(4)

    # ---------------------------------------------------------
    # SECTION 2 — Assessment Summary
    # ---------------------------------------------------------
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, '2. Assessment Summary', ln=True)
    pdf.set_font('Arial', '', 10)

    summary_fields = [
        ('Approval Probability', f"{results['approval_probability']:.1f}%"),
        ('Matching Score', f"{results['matching_score']}/100"),
        ('Credit Category', features['credit_category']),
        ('Monthly Payment Ratio', f"{features['monthly_payment_ratio']:.1f}%"),
        ('Asset Coverage', f"{features['asset_coverage']:.1f}%"),
        ('Stability Score', f"{features['stability_score']}/45")
    ]

    for label, value in summary_fields:
        pdf.cell(60, 7, f"{label}:", 0, 0)
        pdf.cell(0, 7, str(value), 0, 1)

    pdf.ln(4)

    # ---------------------------------------------------------
    # SECTION 3 — Visual Summary (Thumbnails)
    # ---------------------------------------------------------
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, '3. Visual Summary', ln=True)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 6, "Key charts summarising your financial profile and the model's decision factors.")
    pdf.ln(2)

    # Convert figures to PNG bytes
    gauge_png = fig_to_png_bytes(gauge_fig, width=500, height=300, scale=2)
    radar_png = fig_to_png_bytes(radar_fig, width=500, height=300, scale=2)
    shap_png = fig_to_png_bytes(shap_fig, width=500, height=300, scale=2)

    # Thumbnail placement
    thumb_w = 80
    x_left = 15
    x_right = 110

    # Row 1: Gauge + Radar
    y_start = pdf.get_y()
    pdf.image(io.BytesIO(gauge_png), x=x_left, y=y_start, w=thumb_w)
    pdf.image(io.BytesIO(radar_png), x=x_right, y=y_start, w=thumb_w)
    pdf.ln(60)

    # Row 2: SHAP
    pdf.ln(5)
    pdf.image(io.BytesIO(shap_png), x=x_left, y=pdf.get_y(), w=thumb_w)
    pdf.ln(60)

    pdf.ln(5)

    # ---------------------------------------------------------
    # SECTION 4 — Top Decision Factors
    # ---------------------------------------------------------
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, '4. Top Decision Factors', ln=True)
    pdf.set_font('Arial', '', 10)

    if shap_data is not None and not shap_data.empty:
        for idx, row in shap_data.head(5).iterrows():
            factor = row['Feature'].replace('_', ' ').title()
            impact = row['Impact']
            pdf.multi_cell(0, 6, f"- {factor}: impact score {impact:.4f}")
    else:
        pdf.multi_cell(0, 6, "Decision factor details unavailable.")

    pdf.ln(4)

    # ---------------------------------------------------------
    # SECTION 5 — Personalised Recommendations
    # ---------------------------------------------------------
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, '5. Personalised Recommendations', ln=True)
    pdf.set_font('Arial', '', 10)

    for i, rec in enumerate(results['recommendations'][:4], 1):
        pdf.multi_cell(0, 6, f"{i}. {rec['title']}: {rec['message']}")
        if rec.get('actions'):
            for action in rec['actions'][:3]:
                pdf.multi_cell(0, 6, f"   • {action}")
        pdf.ln(2)

    pdf.ln(4)

    # ---------------------------------------------------------
    # FOOTER DISCLAIMER
    # ---------------------------------------------------------
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0,
        4,
        "Disclaimer: This is a preliminary, model-based assessment. Final loan approval is subject to full underwriting, "
        "documentation, and the lender's credit policies. This report is for informational purposes only and does not "
        "constitute financial advice."
    )

    return pdf.output()
# =========================================================
# MODULE 5 — MAIN STREAMLIT UI
# =========================================================

def main():

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------
    st.markdown('<h1 class="main-header">Smart Loan Advisor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-box">
        <strong>Transparent Loan Assessment:</strong> Receive a data-driven, model-based evaluation of your loan eligibility.
        This tool uses a trained machine learning model and UK lending criteria to generate a personalised financial profile.
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SIDEBAR — USER INPUTS
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown('<h3 class="sub-header">Your Information</h3>', unsafe_allow_html=True)

        credit_score = st.slider("Credit Score (0–999)", 0, 999, 750)
        income_annum = st.number_input("Annual Income (£)", min_value=15000, value=35000, step=1000)
        loan_amount = st.number_input("Loan Amount (£)", min_value=5000, value=25000, step=1000)
        loan_term = st.slider("Loan Term (Years)", 1, 30, 5)

        no_of_dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, "5 or more"])
        if no_of_dependents == "5 or more":
            no_of_dependents = 5

        col1, col2 = st.columns(2)
        with col1:
            self_employed = st.radio("Employment Type", ["Employed", "Self-Employed"])
        with col2:
            education = st.radio("Education Level", ["Graduate", "Not Graduate"])

        total_assets = st.number_input("Total Assets (£)", min_value=0, value=50000, step=5000)

        calculate_btn = st.button("Check My Eligibility", type="primary", use_container_width=True)

    # ---------------------------------------------------------
    # PROCESS INPUTS
    # ---------------------------------------------------------
    if calculate_btn:

        # Build applicant data dict
        applicant_data = {
            "credit_score": credit_score,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "no_of_dependents": no_of_dependents,
            "self_employed": self_employed,
            "education": education,
            "total_assets": total_assets
        }

        # Convert to DataFrame
        df_input = pd.DataFrame([applicant_data])

        # Feature engineering
        features_df = create_credit_features(df_input)
        features_row = features_df.iloc[0]

        # Matching score
        matching_score = calculate_matching_score(features_row)

        # Model prediction
        approval_probability, prediction, shap_data = get_model_prediction(
            model, feature_columns, features_df, applicant_data
        )

        # Recommendations
        recommendations = generate_recommendations(matching_score, features_row, applicant_data)

        # Bundle results
        results = {
            "approval_probability": approval_probability,
            "matching_score": matching_score,
            "status": "Likely Approved" if prediction == 1 else "Likely Declined",
            "recommendations": recommendations
        }

        # ---------------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------------
        st.markdown('<h3 class="sub-header">Your Assessment Results</h3>', unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        colA.metric("Approval Probability", f"{approval_probability:.1f}%")
        colB.metric("Matching Score", f"{matching_score}/100")
        colC.metric("Status", results["status"])

        # ---------------------------------------------------------
        # CHARTS
        # ---------------------------------------------------------
        st.markdown('<h3 class="sub-header">Visual Summary</h3>', unsafe_allow_html=True)

        gauge_fig = create_score_gauge(matching_score)
        radar_fig = create_feature_radar(features_row)
        shap_fig = create_shap_chart(shap_data)

        col1, col2 = st.columns(2)
        col1.plotly_chart(gauge_fig, use_container_width=True)
        col2.plotly_chart(radar_fig, use_container_width=True)

        if shap_fig:
            st.plotly_chart(shap_fig, use_container_width=True)

        # ---------------------------------------------------------
        # PDF REPORT GENERATION
        # ---------------------------------------------------------
        pdf_bytes = create_pdf_report(
            applicant_data=applicant_data,
            results=results,
            features=features_row,
            shap_data=shap_data,
            gauge_fig=gauge_fig,
            radar_fig=radar_fig,
            shap_fig=shap_fig
        )

        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_bytes,
            file_name="loan_preapproval_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# Run the app
if __name__ == "__main__":
    main()
