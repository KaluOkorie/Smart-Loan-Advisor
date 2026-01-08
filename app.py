# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import io
import time
import shap
from docx import Document
from docx.shared import Inches
from io import BytesIO

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Advisor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (theme-aware)
st.markdown(
    """
    <style>
        .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 1rem; color:var(--primary-text-color); }
        .sub-header { font-size: 1.25rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; color:var(--secondary-text-color); }
        .tip-box { background-color: var(--background-secondary); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--primary-color); margin: 1rem 0; }
        .success-box { background-color: rgba(16,185,129,0.08); padding: 1rem; border-radius: 8px; border-left: 4px solid #10B981; margin: 0.5rem 0; border: 1px solid rgba(16,185,129,0.12); }
        .warning-box { background-color: rgba(245,158,11,0.06); padding: 1rem; border-radius: 8px; border-left: 4px solid #F59E0B; margin: 0.5rem 0; border: 1px solid rgba(245,158,11,0.12); }
        .info-box { background-color: rgba(14,165,233,0.06); padding: 1rem; border-radius: 8px; border-left: 4px solid #0EA5E9; margin: 0.5rem 0; border: 1px solid rgba(14,165,233,0.12); }
        .stMetric { min-height: 100px; }
        .stMetric > div { min-height: 85px; }
        .stMetric label { font-size: 0.95rem !important; font-weight: 600 !important; }
        :root {
            --primary-color: #3B82F6;
            --secondary-color: #10B981;
            --background-primary: #FFFFFF;
            --background-secondary: #F3F4F6;
            --primary-text-color: #111827;
            --secondary-text-color: #374151;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --primary-color: #60A5FA;
                --secondary-color: #34D399;
                --background-primary: #1F2937;
                --background-secondary: #374151;
                --primary-text-color: #F9FAFB;
                --secondary-text-color: #D1D5DB;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# INITIALIZE SESSION STATE
# ---------------------------------------------------------
if "last_submission" not in st.session_state:
    st.session_state.last_submission = 0
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------
# UK FINANCIAL STANDARDS
# ---------------------------------------------------------
CONVERSION_RATE = 100  # scale factors used by model input
UK_AVERAGE_SALARY = 35000
UK_AVERAGE_HOUSE_PRICE = 250000
UK_ASSET_MULTIPLIER = 5

# ---------------------------------------------------------
# FEATURE ENGINEERING FUNCTIONS
# ---------------------------------------------------------
def create_credit_features(df):
    df_engineered = df.copy()
    monthly_income = df_engineered["income_annum"] / 12
    monthly_loan_payment = (df_engineered["loan_amount"] / df_engineered["loan_term"]) / 12
    df_engineered["monthly_payment_ratio"] = (monthly_loan_payment / monthly_income) * 100
    df_engineered["asset_coverage"] = (df_engineered["total_assets"] / df_engineered["loan_amount"]) * 100
    df_engineered["loan_to_income"] = (df_engineered["loan_amount"] / df_engineered["income_annum"]) * 100

    def categorize_credit_score(score):
        if score >= 900:
            return "Excellent"
        elif score >= 800:
            return "Very Good"
        elif score >= 700:
            return "Good"
        elif score >= 600:
            return "Fair"
        else:
            return "Needs Improvement"

    df_engineered["credit_category"] = df_engineered["credit_score"].apply(categorize_credit_score)
    df_engineered["stability_score"] = 0
    df_engineered.loc[df_engineered["self_employed"] == "No", "stability_score"] += 30
    df_engineered.loc[df_engineered["education"] == "Graduate", "stability_score"] += 20
    df_engineered.loc[df_engineered["no_of_dependents"] <= 2, "stability_score"] += 10

    return df_engineered

def generate_detailed_recommendations(score, features, applicant_data):
    # unchanged logic, uses score and features to produce recommendations
    recommendations = []
    credit_score = features.get("credit_score", 0)
    if credit_score >= 800:
        recommendations.append({
            "title": "Excellent Credit Standing",
            "message": f"Credit score {credit_score} is excellent.",
            "actions": ["You qualify for highly competitive rates", "Maintain credit utilization below 25%"],
            "box_type": "success"
        })
    elif credit_score >= 700:
        recommendations.append({
            "title": "Good Credit Profile",
            "message": f"Score {credit_score} meets most lender requirements.",
            "actions": ["Aim for 750+ to reduce rates", "Avoid new credit applications before applying"],
            "box_type": "success"
        })
    elif credit_score >= 600:
        recommendations.append({
            "title": "Credit Improvement Opportunity",
            "message": f"Score {credit_score} is acceptable but can improve.",
            "actions": ["Register on electoral roll", "Reduce card balances below 30%"],
            "box_type": "warning"
        })
    else:
        recommendations.append({
            "title": "Profile Needs Strengthening",
            "message": f"Score {credit_score} needs attention prior to application.",
            "actions": ["Obtain full credit report", "Build 12 months clean history"],
            "box_type": "warning"
        })

    payment_ratio = features.get("monthly_payment_ratio", 0)
    monthly_payment = applicant_data["loan_amount"] / (applicant_data["loan_term"] * 12)
    if payment_ratio <= 30:
        recommendations.append({
            "title": "Strong Payment Capacity",
            "message": f"Monthly payment £{monthly_payment:.0f} is {payment_ratio:.1f}% of income.",
            "actions": ["Consider shorter terms to reduce interest"],
            "box_type": "success"
        })
    elif payment_ratio <= 40:
        recommendations.append({
            "title": "Manageable Payment Load",
            "message": f"Payment is {payment_ratio:.1f}% of income.",
            "actions": ["Maintain emergency savings 3-6 months"],
            "box_type": "info"
        })
    else:
        recommendations.append({
            "title": "High Payment Burden",
            "message": f"Payment burden {payment_ratio:.1f}% may strain budget.",
            "actions": ["Reduce loan amount or extend term", "Consider joint application"],
            "box_type": "warning"
        })

    asset_coverage = features.get("asset_coverage", 0)
    if asset_coverage >= 200:
        recommendations.append({
            "title": "Exceptional Asset Security",
            "message": f"Assets cover {asset_coverage:.1f}% of the loan.",
            "actions": ["You may secure lower interest rates", "Document asset valuations"],
            "box_type": "success"
        })
    elif asset_coverage >= 125:
        recommendations.append({
            "title": "Adequate Asset Coverage",
            "message": f"Assets cover {asset_coverage:.1f}%.",
            "actions": ["Include pension statements if available"],
            "box_type": "info"
        })
    else:
        recommendations.append({
            "title": "Asset Coverage Could Improve",
            "message": f"Asset coverage {asset_coverage:.1f}% is below ideal.",
            "actions": ["Increase liquid savings", "Consider guarantor options"],
            "box_type": "warning"
        })

    # Score-driven overall recommendation
    if score >= 85:
        recommendations.append({
            "title": "Premium Candidate",
            "message": f"Matching score {score}/100 – strong approval likelihood.",
            "actions": ["Apply to multiple lenders to compare offers"],
            "box_type": "success"
        })
    elif score >= 70:
        recommendations.append({
            "title": "Strong Candidate",
            "message": f"Matching score {score}/100 – high approval likelihood.",
            "actions": ["Complete full application with documentation"],
            "box_type": "success"
        })
    elif score >= 55:
        recommendations.append({
            "title": "Borderline Candidate",
            "message": f"Score {score}/100 – prepare additional documentation.",
            "actions": ["Include employer letters, consider co-signer"],
            "box_type": "warning"
        })
    else:
        recommendations.append({
            "title": "Needs Improvement",
            "message": f"Score {score}/100 – consider strengthening profile before applying.",
            "actions": ["Improve credit, increase savings"],
            "box_type": "warning"
        })

    return recommendations

def generate_shap_explanation(model, X_input, feature_names):
    try:
        # Use SHAP's generic Explainer for compatibility
        explainer = shap.Explainer(model, feature_names=feature_names)
        shap_values = explainer(X_input)
        # shap_values.values shape: (n_samples, n_features) for regression or multiclass, handle appropriately
        values = shap_values.values
        # For binary classification shap.Explainer may return shape (n_samples, n_features) for the single output
        impact = np.abs(values).mean(axis=0)
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": impact
        }).sort_values("Impact", ascending=False).head(10)
        return feature_df
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")
        return None

# ---------------------------------------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def create_score_gauge(score):
    colors = ["#EF4444", "#F59E0B", "#10B981", "#047857"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Profile Match Score", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkgray"},
            "bar": {"color": "#3B82F6"},
            "steps": [
                {"range": [0, 50], "color": colors[0]},
                {"range": [50, 70], "color": colors[1]},
                {"range": [70, 85], "color": colors[2]},
                {"range": [85, 100], "color": colors[3]}
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 70}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_feature_radar(features):
    categories = ["Credit Score", "Affordability", "Asset Security", "Stability"]
    values = [
        min(100, features["credit_score"] / 9.99),
        100 - min(100, features["monthly_payment_ratio"]),
        min(100, features["asset_coverage"] / 2.5),
        min(100, features["stability_score"] / 45 * 100)
    ]
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(59,130,246,0.25)",
        line_color="rgb(59,130,246)",
        line_width=2
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=30, r=30, t=30, b=20))
    return fig

def create_shap_chart(shap_data):
    if shap_data is None or shap_data.empty:
        return None
    plot_data = shap_data.head(8).sort_values("Impact", ascending=True)
    fig = go.Figure(go.Bar(
        x=plot_data["Impact"],
        y=plot_data["Feature"].str.replace("_", " ").str.title(),
        orientation="h",
        text=plot_data["Impact"].round(4),
        textposition="outside"
    ))
    fig.update_layout(title="Top Decision Factors", xaxis_title="Impact on Decision", paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=120, r=20, t=40, b=20))
    return fig

# ---------------------------------------------------------
# WORD REPORT GENERATION
# ---------------------------------------------------------
def create_word_report(applicant_data, results, features, gauge_fig, radar_fig, shap_fig=None):
    doc = Document()
    doc.add_heading("UK Loan Pre-Approval Report", level=1)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}")
    doc.add_paragraph("")  # spacing

    # Applicant info
    doc.add_heading("Applicant Information", level=2)
    info_table = doc.add_table(rows=0, cols=2)
    for k, v in applicant_data.items():
        row = info_table.add_row().cells
        row[0].text = str(k).replace("_", " ").title()
        row[1].text = str(v)

    # Assessment results
    doc.add_heading("Assessment Results", level=2)
    doc.add_paragraph(f"Matching Score: {results['matching_score']}/100")
    doc.add_paragraph(f"Approval Probability: {results['approval_probability']:.1f}%")
    doc.add_paragraph(f"Status: {results['status']}")

    # Financial metrics
    doc.add_heading("Financial Health Metrics", level=2)
    metrics_table = doc.add_table(rows=0, cols=2)
    metrics_table.add_row().cells[0].text = "Monthly Payment Ratio"
    metrics_table.add_row().cells[1].text = f"{features['monthly_payment_ratio']:.1f}%"
    metrics_table.add_row().cells[0].text = "Asset Coverage"
    metrics_table.add_row().cells[1].text = f"{features['asset_coverage']:.1f}%"

    # Insert charts as images (use Plotly + kaleido -> bytes)
    try:
        # Gauge
        gauge_png = gauge_fig.to_image(format="png", engine="kaleido")
        doc.add_heading("Matching Score (Gauge)", level=2)
        doc.add_picture(BytesIO(gauge_png), width=Inches(6))

        # Radar
        radar_png = radar_fig.to_image(format="png", engine="kaleido")
        doc.add_heading("Financial Health Radar", level=2)
        doc.add_picture(BytesIO(radar_png), width=Inches(6))

        # SHAP
        if shap_fig is not None:
            shap_png = shap_fig.to_image(format="png", engine="kaleido")
            doc.add_heading("SHAP: Top Decision Factors", level=2)
            doc.add_picture(BytesIO(shap_png), width=Inches(6))
    except Exception as e:
        # If chart export fails, add a note (user must have kaleido available in runtime).
        doc.add_paragraph("Note: Chart images could not be embedded due to server configuration: " + str(e))

    # Recommendations
    doc.add_heading("Personalized Recommendations", level=2)
    for rec in results["recommendations"]:
        doc.add_paragraph(rec["title"], style="List Bullet")
        doc.add_paragraph(rec["message"])
        if rec.get("actions"):
            for action in rec["actions"]:
                doc.add_paragraph(action, style="List Number")

    # Footer / disclaimer
    doc.add_paragraph("")
    doc.add_paragraph("Disclaimer: This is a preliminary assessment based on the information provided. Final loan approval is subject to documentation and lender policies.")
    doc.add_paragraph("© 2026 Smart Solution to tough data")

    # Save to bytes
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">UK Smart Loan Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<div class="tip-box"><strong>Transparent Loan Assessment:</strong> This tool provides a preliminary assessment based on UK lending criteria. Final approval requires full documentation and verification by a lender.</div>', unsafe_allow_html=True)

    # Sidebar input
    with st.sidebar:
        st.markdown('<h3 class="sub-header">Your Information</h3>', unsafe_allow_html=True)
        credit_score = st.slider("Credit Score (0-999)", 0, 999, 750)
        income_annum = st.number_input("Annual Income (£)", min_value=0, value=35000, step=1000)
        loan_amount = st.number_input("Loan Amount Needed (£)", min_value=1000, value=25000, step=1000)
        loan_term = st.slider("Loan Term (Years)", 1, 30, 5)
        no_of_dependents = st.selectbox("Number of Dependents", options=[0, 1, 2, 3, 4, 5], index=1)
        self_employed = st.radio("Employment Type", ["Employed", "Self-Employed"])
        education = st.radio("Education Level", ["Graduate", "Not Graduate"])
        total_assets = st.number_input("Total Assets Value (£)", min_value=0, value=50000, step=1000)

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_submission
        if time_since_last < 3 and st.session_state.results is not None:
            st.warning(f"Please wait {3 - time_since_last:.1f} seconds")
            calculate_disabled = True
        else:
            calculate_disabled = False

        calculate_btn = st.button("Check Eligibility", disabled=calculate_disabled, use_container_width=True)

    # Load model before computation and abort if missing (no fallback)
    model = None
    feature_columns = None
    model_load_error = None
    try:
        model = joblib.load("best_xgb_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
    except Exception as e:
        model_load_error = e

    if calculate_btn:
        st.session_state.last_submission = time.time()

        if model_load_error is not None:
            st.error(f"Model load error: {model_load_error}")
            st.error("A trained model file (best_xgb_model.pkl and feature_columns.pkl) is required. Place these files in the app directory and retry.")
            st.stop()

        # Prepare model input (scale consistent with training)
        model_input_data = {
            "credit_score": credit_score,
            "income_annum": int(income_annum * CONVERSION_RATE),
            "loan_amount": int(loan_amount * CONVERSION_RATE),
            "loan_term": loan_term,
            "no_of_dependents": int(no_of_dependents),
            "self_employed": "Yes" if self_employed == "Self-Employed" else "No",
            "education": education,
            "total_assets": int(total_assets * CONVERSION_RATE)
        }
        df_input = pd.DataFrame([model_input_data])
        df_features = create_credit_features(df_input)

        # Prepare features for model (one-hot + align)
        X_input = df_input.drop(columns=["total_assets"])
        X_input = pd.get_dummies(X_input)
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)

        # Model prediction (probability -> approval_probability)
        try:
            approval_probability = model.predict_proba(X_input)[0, 1] * 100
            prediction = int(model.predict(X_input)[0])
            matching_score = int(round(approval_probability))
        except Exception as e:
            st.error(f"Model prediction error: {e}")
            st.stop()

        shap_data = generate_shap_explanation(model, X_input, X_input.columns.tolist())

        # Generate recommendations using matching_score (model-driven)
        recommendations = generate_detailed_recommendations(matching_score, df_features.iloc[0].to_dict(), {
            "credit_score": credit_score,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "no_of_dependents": int(no_of_dependents),
            "self_employed": self_employed,
            "education": education,
            "total_assets": total_assets
        })

        # Determine status
        if matching_score >= 80:
            status = "Excellent Match"
            status_box = "success"
        elif matching_score >= 65:
            status = "Good Match"
            status_box = "success"
        elif matching_score >= 50:
            status = "Needs Review"
            status_box = "warning"
        else:
            status = "Needs Improvement"
            status_box = "warning"

        # Store results
        st.session_state.results = {
            "matching_score": matching_score,
            "approval_probability": approval_probability,
            "prediction": prediction,
            "status": status,
            "status_box": status_box,
            "recommendations": recommendations,
            "applicant_data": {
                "credit_score": credit_score,
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "no_of_dependents": int(no_of_dependents),
                "self_employed": self_employed,
                "education": education,
                "total_assets": total_assets
            },
            "features": df_features.iloc[0].to_dict(),
            "shap_data": shap_data
        }

    # Display results area
    if st.session_state.results:
        results = st.session_state.results
        st.markdown('<h2 class="sub-header">Eligibility Report</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Profile Match", value=f"{results['matching_score']}/100", delta=results["status"])
        with col2:
            st.metric(label="Approval Chance", value=f"{results['approval_probability']:.1f}%")
        with col3:
            st.metric(label="Next Action", value="Apply Now" if results["matching_score"] >= 65 else "Improve First")
        with col4:
            processing_time = "2-4 Days" if results["matching_score"] >= 70 else ("5-10 Days" if results["matching_score"] >= 50 else "Manual Review")
            st.metric(label="Processing Time", value=processing_time)

        # Charts
        col1, col2 = st.columns(2)
        gauge_fig = create_score_gauge(results["matching_score"])
        radar_fig = create_feature_radar(results["features"])
        with col1:
            st.plotly_chart(gauge_fig, use_container_width=True)
        with col2:
            st.plotly_chart(radar_fig, use_container_width=True)

        # SHAP chart
        shap_fig = None
        if results.get("shap_data") is not None:
            shap_fig = create_shap_chart(results["shap_data"])
            if shap_fig is not None:
                st.plotly_chart(shap_fig, use_container_width=True)

        # Recommendations
        st.markdown('<h3 class="sub-header">Action Plan</h3>', unsafe_allow_html=True)
        for i, rec in enumerate(results["recommendations"][:6]):
            box_style = "success-box" if rec["box_type"] == "success" else ("warning-box" if rec["box_type"] == "warning" else "info-box")
            with st.expander(rec["title"], expanded=(i < 2)):
                st.markdown(f"<div class='{box_style}'>", unsafe_allow_html=True)
                st.write(rec["message"])
                if rec.get("actions"):
                    for action in rec["actions"]:
                        st.write(f"• {action}")
                st.markdown("</div>", unsafe_allow_html=True)

        # Word report generation + download
        st.markdown('<h3 class="sub-header">Download Full Report</h3>', unsafe_allow_html=True)
        try:
            word_bytes = create_word_report(results["applicant_data"], results, results["features"], gauge_fig, radar_fig, shap_fig)
            st.download_button(
                label="Download Word Report",
                data=word_bytes,
                file_name=f"Loan_Assessment_{datetime.now().strftime('%Y%m%d')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except Exception as e:
            st.error(f"Could not generate Word report: {e}")
    else:
        # Welcome/instructions screen
        st.markdown(
            """
            <h3 class="sub-header">How It Works</h3>
            <div>
                <ol>
                    <li>Enter your financial profile in the sidebar.</li>
                    <li>Click "Check Eligibility" to evaluate using the trained ML model.</li>
                    <li>Download a Word report with charts and recommendations.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div style="text-align:center; color:var(--secondary-text-color); font-size:0.85rem;">Important Disclaimer: This tool provides a preliminary assessment. Final approval depends on documentation and lender policies. © 2026 Smart Solution to tough data</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
