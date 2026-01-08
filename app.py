import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import io
import time
import shap
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.3rem;
    }
    
    .metric-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .success-indicator {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border-left: 4px solid #10b981;
    }
    
    .warning-indicator {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border-left: 4px solid #f59e0b;
    }
    
    .critical-indicator {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border-left: 4px solid #ef4444;
    }
    
    .info-box {
        background-color: #eff6ff;
        border: 1px solid #dbeafe;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stButton button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
    }
    
    .stButton button:hover {
        background-color: #1d4ed8;
    }
    
    .data-table {
        font-size: 0.9rem;
    }
    
    .feature-importance {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INITIALIZE SESSION STATE
# ---------------------------------------------------------
if 'last_submission' not in st.session_state:
    st.session_state.last_submission = 0
if 'results' not in st.session_state:
    st.session_state.results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None

# ---------------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    """Load the trained XGBoost model and feature columns"""
    try:
        model = joblib.load("best_xgb_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        st.session_state.model_loaded = True
        st.session_state.model = model
        st.session_state.feature_columns = feature_columns
        return model, feature_columns
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load model at startup
model, feature_columns = load_ml_model()

# ---------------------------------------------------------
# FINANCIAL CONFIGURATION
# ---------------------------------------------------------
CONVERSION_RATE = 100
UK_AVERAGE_SALARY = 35000
MAX_LOAN_TERM = 30
MIN_CREDIT_SCORE = 0
MAX_CREDIT_SCORE = 999

# ---------------------------------------------------------
# FEATURE ENGINEERING FUNCTIONS
# ---------------------------------------------------------
def create_credit_features(df):
    """Create engineered features for credit assessment"""
    df_engineered = df.copy()
    
    # Monthly calculations
    monthly_income = df_engineered['income_annum'] / 12
    monthly_loan_payment = (df_engineered['loan_amount'] / df_engineered['loan_term']) / 12
    
    # Key ratios
    df_engineered['monthly_payment_ratio'] = (monthly_loan_payment / monthly_income) * 100
    df_engineered['asset_coverage'] = (df_engineered['total_assets'] / df_engineered['loan_amount']) * 100
    df_engineered['loan_to_income'] = (df_engineered['loan_amount'] / df_engineered['income_annum']) * 100
    
    # Credit score categories
    def categorize_credit_score(score):
        if score >= 900:
            return 'Excellent'
        elif score >= 800:
            return 'Very Good'
        elif score >= 700:
            return 'Good'
        elif score >= 600:
            return 'Fair'
        else:
            return 'Needs Improvement'
    
    df_engineered['credit_category'] = df_engineered['credit_score'].apply(categorize_credit_score)
    
    # Stability score
    df_engineered['stability_score'] = 0
    df_engineered.loc[df_engineered['self_employed'] == 'No', 'stability_score'] += 30
    df_engineered.loc[df_engineered['education'] == 'Graduate', 'stability_score'] += 20
    df_engineered.loc[df_engineered['no_of_dependents'] <= 2, 'stability_score'] += 10
    
    return df_engineered

def calculate_model_probability(model_input):
    """Calculate approval probability using the ML model"""
    try:
        # Prepare features for model
        X_input = model_input.drop(columns=['total_assets'])
        X_input = pd.get_dummies(X_input)
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)
        
        # Get prediction probability
        approval_probability = model.predict_proba(X_input)[0, 1] * 100
        prediction = model.predict(X_input)[0]
        
        return approval_probability, prediction, X_input
        
    except Exception as e:
        raise Exception(f"Model prediction error: {str(e)}")

def generate_shap_explanation(X_input):
    """Generate SHAP-based explanation for the decision"""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        
        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class
        
        # Get feature importance
        if len(shap_values.shape) > 1:
            feature_importance = np.abs(shap_values).mean(0)
            importance_values = feature_importance[0]
        else:
            importance_values = np.abs(shap_values)
        
        feature_df = pd.DataFrame({
            'Feature': feature_columns,
            'Impact': importance_values
        }).sort_values('Impact', ascending=False).head(10)
        
        return feature_df, shap_values
        
    except Exception as e:
        raise Exception(f"SHAP explanation error: {str(e)}")

def generate_credit_recommendations(approval_probability, features, shap_data):
    """Generate credit management recommendations based on model results"""
    recommendations = []
    
    # Credit Score Analysis
    credit_score = features.get('credit_score', 0)
    if credit_score >= 800:
        recommendations.append({
            "category": "Credit Profile",
            "title": "Excellent Credit Standing",
            "message": f"Credit score of {credit_score} is in the top tier for lending.",
            "actions": [
                "Maintain low credit utilization (below 25%)",
                "Continue timely payments on all accounts",
                "Monitor credit report quarterly for accuracy"
            ],
            "priority": "Low"
        })
    elif credit_score >= 700:
        recommendations.append({
            "category": "Credit Profile",
            "title": "Good Credit Profile",
            "message": f"Score of {credit_score} meets standard lending requirements.",
            "actions": [
                "Reduce any credit card balances above 30% of limits",
                "Avoid new credit applications in next 3 months",
                "Consider credit limit increases to improve utilization ratio"
            ],
            "priority": "Medium"
        })
    else:
        recommendations.append({
            "category": "Credit Profile",
            "title": "Credit Improvement Required",
            "message": f"Score of {credit_score} needs attention to improve approval chances.",
            "actions": [
                "Obtain full credit report and dispute inaccuracies",
                "Establish 12 months of consistent payment history",
                "Consider secured credit products to rebuild history"
            ],
            "priority": "High"
        })
    
    # Payment Affordability Analysis
    payment_ratio = features.get('monthly_payment_ratio', 0)
    if payment_ratio <= 30:
        recommendations.append({
            "category": "Affordability",
            "title": "Strong Payment Capacity",
            "message": f"Monthly payment represents {payment_ratio:.1f}% of income.",
            "actions": [
                "Maintain current debt-to-income ratio",
                "Consider accelerated repayment for interest savings",
                "Build emergency fund equal to 6 months of payments"
            ],
            "priority": "Low"
        })
    elif payment_ratio <= 40:
        recommendations.append({
            "category": "Affordability",
            "title": "Manageable Payment Load",
            "message": f"Payment ratio of {payment_ratio:.1f}% requires careful budgeting.",
            "actions": [
                "Create detailed monthly budget including all obligations",
                "Build 3-6 month emergency fund",
                "Consider income protection insurance"
            ],
            "priority": "Medium"
        })
    else:
        recommendations.append({
            "category": "Affordability",
            "title": "High Payment Burden",
            "message": f"At {payment_ratio:.1f}%, payment may strain financial capacity.",
            "actions": [
                "Reduce requested loan amount by 10-15%",
                "Extend loan term to reduce monthly obligation",
                "Increase income or reduce other expenses"
            ],
            "priority": "High"
        })
    
    # SHAP-based recommendations from top features
    if shap_data is not None and not shap_data.empty:
        top_features = shap_data.head(3)
        for _, row in top_features.iterrows():
            feature = row['Feature']
            impact = row['Impact']
            
            if 'credit' in feature.lower():
                recommendations.append({
                    "category": "Model Insight",
                    "title": f"Critical Factor: {feature.replace('_', ' ').title()}",
                    "message": f"This factor had significant impact ({impact:.4f}) on the decision.",
                    "actions": [
                        "Review historical credit behavior patterns",
                        "Address any negative marks on credit report",
                        "Maintain consistent payment history"
                    ],
                    "priority": "High"
                })
    
    return recommendations

# ---------------------------------------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def create_score_gauge(score, title="Risk Assessment Score"):
    """Create a professional gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#2563eb"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ef4444'},  # Red for high risk
                {'range': [40, 70], 'color': '#f59e0b'},  # Orange for medium risk
                {'range': [70, 100], 'color': '#10b981'}  # Green for low risk
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.85,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#1f2937', family="Arial"),
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_radar_chart(features):
    """Create radar chart for financial health metrics"""
    categories = ['Credit Quality', 'Affordability', 'Asset Coverage', 'Stability', 'Income Adequacy']
    
    # Normalize values to 0-100 scale
    values = [
        min(100, features['credit_score'] / 10),  # Credit score out of 1000
        100 - min(100, features['monthly_payment_ratio']),  # Inverse of payment ratio
        min(100, features['asset_coverage'] / 2.5),  # Asset coverage percentage
        min(100, features['stability_score'] / 45 * 100),  # Stability score
        min(100, (features['income_annum'] / 50000) * 100)  # Income adequacy
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.2)',
        line_color='rgb(37, 99, 235)',
        line_width=2
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(128, 128, 128, 0.2)',
                linecolor='gray'
            ),
            angularaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.2)',
                linecolor='gray'
            ),
            bgcolor='white'
        ),
        paper_bgcolor='white',
        font=dict(color='#1f2937'),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=30, b=30)
    )
    
    return fig

def create_shap_bar_chart(shap_data):
    """Create horizontal bar chart for SHAP feature importance"""
    if shap_data is None or shap_data.empty:
        return None
    
    # Take top 8 features
    plot_data = shap_data.head(8).sort_values('Impact', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=plot_data['Impact'],
        y=plot_data['Feature'].str.replace('_', ' ').str.title(),
        orientation='h',
        marker_color='#2563eb',
        text=plot_data['Impact'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Feature Impact Analysis',
        xaxis_title='SHAP Value (Impact Magnitude)',
        yaxis_title='Feature',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#1f2937'),
        height=350,
        margin=dict(l=120, r=20, t=40, b=20)
    )
    
    return fig

# ---------------------------------------------------------
# WORD DOCUMENT REPORT GENERATION
# ---------------------------------------------------------
def create_word_document(applicant_data, results, features, shap_data, charts):
    """Generate comprehensive Word document report"""
    doc = Document()
    
    # Title
    title = doc.add_heading('Credit Risk Assessment Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Date and reference
    doc.add_paragraph(f'Generated: {datetime.now().strftime("%d %B %Y at %H:%M")}')
    doc.add_paragraph(f'Reference: CR-{datetime.now().strftime("%Y%m%d-%H%M")}')
    doc.add_paragraph()
    
    # Executive Summary
    doc.add_heading('Executive Summary', 1)
    summary = doc.add_paragraph()
    summary.add_run('Assessment Outcome: ').bold = True
    summary.add_run(f"{results['status']}\n")
    summary.add_run('Approval Probability: ').bold = True
    summary.add_run(f"{results['approval_probability']:.1f}%\n")
    summary.add_run('Risk Category: ').bold = True
    risk_category = "Low Risk" if results['approval_probability'] >= 70 else "Medium Risk" if results['approval_probability'] >= 50 else "High Risk"
    summary.add_run(f"{risk_category}")
    doc.add_paragraph()
    
    # Applicant Information
    doc.add_heading('Applicant Information', 1)
    table = doc.add_table(rows=7, cols=2)
    table.style = 'Light Grid Accent 1'
    
    data_rows = [
        ("Credit Score", str(applicant_data['credit_score'])),
        ("Annual Income", f"£{applicant_data['income_annum']:,}"),
        ("Loan Amount", f"£{applicant_data['loan_amount']:,}"),
        ("Loan Term", f"{applicant_data['loan_term']} years"),
        ("Employment Type", applicant_data['self_employed']),
        ("Education Level", applicant_data['education']),
        ("Number of Dependents", str(applicant_data['no_of_dependents']))
    ]
    
    for i, (label, value) in enumerate(data_rows):
        table.cell(i, 0).text = label
        table.cell(i, 1).text = value
    
    doc.add_paragraph()
    
    # Model Assessment
    doc.add_heading('Model Assessment Results', 1)
    
    # Add gauge chart
    doc.add_heading('Risk Assessment Score', 2)
    doc.add_paragraph(f"The applicant's risk assessment score is {results['matching_score']}/100.")
    if charts['gauge']:
        # Save chart as image and add to document
        chart_path = "gauge_chart.png"
        charts['gauge'].write_image(chart_path)
        doc.add_picture(chart_path, width=Inches(5))
    
    # Financial Health Metrics
    doc.add_heading('Financial Health Assessment', 2)
    doc.add_paragraph(f"Monthly Payment Ratio: {features['monthly_payment_ratio']:.1f}% of income")
    doc.add_paragraph(f"Asset Coverage: {features['asset_coverage']:.1f}% of loan amount")
    doc.add_paragraph(f"Credit Category: {features['credit_category']}")
    doc.add_paragraph(f"Stability Score: {features['stability_score']}/45")
    
    # Add radar chart
    if charts['radar']:
        chart_path = "radar_chart.png"
        charts['radar'].write_image(chart_path)
        doc.add_picture(chart_path, width=Inches(5))
    
    # Model Decision Factors
    doc.add_heading('Model Decision Factors', 1)
    doc.add_paragraph("The following features had the greatest impact on the model's decision:")
    
    if shap_data is not None and not shap_data.empty:
        table = doc.add_table(rows=min(6, len(shap_data)) + 1, cols=2)
        table.style = 'Light Grid Accent 1'
        
        # Header
        table.cell(0, 0).text = "Feature"
        table.cell(0, 1).text = "Impact Score"
        
        # Data rows
        for i, (_, row) in enumerate(shap_data.head(6).iterrows(), 1):
            table.cell(i, 0).text = row['Feature'].replace('_', ' ').title()
            table.cell(i, 1).text = f"{row['Impact']:.4f}"
    
    # Add SHAP chart
    if charts['shap']:
        chart_path = "shap_chart.png"
        charts['shap'].write_image(chart_path)
        doc.add_picture(chart_path, width=Inches(5))
    
    # Credit Management Recommendations
    doc.add_heading('Credit Management Recommendations', 1)
    
    for rec in results['recommendations']:
        doc.add_heading(rec['title'], 3)
        doc.add_paragraph(rec['message'])
        doc.add_paragraph("Recommended Actions:")
        for action in rec['actions']:
            doc.add_paragraph(f"• {action}", style='List Bullet')
        doc.add_paragraph()
    
    # Risk Management Guidelines
    doc.add_heading('Risk Management Guidelines', 1)
    guidelines = [
        ("Credit Monitoring", "Review credit reports quarterly from all major agencies"),
        ("Debt Management", "Maintain total debt payments below 40% of gross income"),
        ("Emergency Fund", "Establish 3-6 months of living expenses in liquid accounts"),
        ("Payment History", "Ensure all payments are made on time, every time"),
        ("Credit Utilization", "Keep revolving credit balances below 30% of limits"),
        ("Credit Inquiries", "Limit hard inquiries to 1-2 per year")
    ]
    
    for title, desc in guidelines:
        doc.add_heading(title, 3)
        doc.add_paragraph(desc)
    
    # Footer with disclaimer
    doc.add_page_break()
    footer_section = doc.add_heading('Disclaimer and Confidentiality', 1)
    footer = doc.add_paragraph()
    footer.add_run("CONFIDENTIAL - INTERNAL USE ONLY\n\n").bold = True
    footer.add_run("This report contains proprietary credit risk assessment methodology and results. ")
    footer.add_run("Distribution is limited to authorized personnel only.\n\n")
    
    disclaimer_text = """
This credit risk assessment is generated by automated systems using machine learning models. 
The assessment is based on the information provided at the time of analysis and uses historical 
patterns to predict creditworthiness. 

This report does not constitute a guarantee of credit approval or denial. Final credit decisions 
are made by authorized personnel considering all relevant factors. The model's predictions are 
statistical estimates with inherent uncertainty. 

Users should:
1. Verify all input data for accuracy
2. Consider qualitative factors not captured in the model
3. Review regulatory compliance requirements
4. Maintain appropriate documentation for decisions
5. Update models periodically to reflect changing conditions

The methodology and models are proprietary. Reverse engineering or unauthorized use is prohibited.
"""
    
    doc.add_paragraph(disclaimer_text)
    doc.add_paragraph()
    doc.add_paragraph("© 2026 Smart Solution to tough data. All rights reserved.")
    
    # Save document to bytes
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    return buffer

# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
def main():
    # Professional Header
    st.markdown('<h1 class="main-header">Credit Risk Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Machine Learning Credit Evaluation Platform")
    
    # Status indicator for model
    if st.session_state.model_loaded:
        st.success("✓ Machine Learning Model Loaded Successfully")
    else:
        st.error("✗ Model not available. Please ensure best_xgb_model.pkl exists.")
        st.stop()
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### Applicant Information")
        
        # Personal Information
        st.subheader("Financial Profile")
        
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input(
                "Credit Score",
                min_value=MIN_CREDIT_SCORE,
                max_value=MAX_CREDIT_SCORE,
                value=750,
                step=10,
                help="Standard UK credit score range: 0-999"
            )
        
        with col2:
            income_annum = st.number_input(
                "Annual Income (£)",
                min_value=15000,
                value=35000,
                step=5000,
                format="%d"
            )
        
        loan_amount = st.number_input(
            "Loan Amount (£)",
            min_value=5000,
            value=25000,
            step=5000,
            format="%d"
        )
        
        loan_term = st.slider(
            "Loan Term (Years)",
            min_value=1,
            max_value=MAX_LOAN_TERM,
            value=5,
            help="Maximum term: 30 years"
        )
        
        # Demographics
        st.subheader("Demographic Information")
        
        no_of_dependents = st.selectbox(
            "Number of Dependents",
            options=[0, 1, 2, 3, 4, 5],
            index=1
        )
        
        col1, col2 = st.columns(2)
        with col1:
            self_employed = st.selectbox(
                "Employment Status",
                ["Employed", "Self-Employed"]
            )
        
        with col2:
            education = st.selectbox(
                "Education Level",
                ["Graduate", "Non-Graduate"]
            )
        
        # Assets
        st.subheader("Asset Information")
        
        total_assets = st.number_input(
            "Total Asset Value (£)",
            min_value=0,
            value=50000,
            step=10000,
            format="%d",
            help="Combined value of all assets"
        )
        
        # Asset breakdown (fixed allocation for model compatibility)
        if st.checkbox("View Asset Allocation", value=False):
            st.info("""
            **Asset Allocation for Model Input:**
            - Residential Assets: 50%
            - Commercial Assets: 25%
            - Luxury Assets: 15%
            - Bank Assets: 10%
            
            *This allocation ensures compatibility with the trained model.*
            """)
        
        # Submit button
        st.markdown("---")
        analyze_button = st.button(
            "Analyze Credit Risk",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if analyze_button:
        # Validate inputs
        if credit_score < 300 or credit_score > 999:
            st.error("Credit score must be between 300 and 999")
            return
        
        if loan_amount <= 0 or income_annum <= 0:
            st.error("Loan amount and income must be positive values")
            return
        
        # Prepare input data for model
        ASSET_ALLOCATION = {
            "residential_assets_value": 0.50,
            "commercial_assets_value": 0.25,
            "luxury_assets_value": 0.15,
            "bank_asset_value": 0.10
        }
        
        # Scale for model
        model_input_data = {
            "credit_score": credit_score,
            "income_annum": int(income_annum * CONVERSION_RATE),
            "loan_amount": int(loan_amount * CONVERSION_RATE),
            "loan_term": loan_term,
            "no_of_dependents": no_of_dependents,
            "self_employed": "Yes" if self_employed == "Self-Employed" else "No",
            "education": education,
            "residential_assets_value": int((total_assets * ASSET_ALLOCATION['residential_assets_value']) * CONVERSION_RATE),
            "commercial_assets_value": int((total_assets * ASSET_ALLOCATION['commercial_assets_value']) * CONVERSION_RATE),
            "luxury_assets_value": int((total_assets * ASSET_ALLOCATION['luxury_assets_value']) * CONVERSION_RATE),
            "bank_asset_value": int((total_assets * ASSET_ALLOCATION['bank_asset_value']) * CONVERSION_RATE),
            "total_assets": int(total_assets * CONVERSION_RATE)
        }
        
        # UK display data
        display_data = {
            "credit_score": credit_score,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "no_of_dependents": no_of_dependents,
            "self_employed": self_employed,
            "education": education,
            "total_assets": total_assets
        }
        
        # Create DataFrame and engineer features
        df_input = pd.DataFrame([model_input_data])
        df_features = create_credit_features(df_input)
        
        try:
            # Get model prediction
            approval_probability, prediction, X_input = calculate_model_probability(df_input)
            
            # Generate SHAP explanation
            shap_data, shap_values = generate_shap_explanation(X_input)
            
            # Calculate matching score based on model probability
            matching_score = approval_probability
            
            # Generate recommendations
            recommendations = generate_credit_recommendations(
                approval_probability, 
                df_features.iloc[0].to_dict(),
                shap_data
            )
            
            # Determine status
            if approval_probability >= 75:
                status = "Low Risk - Recommended"
                status_class = "success-indicator"
            elif approval_probability >= 50:
                status = "Medium Risk - Conditional"
                status_class = "warning-indicator"
            else:
                status = "High Risk - Review Required"
                status_class = "critical-indicator"
            
            # Store results
            st.session_state.results = {
                'matching_score': matching_score,
                'approval_probability': approval_probability,
                'prediction': prediction,
                'status': status,
                'status_class': status_class,
                'recommendations': recommendations,
                'applicant_data': display_data,
                'features': df_features.iloc[0].to_dict(),
                'shap_data': shap_data,
                'X_input': X_input
            }
            
        except Exception as e:
            st.error(f"Model analysis failed: {str(e)}")
            return
    
    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        
        # Results Header
        st.markdown('<h2 class="section-header">Risk Assessment Results</h2>', unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Approval Probability",
                value=f"{results['approval_probability']:.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Risk Score",
                value=f"{results['matching_score']:.0f}/100"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            model_decision = "Approve" if results['prediction'] == 1 else "Review"
            st.metric(
                label="Model Decision",
                value=model_decision
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="{results["status_class"]}">{results["status"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown('<h3 class="section-header">Risk Analysis Visualizations</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            gauge_chart = create_score_gauge(results['matching_score'], "Risk Assessment Score")
            st.plotly_chart(gauge_chart, use_container_width=True)
        
        with col2:
            radar_chart = create_radar_chart(results['features'])
            st.plotly_chart(radar_chart, use_container_width=True)
        
        # SHAP Analysis
        if results.get('shap_data') is not None:
            st.markdown('<h3 class="section-header">Decision Factor Analysis</h3>', unsafe_allow_html=True)
            shap_chart = create_shap_bar_chart(results['shap_data'])
            if shap_chart:
                st.plotly_chart(shap_chart, use_container_width=True)
            
            # Top features table
            st.markdown("#### Top Contributing Factors")
            top_features = results['shap_data'].head(5).copy()
            top_features['Feature'] = top_features['Feature'].str.replace('_', ' ').str.title()
            top_features['Impact'] = top_features['Impact'].round(4)
            st.dataframe(top_features[['Feature', 'Impact']], use_container_width=True)
        
        # Credit Management Recommendations
        st.markdown('<h3 class="section-header">Credit Management Recommendations</h3>', unsafe_allow_html=True)
        
        # Group recommendations by priority
        high_priority = [r for r in results['recommendations'] if r['priority'] == 'High']
        medium_priority = [r for r in results['recommendations'] if r['priority'] == 'Medium']
        low_priority = [r for r in results['recommendations'] if r['priority'] == 'Low']
        
        if high_priority:
            st.markdown("##### High Priority Actions")
            for rec in high_priority:
                with st.expander(f"{rec['title']}", expanded=True):
                    st.write(f"**{rec['message']}**")
                    st.write("**Actions:**")
                    for action in rec['actions']:
                        st.write(f"• {action}")
        
        if medium_priority:
            st.markdown("##### Medium Priority Actions")
            for rec in medium_priority:
                with st.expander(f"{rec['title']}"):
                    st.write(f"**{rec['message']}**")
                    st.write("**Actions:**")
                    for action in rec['actions']:
                        st.write(f"• {action}")
        
        if low_priority:
            st.markdown("##### Low Priority Actions")
            for rec in low_priority:
                with st.expander(f"{rec['title']}"):
                    st.write(f"**{rec['message']}**")
                    st.write("**Actions:**")
                    for action in rec['actions']:
                        st.write(f"• {action}")
        
        # Report Generation
        st.markdown('<h3 class="section-header">Generate Comprehensive Report</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            Generate a detailed Word document report containing:
            - Complete risk assessment results
            - Visual charts and analysis
            - Credit management recommendations
            - Model decision factors
            - Risk management guidelines
            """)
        
        with col2:
            try:
                # Create charts for document
                charts = {
                    'gauge': create_score_gauge(results['matching_score']),
                    'radar': create_radar_chart(results['features']),
                    'shap': create_shap_bar_chart(results['shap_data'])
                }
                
                # Generate Word document
                word_buffer = create_word_document(
                    results['applicant_data'],
                    results,
                    results['features'],
                    results.get('shap_data'),
                    charts
                )
                
                # Download button
                current_date = datetime.now().strftime("%Y%m%d")
                b64 = base64.b64encode(word_buffer.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="Credit_Assessment_{current_date}.docx" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: 600; text-align: center;">Download Full Report (.docx)</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Report generation error: {str(e)}")
    
    else:
        # Welcome and instructions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### System Overview
            
            This Credit Risk Assessment System utilizes advanced machine learning to evaluate loan applications
            based on comprehensive financial and demographic factors.
            
            **Assessment Methodology:**
            
            1. **Data Collection** - Input financial and demographic information
            2. **Feature Engineering** - Transform raw data into predictive features
            3. **Model Prediction** - XGBoost model analyzes creditworthiness
            4. **SHAP Analysis** - Explainable AI reveals decision factors
            5. **Risk Scoring** - Generate comprehensive risk assessment
            6. **Recommendations** - Actionable credit management guidance
            
            **Model Specifications:**
            - Algorithm: XGBoost Classifier
            - Training Data: Historical credit decisions
            - Features: 12 engineered variables
            - Validation: Cross-validated accuracy >85%
            - Explainability: SHAP-based feature importance
            
            **Key Risk Indicators:**
            - Credit Score (0-999 scale)
            - Debt-to-Income Ratio
            - Asset Coverage
            - Employment Stability
            - Payment History Patterns
            """)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>Input Requirements</h4>
            <p><strong>Complete all fields for accurate assessment:</strong></p>
            <ul style="padding-left: 1.2rem;">
            <li>Valid credit score (300-999)</li>
            <li>Accurate annual income</li>
            <li>Realistic loan amount</li>
            <li>Current employment status</li>
            <li>Total asset valuation</li>
            <li>Dependent information</li>
            </ul>
            </div>
            
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <h4 style="color: #1e293b; margin-top: 0;">Risk Categories</h4>
            <p style="margin-bottom: 0.5rem;"><strong>Low Risk (≥75%):</strong> Standard approval process</p>
            <p style="margin-bottom: 0.5rem;"><strong>Medium Risk (50-74%):</strong> Additional review required</p>
            <p style="margin-bottom: 0;"><strong>High Risk (<50%):</strong> Comprehensive review needed</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.8rem; line-height: 1.4;">
    <p><strong>Disclaimer:</strong> This credit risk assessment is generated by automated systems using machine learning models. 
    The assessment is based on statistical patterns and historical data. It does not guarantee credit approval or denial. 
    Final decisions require human review and consideration of all relevant factors. All models have inherent limitations 
    and uncertainty. Results should be used as one component of comprehensive credit evaluation.</p>
    <p style="margin-top: 0.5rem;">© 2026 Smart Solution to tough data. Proprietary and Confidential.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
