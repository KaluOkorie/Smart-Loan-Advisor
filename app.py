import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
import base64
import numpy as np
import io
import time
import shap
import os
import logging

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Base styles for both themes */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Theme-aware styling */
    .main-header {
        color: var(--primary-text-color);
    }
    
    .sub-header {
        color: var(--secondary-text-color);
    }
    
    .tip-box {
        background-color: var(--background-secondary);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: rgba(16, 185, 129, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin: 0.5rem 0;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .warning-box {
        background-color: rgba(245, 158, 11, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 0.5rem 0;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .info-box {
        background-color: rgba(14, 165, 233, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0EA5E9;
        margin: 0.5rem 0;
        border: 1px solid rgba(14, 165, 233, 0.2);
    }
    
    /* Fix metric width */
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
    
    .stMetric .css-1r6slb0 {
        overflow: visible !important;
        white-space: normal !important;
    }
    
    /* Define CSS variables for theme support */
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
        
        .tip-box {
            background-color: var(--background-secondary);
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL LOADING WITH VALIDATION - NO FALLBACK
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    """Load the trained model and feature columns - NO FALLBACK"""
    model_path = "best_xgb_model.pkl"
    features_path = "feature_columns.pkl"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None, False
    
    if not os.path.exists(features_path):
        logger.error(f"Feature columns file not found: {features_path}")
        return None, None, False
    
    try:
        model = joblib.load(model_path)
        feature_columns = joblib.load(features_path)
        logger.info("Model and features loaded successfully")
        return model, feature_columns, True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, False

# Initialize model
model, feature_columns, model_loaded = load_model_and_features()

# ---------------------------------------------------------
# INITIALIZE SESSION STATE
# ---------------------------------------------------------
if 'last_submission' not in st.session_state:
    st.session_state.last_submission = 0
if 'results' not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------
# UK FINANCIAL STANDARDS
# ---------------------------------------------------------
CONVERSION_RATE = 100
UK_AVERAGE_SALARY = 35000
UK_AVERAGE_HOUSE_PRICE = 250000
UK_ASSET_MULTIPLIER = 5

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
    
    # Stability score (for display only, not used in model prediction)
    df_engineered['stability_score'] = 0
    df_engineered.loc[df_engineered['self_employed'] == 'No', 'stability_score'] += 30
    df_engineered.loc[df_engineered['education'] == 'Graduate', 'stability_score'] += 20
    df_engineered.loc[df_engineered['no_of_dependents'] <= 2, 'stability_score'] += 10
    
    return df_engineered

def generate_detailed_recommendations(approval_probability, features, applicant_data):
    """Generate personalized, actionable recommendations based on model prediction"""
    recommendations = []
    
    # Credit Score Analysis
    credit_score = features.get('credit_score', 0)
    if credit_score >= 800:
        recommendations.append({
            "title": "Excellent Credit Standing",
            "message": f"Your credit score of {credit_score} is in the top tier for UK lending.",
            "actions": [
                "You qualify for the best interest rates available",
                "Consider negotiating for premium loan terms",
                "Maintain this score by keeping credit utilization below 25%"
            ],
            "box_type": "success"
        })
    elif credit_score >= 700:
        recommendations.append({
            "title": "Good Credit Profile",
            "message": f"Your score of {credit_score} meets most lenders' requirements.",
            "actions": [
                f"Aim for 750+ to qualify for 0.5% lower rates",
                "Check your credit report for any minor issues",
                "Avoid new credit applications for 3 months before applying"
            ],
            "box_type": "success"
        })
    elif credit_score >= 600:
        recommendations.append({
            "title": "Credit Improvement Opportunity",
            "message": f"Your score of {credit_score} is acceptable but could be improved.",
            "actions": [
                "Register on the electoral roll if not already",
                "Reduce credit card balances below 30% of limits",
                "Consider a credit builder card for 6 months"
            ],
            "box_type": "warning"
        })
    else:
        recommendations.append({
            "title": "Credit Building Needed",
            "message": f"Your score of {credit_score} needs attention before applying.",
            "actions": [
                "Obtain your full credit report from all three UK agencies",
                "Dispute any incorrect information immediately",
                "Build 12 months of clean credit history before applying"
            ],
            "box_type": "warning"
        })
    
    # Payment Affordability Analysis
    payment_ratio = features.get('monthly_payment_ratio', 0)
    monthly_payment = applicant_data['loan_amount'] / (applicant_data['loan_term'] * 12)
    
    if payment_ratio <= 30:
        recommendations.append({
            "title": "Strong Payment Capacity",
            "message": f"Your monthly payment of £{monthly_payment:.0f} represents only {payment_ratio:.1f}% of your income.",
            "actions": [
                "You have comfortable repayment capacity",
                "Consider shorter loan terms for lower total interest",
                "You could potentially borrow 15-20% more if needed"
            ],
            "box_type": "success"
        })
    elif payment_ratio <= 40:
        recommendations.append({
            "title": "Manageable Payment Load",
            "message": f"Your payment of £{monthly_payment:.0f} is {payment_ratio:.1f}% of income - within acceptable limits.",
            "actions": [
                "Ensure you have 3-6 months of emergency savings",
                "Consider increasing loan term by 1-2 years to reduce monthly burden",
                "Review your monthly budget for other expenses"
            ],
            "box_type": "info"
        })
    else:
        recommendations.append({
            "title": "High Payment Burden",
            "message": f"At {payment_ratio:.1f}% of income, your payment may strain your budget.",
            "actions": [
                f"Consider reducing loan amount by £{applicant_data['loan_amount'] * 0.1:.0f} to improve affordability",
                "Extend loan term to reduce monthly payments",
                "Explore joint applications to increase household income"
            ],
            "box_type": "warning"
        })
    
    # Asset Coverage Analysis
    asset_coverage = features.get('asset_coverage', 0)
    if asset_coverage >= 200:
        recommendations.append({
            "title": "Exceptional Asset Security",
            "message": f"Your assets cover {asset_coverage:.1f}% of the loan - excellent security.",
            "actions": [
                "You may qualify for lower interest rates due to strong collateral",
                "Consider using assets as security for better terms",
                "Document all assets clearly in your application"
            ],
            "box_type": "success"
        })
    elif asset_coverage >= 125:
        recommendations.append({
            "title": "Adequate Asset Coverage",
            "message": f"Your {asset_coverage:.1f}% asset coverage meets standard requirements.",
            "actions": [
                "Include pension fund statements if applicable",
                "Document property equity with recent valuations",
                "Consider consolidating smaller assets in your application"
            ],
            "box_type": "info"
        })
    else:
        recommendations.append({
            "title": "Asset Coverage Could Improve",
            "message": f"At {asset_coverage:.1f}%, your asset coverage is below ideal levels.",
            "actions": [
                "Build savings for 6 months to increase liquid assets",
                "Consider a guarantor if asset coverage is a concern",
                "Focus on building emergency fund to 3-6 months of expenses"
            ],
            "box_type": "warning"
        })
    
    # Model-Based Recommendation
    if approval_probability >= 85:
        recommendations.append({
            "title": "Premium Application Candidate",
            "message": f"With an approval probability of {approval_probability:.1f}%, you're in the top tier of applicants.",
            "actions": [
                "Apply to multiple lenders to compare offers",
                "Expect decision within 24-48 hours",
                "You have strong negotiating power for terms"
            ],
            "box_type": "success"
        })
    elif approval_probability >= 70:
        recommendations.append({
            "title": "Strong Application Position",
            "message": f"Your approval probability of {approval_probability:.1f}% indicates high approval likelihood.",
            "actions": [
                "Complete full application with all supporting documents",
                "Apply to 2-3 preferred lenders",
                "Expected processing: 3-5 working days"
            ],
            "box_type": "success"
        })
    elif approval_probability >= 55:
        recommendations.append({
            "title": "Borderline Application",
            "message": f"At {approval_probability:.1f}% approval probability, your application needs careful preparation.",
            "actions": [
                "Provide detailed explanations for any credit issues",
                "Include letters from employers confirming stable income",
                "Consider applying with a co-signer for better terms"
            ],
            "box_type": "warning"
        })
    else:
        recommendations.append({
            "title": "Profile Needs Strengthening",
            "message": f"Your current approval probability of {approval_probability:.1f}% suggests waiting to apply.",
            "actions": [
                "Focus on improving credit score for 6-12 months",
                "Increase savings by £5,000-£10,000",
                "Consult with a free financial advisor before applying"
            ],
            "box_type": "warning"
        })
    
    return recommendations

def generate_shap_explanation(model, X_input, feature_names):
    """Generate SHAP-based explanation for the decision"""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        
        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class
        
        # Get feature importance
        feature_importance = np.abs(shap_values).mean(0)
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': feature_importance[0] if len(feature_importance.shape) > 1 else feature_importance
        }).sort_values('Impact', ascending=False).head(10)
        
        return feature_df
    except Exception as e:
        logger.warning(f"SHAP explanation limited: {str(e)}")
        return None

# ---------------------------------------------------------
# MODEL PREDICTION FUNCTION - NO FALLBACK
# ---------------------------------------------------------
def get_model_prediction(model, feature_columns, df_input):
    """Get prediction ONLY from the trained model - NO FALLBACK"""
    if model is None or feature_columns is None:
        raise ValueError("Model or feature columns not available")
    
    try:
        # Prepare features for model
        X_input = df_input.drop(columns=['total_assets'])
        X_input = pd.get_dummies(X_input)
        X_input = X_input.reindex(columns=feature_columns, fill_value=0)
        
        # Get prediction
        approval_probability = model.predict_proba(X_input)[0, 1] * 100
        prediction = model.predict(X_input)[0]
        
        # Generate SHAP explanation
        shap_data = generate_shap_explanation(model, X_input, feature_columns)
        
        logger.info(f"Model prediction successful: {approval_probability:.1f}%")
        return approval_probability, prediction, shap_data
        
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        raise ValueError(f"Model prediction failed: {str(e)}")

# ---------------------------------------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def create_score_gauge(score):
    """Create a gauge chart for approval probability"""
    colors = ['#EF4444', '#F59E0B', '#10B981', '#047857']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Approval Probability", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#3B82F6"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': colors[0]},
                {'range': [50, 70], 'color': colors[1]},
                {'range': [70, 85], 'color': colors[2]},
                {'range': [85, 100], 'color': colors[3]}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.85,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='var(--primary-text-color)', family="Arial"),
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_feature_radar(features):
    """Create radar chart for key metrics"""
    categories = ['Credit Score', 'Affordability', 'Asset Security', 'Stability']
    values = [
        min(100, features['credit_score'] / 9.99),
        100 - min(100, features['monthly_payment_ratio']),
        min(100, features['asset_coverage'] / 2.5),
        min(100, features['stability_score'] / 45 * 100)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line_color='rgb(59, 130, 246)',
        line_width=2
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(128, 128, 128, 0.3)',
                linecolor='gray'
            ),
            angularaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.3)',
                linecolor='gray'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='var(--primary-text-color)'),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=30, b=30)
    )
    
    return fig

def create_shap_chart(shap_data):
    """Create horizontal bar chart for SHAP feature importance"""
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
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='var(--primary-text-color)'),
        height=300,
        margin=dict(l=100, r=20, t=40, b=20)
    )
    
    return fig

# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
def main():
    # Professional header
    st.markdown('<h1 class="main-header">Smart Loan Advisor</h1>', unsafe_allow_html=True)
    
    # Check if model is loaded - CRITICAL: NO FALLBACK
    if not model_loaded:
        st.error("""
        ## Model Not Available
        
        The machine learning model is required to run this application but could not be loaded.
        
        **Please ensure the following files are in the application directory:**
        1. `best_xgb_model.pkl` - The trained model
        2. `feature_columns.pkl` - Feature columns for the model
        
        Without these files, the application cannot perform loan assessments.
        
        **Next Steps:**
        - Upload the model files to the app directory
        - Restart the application
        - Contact support if you need assistance obtaining the model files
        """)
        st.stop()  # Stop execution completely
    
    st.markdown("""
    <div class="tip-box">
    <strong>Transparent Loan Assessment:</strong> Get instant, data-driven feedback on your loan eligibility. 
    This tool uses a trained machine learning model to analyze your profile against UK lending criteria. 
    Final approval requires full documentation and verification by a UK lender.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown('<h3 class="sub-header">Your Information</h3>', unsafe_allow_html=True)
        
        # Personal Information
        st.subheader("Personal Details")
        credit_score = st.slider(
            "Credit Score (UK 0-999)",
            min_value=0,
            max_value=999,
            value=750,
            help="UK credit scores typically range from 0-999. Scores above 700 are considered good."
        )
        
        income_annum = st.number_input(
            "Annual Income (£)",
            min_value=15000,
            value=35000,
            step=5000,
            help="Your gross annual income before tax deductions"
        )
        
        loan_amount = st.number_input(
            "Loan Amount Needed (£)",
            min_value=5000,
            value=25000,
            step=5000,
            help="The total amount you wish to borrow"
        )
        
        loan_term = st.slider(
            "Loan Term (Years)",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of years to repay the loan"
        )
        
        no_of_dependents = st.selectbox(
            "Number of Dependents",
            options=[0, 1, 2, 3, 4, "5 or more"],
            index=1,
            help="People who financially depend on you"
        )
        
        # Employment & Education
        col1, col2 = st.columns(2)
        with col1:
            self_employed = st.radio(
                "Employment Type",
                ["Employed", "Self-Employed"],
                help="Salaried employment vs self-employment"
            )
        with col2:
            education = st.radio(
                "Education Level",
                ["Graduate", "Not Graduate"],
                help="University degree holder or not"
            )
        
        # Assets
        st.markdown('<h3 class="sub-header">Your Assets</h3>', unsafe_allow_html=True)
        total_assets = st.number_input(
            "Total Assets Value (£)",
            min_value=0,
            value=50000,
            step=10000,
            help="Combined value of savings, properties, investments, and vehicles"
        )
        
        if st.checkbox("Show asset breakdown", value=False):
            st.info("""
            **For Model Compatibility:**
            - 50% → Residential Assets (property, land)
            - 25% → Commercial Assets (business equipment)
            - 15% → Luxury Assets (vehicles, jewelry)
            - 10% → Bank Assets (savings, investments)
            """)
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_submission
        
        if time_since_last < 3 and st.session_state.results is not None:
            remaining = 3 - time_since_last
            st.warning(f"Please wait {remaining:.1f} seconds")
            calculate_disabled = True
        else:
            calculate_disabled = False
        
        # Calculate Button
        calculate_btn = st.button(
            "Check My Eligibility",
            type="primary",
            disabled=calculate_disabled,
            use_container_width=True
        )
    
    # Main Content Area
    if calculate_btn:
        # Update last submission time
        st.session_state.last_submission = time.time()
        
        # Validate inputs
        validation_errors = []
        if loan_term > 30:
            validation_errors.append("Loan term cannot exceed 30 years")
        if income_annum <= 0:
            validation_errors.append("Annual income must be positive")
        if not (0 <= credit_score <= 999):
            validation_errors.append("Credit score must be between 0-999")
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"{error}")
            return
        
        # Asset allocation
        ASSET_ALLOCATION = {
            "residential_assets_value": 0.50,
            "commercial_assets_value": 0.25,
            "luxury_assets_value": 0.15,
            "bank_asset_value": 0.10
        }
        
        # Convert to model scale
        residential_assets = int((total_assets * ASSET_ALLOCATION['residential_assets_value']) * CONVERSION_RATE)
        commercial_assets = int((total_assets * ASSET_ALLOCATION['commercial_assets_value']) * CONVERSION_RATE)
        luxury_assets = int((total_assets * ASSET_ALLOCATION['luxury_assets_value']) * CONVERSION_RATE)
        bank_assets = int((total_assets * ASSET_ALLOCATION['bank_asset_value']) * CONVERSION_RATE)
        
        # Prepare input data for model
        model_input_data = {
            "credit_score": credit_score,
            "income_annum": int(income_annum * CONVERSION_RATE),
            "loan_amount": int(loan_amount * CONVERSION_RATE),
            "loan_term": loan_term,
            "no_of_dependents": no_of_dependents if isinstance(no_of_dependents, int) else 5,
            "self_employed": "Yes" if self_employed == "Self-Employed" else "No",
            "education": education,
            "residential_assets_value": residential_assets,
            "commercial_assets_value": commercial_assets,
            "luxury_assets_value": luxury_assets,
            "bank_asset_value": bank_assets,
            "total_assets": int(total_assets * CONVERSION_RATE)
        }
        
        # UK display data (for user-facing info)
        uk_display_data = {
            "credit_score": credit_score,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "no_of_dependents": no_of_dependents if isinstance(no_of_dependents, int) else 5,
            "self_employed": self_employed,
            "education": education,
            "total_assets": total_assets
        }
        
        # Create DataFrame for model
        df_input = pd.DataFrame([model_input_data])
        
        # Feature engineering for display purposes
        df_features = create_credit_features(df_input)
        
        try:
            # Get model prediction - NO FALLBACK
            approval_probability, prediction, shap_data = get_model_prediction(
                model, feature_columns, df_input
            )
            
            # Generate detailed recommendations based on model prediction
            recommendations = generate_detailed_recommendations(
                approval_probability, 
                df_features.iloc[0].to_dict(),
                uk_display_data
            )
            
            # Determine status based on model prediction
            if approval_probability >= 80:
                status = "Excellent Match"
                status_box = "success"
            elif approval_probability >= 65:
                status = "Good Match"
                status_box = "success"
            elif approval_probability >= 50:
                status = "Needs Review"
                status_box = "warning"
            else:
                status = "Needs Improvement"
                status_box = "warning"
            
            # Store results
            st.session_state.results = {
                'approval_probability': approval_probability,
                'prediction': prediction,
                'status': status,
                'status_box': status_box,
                'recommendations': recommendations,
                'applicant_data': uk_display_data,
                'features': df_features.iloc[0].to_dict(),
                'shap_data': shap_data,
                'model_used': True
            }
            
        except Exception as e:
            st.error(f"""
            ## Model Prediction Error
            
            The machine learning model failed to process your application:
            
            **Error:** {str(e)}
            
            **Please try:**
            1. Checking your input values are within reasonable ranges
            2. Ensuring all required fields are filled
            3. Contacting support if the issue persists
            
            Without a successful model prediction, no assessment can be provided.
            """)
            return
    
    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        
        # Results Header
        st.markdown('<h2 class="sub-header">Your Eligibility Report</h2>', unsafe_allow_html=True)
        
        # Key Metrics in Columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Approval Probability",
                value=f"{results['approval_probability']:.1f}%",
                delta=f"{results['status']}",
                delta_color="normal" if results['status_box'] == "success" else "off"
            )
        
        with col2:
            # Determine processing time based on approval probability
            if results['approval_probability'] >= 70:
                time_value = "2-4 Days"
            elif results['approval_probability'] >= 50:
                time_value = "5-10 Days"
            else:
                time_value = "Manual Review"
            st.metric(
                label="Processing Time",
                value=time_value
            )
        
        with col3:
            action_value = "Apply Now" if results['approval_probability'] >= 65 else "Improve First"
            st.metric(
                label="Next Action",
                value=action_value
            )
        
        with col4:
            # Loan decision
            decision = "Likely Approved" if results['approval_probability'] >= 60 else "Further Review Needed"
            st.metric(
                label="Preliminary Decision",
                value=decision
            )
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_score_gauge(results['approval_probability']), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_feature_radar(results['features']), use_container_width=True)
        
        # SHAP Explanation Chart if available
        if results.get('shap_data') is not None:
            shap_chart = create_shap_chart(results['shap_data'])
            if shap_chart:
                st.plotly_chart(shap_chart, use_container_width=True)
        
        # Financial Health Indicators
        st.markdown('<h3 class="sub-header">Financial Health Check</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            payment_ratio = results['features']['monthly_payment_ratio']
            st.write("**Payment Affordability**")
            uk_target = 35
            progress_value = max(0.0, min(1.0, (uk_target - min(payment_ratio, uk_target)) / uk_target))
            st.progress(
                progress_value,
                text=f"£{results['applicant_data']['loan_amount']/results['applicant_data']['loan_term']/12:.0f}/month"
            )
            st.caption(f"{payment_ratio:.1f}% of monthly income (UK target: ≤35%)")
        
        with col2:
            asset_coverage = results['features']['asset_coverage']
            st.write("**Asset Security**")
            progress_value = min(1.0, asset_coverage / 250)
            st.progress(
                progress_value,
                text=f"£{results['applicant_data']['total_assets']:,}"
            )
            st.caption(f"{asset_coverage:.1f}% of loan amount (Ideal: ≥125%)")
        
        with col3:
            stability = results['features']['stability_score']
            st.write("**Financial Stability**")
            progress_value = min(1.0, stability / 60)
            st.progress(
                progress_value,
                text=f"{stability}/60 points"
            )
            st.caption("Based on employment type, education, and family size")
        
        # Personalized Recommendations
        st.markdown('<h3 class="sub-header">Your Action Plan</h3>', unsafe_allow_html=True)
        
        for i, rec in enumerate(results['recommendations'][:6]):
            if rec['box_type'] == "success":
                box_class = "success-box"
            elif rec['box_type'] == "warning":
                box_class = "warning-box"
            else:
                box_class = "info-box"
            
            with st.expander(f"{rec['title']}", expanded=i<2):
                st.markdown(f"<div class='{box_class}'>", unsafe_allow_html=True)
                st.write(f"**{rec['message']}**")
                st.write("")
                st.write("**Recommended Actions:**")
                for action in rec['actions']:
                    st.write(f"• {action}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Next Steps
        st.markdown('<h3 class="sub-header">Next Steps</h3>', unsafe_allow_html=True)
        
        if results['approval_probability'] >= 65:
            st.markdown("""
            <div class="success-box">
            <h4>Ready to Apply Pathway</h4>
            <p><strong>Your profile shows strong approval potential.</strong> Here's your recommended approach:</p>
            <ol>
            <li><strong>Gather Documentation:</strong> 3 months of bank statements, proof of address, ID documents</li>
            <li><strong>Compare Lenders:</strong> Apply to 2-3 reputable UK lenders to compare offers</li>
            <li><strong>Submit Application:</strong> Complete online forms with all supporting documents</li>
            <li><strong>Follow Up:</strong> Expect initial decision within 2-4 working days</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            <h4>Profile Improvement Pathway</h4>
            <p><strong>Your profile needs strengthening before application.</strong> Focus on these areas:</p>
            <ol>
            <li><strong>Credit Building:</strong> Obtain free credit reports and address any issues</li>
            <li><strong>Savings Growth:</strong> Build emergency fund to 3-6 months of expenses</li>
            <li><strong>Debt Management:</strong> Reduce existing debts to improve affordability ratios</li>
            <li><strong>Professional Advice:</strong> Consult with free financial advisors for personalized guidance</li>
            </ol>
            <p><em>Reassess in 3-6 months for improved eligibility.</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Welcome screen - only shown if model is loaded
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### How It Works
            
            **1. Enter Your Financial Profile**
            Provide details about your income, expenses, assets, and credit history.
            
            **2. Get Machine Learning Assessment**
            Our trained model analyzes your profile using advanced algorithms against UK lending criteria.
            
            **3. Understand Your Probability**
            See your approval probability percentage and detailed breakdown of key factors.
            
            **4. Receive Actionable Advice**
            Get personalized recommendations to improve your financial position.
            
            **5. Download Professional Report**
            Generate a comprehensive PDF report with all assessment details.
            
            ### What We Assess
            
            • **Credit Health**: Your credit score and payment history  
            • **Payment Capacity**: Can you comfortably afford the monthly payments?  
            • **Asset Security**: Do you have sufficient assets as financial backup?  
            • **Employment Stability**: Job security and income consistency  
            • **Risk Factors**: Self-employment status, dependents, loan terms  
            
            ### Why Use This Tool?
            
            • **Data-Driven**: Uses trained machine learning model for accurate predictions  
            • **Transparent**: See exactly how decisions are made with SHAP explanations  
            • **Educational**: Learn what lenders look for  
            • **Actionable**: Get specific steps to improve your profile  
            • **Private**: Your data is processed securely and not stored  
            """)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>UK Financial Tips</h4>
            <ul style="padding-left: 1.2rem; margin-bottom: 0;">
            <li><strong>Credit Score ≥700</strong> for best interest rates</li>
            <li><strong>Monthly payments ≤35%</strong> of net income</li>
            <li><strong>Assets should cover ≥125%</strong> of loan amount</li>
            <li><strong>Being on electoral roll</strong> boosts credit score</li>
            <li><strong>3+ years at current address</strong> improves stability</li>
            <li><strong>No missed payments</strong> for 12+ months</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick example
            st.markdown("""
            <div style="background-color: var(--background-secondary); padding: 1.5rem; border-radius: 10px; margin-top: 1.5rem;">
            <h4 style="margin-top: 0;">Typical Strong Profile</h4>
            <p style="margin-bottom: 0.5rem;"><strong>Credit Score:</strong> 750+</p>
            <p style="margin-bottom: 0.5rem;"><strong>Income:</strong> £40,000+</p>
            <p style="margin-bottom: 0.5rem;"><strong>Assets:</strong> £75,000+</p>
            <p style="margin-bottom: 0;"><strong>Result:</strong> 85-95% Approval</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--secondary-text-color); font-size: 0.85rem; line-height: 1.5;">
    <p><strong>Important Disclaimer:</strong> This tool provides preliminary assessment only using a trained machine learning model. Final loan approval is subject to complete documentation, credit checks, and individual lender policies. Approval probability estimates are based on historical data and machine learning patterns. Results are not a guarantee of approval. Always consult with qualified financial advisors before making borrowing decisions.</p>
    <p style="margin-top: 0.5rem;">© 2024 Smart Loan Advisor • Powered by Machine Learning • UK Representative APR 4.9% - 19.9%</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
