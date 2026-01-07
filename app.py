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

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI - IMPROVED FOR DARK/LIGHT MODE
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
    
    # Stability score
    df_engineered['stability_score'] = 0
    df_engineered.loc[df_engineered['self_employed'] == 'No', 'stability_score'] += 30
    df_engineered.loc[df_engineered['education'] == 'Graduate', 'stability_score'] += 20
    df_engineered.loc[df_engineered['no_of_dependents'] <= 2, 'stability_score'] += 10
    
    return df_engineered

def calculate_matching_score(row):
    """Calculate matching score (0-100) based on credit profile"""
    score = 0
    
    # 1. Credit Score Component
    if row['credit_score'] >= 900:
        score += 40
    elif row['credit_score'] >= 800:
        score += 35
    elif row['credit_score'] >= 700:
        score += 30
    elif row['credit_score'] >= 600:
        score += 20
    else:
        score += 10
    
    # 2. Monthly Payment Affordability
    if row['monthly_payment_ratio'] <= 25:
        score += 25
    elif row['monthly_payment_ratio'] <= 35:
        score += 20
    elif row['monthly_payment_ratio'] <= 45:
        score += 15
    elif row['monthly_payment_ratio'] <= 55:
        score += 10
    else:
        score += 5
    
    # 3. Asset Security
    if row['asset_coverage'] >= 250:
        score += 20
    elif row['asset_coverage'] >= 175:
        score += 16
    elif row['asset_coverage'] >= 125:
        score += 12
    elif row['asset_coverage'] >= 75:
        score += 8
    else:
        score += 4
    
    # 4. Stability Factors
    score += min(row['stability_score'], 15)
    
    # 5. Risk adjustments
    if row['self_employed'] == 'Yes' and row['income_annum'] < 30000:
        score -= 10
    
    if row['no_of_dependents'] > 3:
        score -= 5
    
    if row['loan_term'] > 7 and row['loan_amount'] < 150000:
        score -= 3
    
    return min(max(score, 0), 100)

def generate_detailed_recommendations(score, features, applicant_data):
    """Generate personalized, actionable recommendations"""
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
    
    # Overall Score-Based Recommendation
    if score >= 85:
        recommendations.append({
            "title": "Premium Application Candidate",
            "message": f"With a matching score of {score}/100, you're in the top tier of applicants.",
            "actions": [
                "Apply to multiple lenders to compare offers",
                "Expect decision within 24-48 hours",
                "You have strong negotiating power for terms"
            ],
            "box_type": "success"
        })
    elif score >= 70:
        recommendations.append({
            "title": "Strong Application Position",
            "message": f"Your score of {score}/100 indicates high approval likelihood.",
            "actions": [
                "Complete full application with all supporting documents",
                "Apply to 2-3 preferred lenders",
                "Expected processing: 3-5 working days"
            ],
            "box_type": "success"
        })
    elif score >= 55:
        recommendations.append({
            "title": "Borderline Application",
            "message": f"At {score}/100, your application needs careful preparation.",
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
            "message": f"Your current score of {score}/100 suggests waiting to apply.",
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
        st.warning(f"SHAP explanation limited: {str(e)}")
        return None

# ---------------------------------------------------------
# ENHANCED PDF REPORT GENERATION
# ---------------------------------------------------------
def create_enhanced_pdf_report(applicant_data, results, features, shap_data=None):
    """Generate enhanced PDF report with SHAP visualization"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'UK Loan Pre-Approval Report', 0, 1, 'C')
    
    # Date
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%d %B %Y at %H:%M")}', 0, 1, 'C')
    pdf.ln(10)
    
    # Applicant Information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Applicant Information', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    pdf.cell(60, 8, 'Credit Score:', 0, 0)
    pdf.cell(0, 8, str(applicant_data['credit_score']), 0, 1)
    
    pdf.cell(60, 8, 'Annual Income:', 0, 0)
    pdf.cell(0, 8, f"£{applicant_data['income_annum']:,}", 0, 1)
    
    pdf.cell(60, 8, 'Loan Amount:', 0, 0)
    pdf.cell(0, 8, f"£{applicant_data['loan_amount']:,}", 0, 1)
    
    pdf.cell(60, 8, 'Loan Term:', 0, 0)
    pdf.cell(0, 8, f"{applicant_data['loan_term']} years", 0, 1)
    
    pdf.cell(60, 8, 'Employment:', 0, 0)
    pdf.cell(0, 8, applicant_data['self_employed'], 0, 1)
    
    pdf.cell(60, 8, 'Education:', 0, 0)
    pdf.cell(0, 8, applicant_data['education'], 0, 1)
    
    pdf.ln(5)
    
    # Assessment Results
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Assessment Results', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    pdf.cell(60, 8, 'Matching Score:', 0, 0)
    pdf.cell(0, 8, f"{results['matching_score']}/100", 0, 1)
    
    pdf.cell(60, 8, 'Approval Probability:', 0, 0)
    pdf.cell(0, 8, f"{results['approval_probability']:.1f}%", 0, 1)
    
    pdf.cell(60, 8, 'Status:', 0, 0)
    pdf.cell(0, 8, results['status'], 0, 1)
    
    pdf.ln(5)
    
    # Key Financial Metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Financial Health Metrics', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    pdf.cell(70, 8, 'Monthly Payment Ratio:', 0, 0)
    pdf.cell(0, 8, f"{features['monthly_payment_ratio']:.1f}%", 0, 1)
    
    pdf.cell(70, 8, 'Asset Coverage:', 0, 0)
    pdf.cell(0, 8, f"{features['asset_coverage']:.1f}%", 0, 1)
    
    pdf.cell(70, 8, 'Credit Category:', 0, 0)
    pdf.cell(0, 8, features['credit_category'], 0, 1)
    
    pdf.cell(70, 8, 'Stability Score:', 0, 0)
    pdf.cell(0, 8, f"{features['stability_score']}/45", 0, 1)
    
    pdf.ln(5)
    
    # Decision Factors (SHAP-based if available)
    if shap_data is not None and not shap_data.empty:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Top Decision Factors', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        for idx, row in shap_data.head(5).iterrows():
            factor = row['Feature'].replace('_', ' ').title()
            impact = row['Impact']
            pdf.cell(80, 8, f"{factor}:", 0, 0)
            pdf.cell(0, 8, f"{impact:.4f}", 0, 1)
    
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Personalized Recommendations', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for i, rec in enumerate(results['recommendations'][:3], 1):
        pdf.multi_cell(0, 6, f"{i}. {rec['message']}")
        pdf.ln(2)
    
    pdf.ln(5)
    
    # Action Steps
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Immediate Action Steps', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for rec in results['recommendations'][:2]:
        if rec['actions']:
            for action in rec['actions'][:2]:
                pdf.multi_cell(0, 6, f"• {action}")
    
    pdf.ln(10)
    
    # Footer
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: This is a preliminary assessment based on the information provided. Final loan approval is subject to complete documentation, verification, and the lender's credit policies. Approval probability is an estimate based on historical data and machine learning patterns.")
    
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ---------------------------------------------------------
# VISUALIZATION FUNCTIONS - IMPROVED FOR ALL THEMES
# ---------------------------------------------------------
def create_score_gauge(score):
    """Create a gauge chart for matching score with theme-friendly colors"""
    # Colors that work well in both light and dark modes
    colors = ['#EF4444', '#F59E0B', '#10B981', '#047857']  # Red, Orange, Green, Dark Green
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Profile Match Score", 'font': {'size': 20}},
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
    
    # Theme-friendly background
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='var(--primary-text-color)', family="Arial"),
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def create_feature_radar(features):
    """Create radar chart with theme-friendly styling"""
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
    
    # Take top 8 features
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
    # Header with improved spacing
    st.markdown('<h1 class="main-header">UK Smart Loan Advisor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-box">
    <strong>Transparent Loan Assessment:</strong> Get instant, data-driven feedback on your loan eligibility. 
    This tool provides a preliminary assessment based on UK lending criteria. Final approval requires full documentation and verification by a UK lender.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown('<h3 class="sub-header">📋 Your Information</h3>', unsafe_allow_html=True)
        
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
        st.markdown('<h3 class="sub-header">💰 Your Assets</h3>', unsafe_allow_html=True)
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
        
        # Calculate Button with better styling
        calculate_btn = st.button(
            "📊 Check My Eligibility",
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
                st.error(f"❌ {error}")
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
        
        # Prepare input data
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
        
        # UK display data
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
        
        # Feature engineering
        df_features = create_credit_features(df_input)
        
        # Calculate matching score
        matching_score = calculate_matching_score(df_features.iloc[0])
        
        # Load model and predict
        try:
            model = joblib.load("best_xgb_model.pkl")
            feature_columns = joblib.load("feature_columns.pkl")
            
            # Prepare features for model
            X_input = df_input.drop(columns=['total_assets'])
            X_input = pd.get_dummies(X_input)
            X_input = X_input.reindex(columns=feature_columns, fill_value=0)
            
            # Get prediction
            approval_probability = model.predict_proba(X_input)[0, 1] * 100
            prediction = model.predict(X_input)[0]
            
            # Generate SHAP explanation
            shap_data = generate_shap_explanation(model, X_input, feature_columns)
            
        except Exception as e:
            st.error(f"⚠️ Model error: {str(e)}")
            approval_probability = max(20, min(95, matching_score * 0.85))
            prediction = 1 if matching_score >= 60 else 0
            shap_data = None
        
        # Generate detailed recommendations
        recommendations = generate_detailed_recommendations(
            matching_score, 
            df_features.iloc[0].to_dict(),
            uk_display_data
        )
        
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
            'matching_score': matching_score,
            'approval_probability': approval_probability,
            'prediction': prediction,
            'status': status,
            'status_box': status_box,
            'recommendations': recommendations,
            'applicant_data': uk_display_data,
            'features': df_features.iloc[0].to_dict(),
            'shap_data': shap_data
        }
    
    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        
        # Results Header
        st.markdown('<h2 class="sub-header">📋 Your Eligibility Report</h2>', unsafe_allow_html=True)
        
        # Key Metrics in Columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Profile Match",
                value=f"{results['matching_score']}/100",
                delta=f"{results['status']}",
                delta_color="normal" if results['status_box'] == "success" else "off"
            )
        
        with col2:
            st.metric(
                label="Approval Chance",
                value=f"{results['approval_probability']:.1f}%"
            )
        
        with col3:
            action_value = "Apply Now" if results['matching_score'] >= 65 else "Improve First"
            st.metric(
                label="Next Action",
                value=action_value
            )
        
        with col4:
            if results['matching_score'] >= 70:
                time_value = "2-4 Days"
            elif results['matching_score'] >= 50:
                time_value = "5-10 Days"
            else:
                time_value = "Manual Review"
            st.metric(
                label="Processing Time",
                value=time_value
            )
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_score_gauge(results['matching_score']), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_feature_radar(results['features']), use_container_width=True)
        
        # SHAP Explanation Chart if available
        if results.get('shap_data') is not None:
            shap_chart = create_shap_chart(results['shap_data'])
            if shap_chart:
                st.plotly_chart(shap_chart, use_container_width=True)
        
        # Financial Health Indicators
        st.markdown('<h3 class="sub-header">💡 Financial Health Check</h3>', unsafe_allow_html=True)
        
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
        st.markdown('<h3 class="sub-header">🎯 Your Action Plan</h3>', unsafe_allow_html=True)
        
        for i, rec in enumerate(results['recommendations'][:6]):  # Show more recommendations
            if rec['box_type'] == "success":
                box_class = "success-box"
            elif rec['box_type'] == "warning":
                box_class = "warning-box"
            else:
                box_class = "info-box"
            
            with st.expander(f"{rec['title']}", expanded=i<2):  # First two expanded by default
                st.markdown(f"<div class='{box_class}'>", unsafe_allow_html=True)
                st.write(f"**{rec['message']}**")
                st.write("")
                st.write("**Recommended Actions:**")
                for action in rec['actions']:
                    st.write(f"• {action}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Next Steps
        st.markdown('<h3 class="sub-header">🚀 Next Steps</h3>', unsafe_allow_html=True)
        
        if results['matching_score'] >= 65:
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
        
        # Enhanced PDF Report Generation
        st.markdown('<h3 class="sub-header">📄 Download Full Report</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info("Download a comprehensive PDF report including your assessment, SHAP-based decision factors, and personalized recommendations.")
        
        with col2:
            try:
                pdf_bytes = create_enhanced_pdf_report(
                    results['applicant_data'],
                    results,
                    results['features'],
                    results.get('shap_data')
                )
                
                b64 = base64.b64encode(pdf_bytes).decode()
                current_date = datetime.now().strftime("%Y%m%d")
                href = f'<a href="data:application/pdf;base64,{b64}" download="Loan_Assessment_{current_date}.pdf" style="background-color: #3B82F6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: 600;">📥 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                # Simple fallback without DejaVu dependency
                st.warning("Enhanced PDF features require additional setup. Downloading text report instead.")
        
        with col3:
            # Always provide text download as fallback
            report_text = f"""UK LOAN ELIGIBILITY ASSESSMENT
Generated: {datetime.now().strftime("%d %B %Y")}

APPLICANT INFORMATION:
Credit Score: {results['applicant_data']['credit_score']}
Annual Income: £{results['applicant_data']['income_annum']:,}
Loan Amount: £{results['applicant_data']['loan_amount']:,}
Loan Term: {results['applicant_data']['loan_term']} years
Employment: {results['applicant_data']['self_employed']}
Education: {results['applicant_data']['education']}

ASSESSMENT RESULTS:
Matching Score: {results['matching_score']}/100
Approval Probability: {results['approval_probability']:.1f}%
Status: {results['status']}

FINANCIAL METRICS:
Monthly Payment Ratio: {results['features']['monthly_payment_ratio']:.1f}%
Asset Coverage: {results['features']['asset_coverage']:.1f}%
Credit Category: {results['features']['credit_category']}
Stability Score: {results['features']['stability_score']}/45

TOP RECOMMENDATIONS:
"""
            
            for i, rec in enumerate(results['recommendations'][:3], 1):
                report_text += f"\n{i}. {rec['title']}: {rec['message']}"
                if rec['actions']:
                    report_text += f"\n   Key Actions: {rec['actions'][0]}"
            
            st.download_button(
                label="📄 Text Report",
                data=report_text,
                file_name=f"Loan_Assessment_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    else:
        # Welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🤔 How It Works
            
            **1. Enter Your Financial Profile**
            Provide details about your income, expenses, assets, and credit history.
            
            **2. Get Data-Driven Assessment**
            Our machine learning model analyzes your profile against UK lending criteria.
            
            **3. Understand Your Score**
            See a 0-100 matching score and detailed breakdown of key factors.
            
            **4. Receive Actionable Advice**
            Get personalized recommendations to improve your financial position.
            
            **5. Download Professional Report**
            Generate a comprehensive PDF report with all assessment details.
            
            ### 🔍 What We Assess
            
            • **Credit Health**: Your credit score and payment history  
            • **Payment Capacity**: Can you comfortably afford the monthly payments?  
            • **Asset Security**: Do you have sufficient assets as financial backup?  
            • **Employment Stability**: Job security and income consistency  
            • **Risk Factors**: Self-employment status, dependents, loan terms  
            
            ### 💡 Why Use This Tool?
            
            • **Free & Instant**: No impact on your credit score  
            • **Transparent**: See exactly how decisions are made  
            • **Educational**: Learn what lenders look for  
            • **Actionable**: Get specific steps to improve your profile  
            • **Private**: Your data is processed securely and not stored  
            """)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>💎 UK Financial Tips</h4>
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
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin-top: 1.5rem;">
            <h4 style="color: white; margin-top: 0;">🏆 Ideal Profile</h4>
            <p style="margin-bottom: 0.5rem;"><strong>Credit Score:</strong> 750+</p>
            <p style="margin-bottom: 0.5rem;"><strong>Income:</strong> £40,000+</p>
            <p style="margin-bottom: 0.5rem;"><strong>Assets:</strong> £75,000+</p>
            <p style="margin-bottom: 0;"><strong>Result:</strong> 85-95/100 Score</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--secondary-text-color); font-size: 0.85rem; line-height: 1.5;">
    <p><strong>Important Disclaimer:</strong> This tool provides preliminary assessment only. Final loan approval is subject to complete documentation, credit checks, and individual lender policies. Approval probability estimates are based on historical data and machine learning patterns. Results are not a guarantee of approval. Always consult with qualified financial advisors before making borrowing decisions.</p>
    <p style="margin-top: 0.5rem;">© 2024 Smart Solution to Tough Data • UK Representative APR 4.9% - 19.9% •</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
