import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import base64
import shap
from docx import Document
from docx.shared import Inches
import io

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
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .section-title {
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INITIALIZE SESSION STATE
# ---------------------------------------------------------
if 'results' not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    """Load the trained XGBoost model and feature columns"""
    try:
        model = joblib.load("best_xgb_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, feature_columns
    except FileNotFoundError:
        st.error("Model files not found. Please ensure best_xgb_model.pkl and feature_columns.pkl exist in the project directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load model at startup
model, feature_columns = load_ml_model()

# ---------------------------------------------------------
# FINANCIAL CONFIGURATION
# ---------------------------------------------------------
CONVERSION_RATE = 100

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def prepare_model_input(applicant_data):
    """Prepare input data for ML model"""
    ASSET_ALLOCATION = {
        "residential_assets_value": 0.50,
        "commercial_assets_value": 0.25,
        "luxury_assets_value": 0.15,
        "bank_asset_value": 0.10
    }
    
    model_input_data = {
        "credit_score": applicant_data['credit_score'],
        "income_annum": int(applicant_data['income_annum'] * CONVERSION_RATE),
        "loan_amount": int(applicant_data['loan_amount'] * CONVERSION_RATE),
        "loan_term": applicant_data['loan_term'],
        "no_of_dependents": applicant_data['no_of_dependents'],
        "self_employed": "Yes" if applicant_data['self_employed'] == "Self-Employed" else "No",
        "education": applicant_data['education'],
        "residential_assets_value": int((applicant_data['total_assets'] * ASSET_ALLOCATION['residential_assets_value']) * CONVERSION_RATE),
        "commercial_assets_value": int((applicant_data['total_assets'] * ASSET_ALLOCATION['commercial_assets_value']) * CONVERSION_RATE),
        "luxury_assets_value": int((applicant_data['total_assets'] * ASSET_ALLOCATION['luxury_assets_value']) * CONVERSION_RATE),
        "bank_asset_value": int((applicant_data['total_assets'] * ASSET_ALLOCATION['bank_asset_value']) * CONVERSION_RATE),
        "total_assets": int(applicant_data['total_assets'] * CONVERSION_RATE)
    }
    
    return model_input_data

def calculate_model_probability(model_input):
    """Calculate approval probability using the ML model"""
    df_input = pd.DataFrame([model_input])
    X_input = df_input.drop(columns=['total_assets'])
    X_input = pd.get_dummies(X_input)
    X_input = X_input.reindex(columns=feature_columns, fill_value=0)
    
    approval_probability = model.predict_proba(X_input)[0, 1] * 100
    prediction = model.predict(X_input)[0]
    
    return approval_probability, prediction, X_input

def generate_shap_explanation(X_input):
    """Generate SHAP-based explanation for the decision"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    if len(shap_values.shape) > 1:
        feature_importance = np.abs(shap_values).mean(0)
        importance_values = feature_importance[0]
    else:
        importance_values = np.abs(shap_values)
    
    feature_df = pd.DataFrame({
        'Feature': feature_columns,
        'Impact': importance_values
    }).sort_values('Impact', ascending=False).head(10)
    
    return feature_df

def generate_credit_recommendations(approval_probability, credit_score, loan_to_income_ratio):
    """Generate credit management recommendations"""
    recommendations = []
    
    # Credit Score Analysis
    if credit_score >= 800:
        recommendations.append({
            "category": "Credit Profile",
            "title": "Excellent Credit Standing",
            "actions": [
                "Maintain credit utilization below 25%",
                "Continue timely payments on all accounts",
                "Monitor credit report quarterly"
            ]
        })
    elif credit_score >= 700:
        recommendations.append({
            "category": "Credit Profile",
            "title": "Good Credit Profile",
            "actions": [
                "Reduce credit card balances above 30% of limits",
                "Avoid new credit applications for 3 months",
                "Consider credit limit increases to improve utilization"
            ]
        })
    else:
        recommendations.append({
            "category": "Credit Profile",
            "title": "Credit Improvement Required",
            "actions": [
                "Obtain full credit report and dispute inaccuracies",
                "Establish 12 months of consistent payment history",
                "Consider secured credit products to rebuild history"
            ]
        })
    
    # Affordability Analysis
    if loan_to_income_ratio <= 30:
        recommendations.append({
            "category": "Affordability",
            "title": "Strong Payment Capacity",
            "actions": [
                "Maintain current debt-to-income ratio",
                "Build emergency fund equal to 6 months of payments"
            ]
        })
    elif loan_to_income_ratio <= 40:
        recommendations.append({
            "category": "Affordability",
            "title": "Manageable Payment Load",
            "actions": [
                "Create detailed monthly budget",
                "Build 3-6 month emergency fund"
            ]
        })
    else:
        recommendations.append({
            "category": "Affordability",
            "title": "High Payment Burden",
            "actions": [
                "Reduce requested loan amount by 10-15%",
                "Extend loan term to reduce monthly obligation",
                "Increase income or reduce other expenses"
            ]
        })
    
    return recommendations

# ---------------------------------------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def create_score_gauge(score):
    """Create professional gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Assessment Score", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#2563eb"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ef4444'},
                {'range': [40, 70], 'color': '#f59e0b'},
                {'range': [70, 100], 'color': '#10b981'}
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

def create_radar_chart(credit_score, loan_to_income, asset_coverage, stability):
    """Create radar chart for financial health metrics"""
    categories = ['Credit Quality', 'Affordability', 'Asset Coverage', 'Stability']
    
    values = [
        min(100, credit_score / 10),
        100 - min(100, loan_to_income),
        min(100, asset_coverage / 2.5),
        min(100, stability)
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
    
    plot_data = shap_data.head(8).sort_values('Impact', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=plot_data['Impact'],
        y=plot_data['Feature'].str.replace('_', ' ').str.title(),
        orientation='h',
        marker_color='#2563eb'
    ))
    
    fig.update_layout(
        title='Feature Impact Analysis',
        xaxis_title='SHAP Value',
        yaxis_title='Feature',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#1f2937'),
        height=350,
        margin=dict(l=120, r=20, t=40, b=20)
    )
    
    return fig

# ---------------------------------------------------------
# WORD DOCUMENT REPORT
# ---------------------------------------------------------
def create_word_document(applicant_data, results, shap_data, charts):
    """Generate professional Word document report"""
    doc = Document()
    
    # Title
    title = doc.add_heading('Credit Risk Assessment Report', 0)
    
    # Date and reference
    doc.add_paragraph(f'Generated: {datetime.now().strftime("%d %B %Y at %H:%M")}')
    doc.add_paragraph()
    
    # Applicant Information
    doc.add_heading('Applicant Information', 1)
    doc.add_paragraph(f"Credit Score: {applicant_data['credit_score']}")
    doc.add_paragraph(f"Annual Income: £{applicant_data['income_annum']:,}")
    doc.add_paragraph(f"Loan Amount: £{applicant_data['loan_amount']:,}")
    doc.add_paragraph(f"Loan Term: {applicant_data['loan_term']} years")
    
    # Assessment Results
    doc.add_heading('Assessment Results', 1)
    doc.add_paragraph(f"Approval Probability: {results['approval_probability']:.1f}%")
    doc.add_paragraph(f"Risk Category: {results['status']}")
    doc.add_paragraph()
    
    # Charts
    doc.add_heading('Risk Analysis', 1)
    
    # Save and add charts as images
    for chart_name, chart in charts.items():
        if chart:
            chart_path = f"{chart_name}.png"
            chart.write_image(chart_path)
            doc.add_picture(chart_path, width=Inches(5))
            doc.add_paragraph()
    
    # Recommendations
    doc.add_heading('Credit Management Recommendations', 1)
    for rec in results['recommendations']:
        doc.add_heading(rec['title'], 2)
        for action in rec['actions']:
            doc.add_paragraph(f"• {action}", style='List Bullet')
    
    # Footer with disclaimer
    doc.add_page_break()
    doc.add_heading('Disclaimer', 1)
    disclaimer = """
This credit risk assessment is generated by automated systems using machine learning models. 
The assessment is based on statistical patterns and historical data. 

This report does not constitute a guarantee of credit approval or denial. Final credit decisions 
are made by authorized personnel considering all relevant factors.

The methodology and models are proprietary. Reverse engineering or unauthorized use is prohibited.
"""
    doc.add_paragraph(disclaimer)
    doc.add_paragraph()
    doc.add_paragraph("© 2026 Smart Solution to tough data. All rights reserved.")
    
    # Save to bytes
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    return buffer

# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
def main():
    # Professional Header
    st.markdown('<h1 class="main-title">Credit Risk Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("### Machine Learning Based Credit Evaluation")
    
    # Status indicator
    st.success("Machine Learning Model Loaded Successfully")
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### Applicant Information")
        
        credit_score = st.number_input(
            "Credit Score",
            min_value=0,
            max_value=999,
            value=750,
            step=10
        )
        
        income_annum = st.number_input(
            "Annual Income (£)",
            min_value=15000,
            value=35000,
            step=5000
        )
        
        loan_amount = st.number_input(
            "Loan Amount (£)",
            min_value=5000,
            value=25000,
            step=5000
        )
        
        loan_term = st.slider(
            "Loan Term (Years)",
            min_value=1,
            max_value=30,
            value=5
        )
        
        no_of_dependents = st.selectbox(
            "Number of Dependents",
            options=[0, 1, 2, 3, 4, 5],
            index=1
        )
        
        self_employed = st.selectbox(
            "Employment Status",
            ["Employed", "Self-Employed"]
        )
        
        education = st.selectbox(
            "Education Level",
            ["Graduate", "Non-Graduate"]
        )
        
        total_assets = st.number_input(
            "Total Asset Value (£)",
            min_value=0,
            value=50000,
            step=10000
        )
        
        analyze_button = st.button(
            "Analyze Credit Risk",
            type="primary",
            use_container_width=True
        )
    
    # Main content
    if analyze_button:
        # Prepare data
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
        
        model_input = prepare_model_input(applicant_data)
        
        try:
            # Get model prediction
            approval_probability, prediction, X_input = calculate_model_probability(model_input)
            
            # Generate SHAP explanation
            shap_data = generate_shap_explanation(X_input)
            
            # Generate recommendations
            loan_to_income = (loan_amount / income_annum) * 100
            recommendations = generate_credit_recommendations(
                approval_probability, 
                credit_score, 
                loan_to_income
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
                'approval_probability': approval_probability,
                'prediction': prediction,
                'status': status,
                'status_class': status_class,
                'recommendations': recommendations,
                'applicant_data': applicant_data,
                'shap_data': shap_data
            }
            
        except Exception as e:
            st.error(f"Model analysis failed: {str(e)}")
            return
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        
        # Results Header
        st.markdown('<h2 class="section-title">Risk Assessment Results</h2>', unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Approval Probability", f"{results['approval_probability']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            model_decision = "Approve" if results['prediction'] == 1 else "Review"
            st.metric("Model Decision", model_decision)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="{results["status_class"]}">{results["status"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create charts
        gauge_chart = create_score_gauge(results['approval_probability'])
        
        # Calculate radar chart values
        loan_to_income = (results['applicant_data']['loan_amount'] / results['applicant_data']['income_annum']) * 100
        asset_coverage = (results['applicant_data']['total_assets'] / results['applicant_data']['loan_amount']) * 100
        stability = 60  # Simplified stability score
        
        radar_chart = create_radar_chart(
            results['applicant_data']['credit_score'],
            loan_to_income,
            asset_coverage,
            stability
        )
        
        shap_chart = create_shap_bar_chart(results.get('shap_data'))
        
        # Display charts
        st.markdown('<h3 class="section-title">Risk Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(gauge_chart, use_container_width=True)
        with col2:
            st.plotly_chart(radar_chart, use_container_width=True)
        
        if shap_chart:
            st.plotly_chart(shap_chart, use_container_width=True)
        
        # Recommendations
        st.markdown('<h3 class="section-title">Credit Management Recommendations</h3>', unsafe_allow_html=True)
        
        for rec in results['recommendations']:
            with st.expander(rec['title']):
                for action in rec['actions']:
                    st.write(f"• {action}")
        
        # Report Generation
        st.markdown('<h3 class="section-title">Generate Report</h3>', unsafe_allow_html=True)
        
        charts_dict = {
            'gauge': gauge_chart,
            'radar': radar_chart,
            'shap': shap_chart
        }
        
        try:
            word_buffer = create_word_document(
                results['applicant_data'],
                results,
                results.get('shap_data'),
                charts_dict
            )
            
            current_date = datetime.now().strftime("%Y%m%d")
            b64 = base64.b64encode(word_buffer.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="Credit_Assessment_{current_date}.docx" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: 600;">Download Full Report (.docx)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Report generation error: {str(e)}")
    
    else:
        # Instructions
        st.markdown("""
        ### System Overview
        
        This system uses machine learning to assess credit risk based on:
        
        • Credit Score (0-999 scale)
        • Annual Income
        • Loan Amount Requested
        • Loan Term
        • Employment Status
        • Asset Value
        
        ### How to Use
        
        1. Enter applicant information in the sidebar
        2. Click 'Analyze Credit Risk'
        3. Review the assessment results
        4. Download detailed report
        """)
    
    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.8rem; line-height: 1.4;">
    <p><strong>Disclaimer:</strong> This credit risk assessment is generated by automated systems using machine learning models. 
    The assessment is based on statistical patterns and historical data. It does not guarantee credit approval or denial. 
    Final decisions require human review and consideration of all relevant factors.</p>
    <p style="margin-top: 0.5rem;">© 2026 Smart Solution to tough data. Proprietary and Confidential.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
