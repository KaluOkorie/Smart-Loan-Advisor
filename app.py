"""
Smart Credit Risk Advisor
Professional Loan Eligibility Assessment System with Report Export
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import time
import shap
import base64
from io import BytesIO
import json
import tempfile
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
import plotly.io as pio

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        color: #1a237e;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3f51b5;
    }
    
    .report-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a237e;
        margin: 1.5rem 0 1rem 0;
    }
    
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #283593;
        padding-left: 0.5rem;
        border-left: 4px solid #3f51b5;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3f2fd 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .success-container {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(76,175,80,0.1);
    }
    
    .warning-container {
        background: linear-gradient(135deg, #fff3e0 0%, #ffecb3 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(255,152,0,0.1);
    }
    
    .info-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(33,150,243,0.1);
    }
    
    .stMetric {
        min-height: 100px;
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stMetric > div {
        min-height: 85px;
    }
    
    .stMetric label {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #455a64 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    
    .disclaimer-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1.5rem 0;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    
    .report-section {
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .feature-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #3f51b5;
        margin: 0.75rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .feature-box:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateX(4px);
    }
    
    .export-buttons {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    
    .export-btn {
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        text-align: center;
        min-width: 180px;
    }
    
    .export-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #3f51b5 0%, #283593 100%);
        color: white;
        border: none;
    }
    
    .btn-secondary {
        background: linear-gradient(135deg, #607d8b 0%, #455a64 100%);
        color: white;
        border: none;
    }
    
    .btn-success {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
        color: white;
        border: none;
    }
    
    .chart-container {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .timeline {
        position: relative;
        padding-left: 2rem;
        margin: 1.5rem 0;
    }
    
    .timeline-item {
        position: relative;
        margin-bottom: 2rem;
        padding-left: 2rem;
    }
    
    .timeline-item:before {
        content: '';
        position: absolute;
        left: -8px;
        top: 0;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #3f51b5;
    }
    
    .timeline-item:after {
        content: '';
        position: absolute;
        left: 0;
        top: 16px;
        width: 2px;
        height: calc(100% + 1rem);
        background: #e0e0e0;
    }
    
    .timeline-item:last-child:after {
        display: none;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .risk-low {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .risk-medium {
        background: #fff3e0;
        color: #ef6c00;
    }
    
    .risk-high {
        background: #ffebee;
        color: #c62828;
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
if 'charts' not in st.session_state:
    st.session_state.charts = {}
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# ---------------------------------------------------------
# UK FINANCIAL STANDARDS
# ---------------------------------------------------------
CONVERSION_RATE = 100
UK_AVERAGE_SALARY = 35000
UK_AVERAGE_HOUSE_PRICE = 250000
UK_ASSET_MULTIPLIER = 5

# ---------------------------------------------------------
# REPORT GENERATION FUNCTIONS
# ---------------------------------------------------------
def generate_html_report(results):
    """Generate a comprehensive HTML report"""
    
    # Prepare data
    applicant_data = results['applicant_data']
    features = results['features']
    
    # Generate charts as static images (base64)
    gauge_chart = create_score_gauge(results['matching_score'])
    radar_chart = create_feature_radar(features)
    
    # Convert charts to HTML divs
    gauge_div = pio.to_html(gauge_chart, full_html=False, include_plotlyjs='cdn')
    radar_div = pio.to_html(radar_chart, full_html=False, include_plotlyjs='cdn')
    
    # SHAP chart if available
    shap_div = ""
    if results.get('shap_data') is not None and not results['shap_data'].empty:
        shap_chart = create_shap_chart(results['shap_data'])
        if shap_chart:
            shap_div = pio.to_html(shap_chart, full_html=False, include_plotlyjs='cdn')
    
    # Determine risk level
    matching_score = results['matching_score']
    if matching_score >= 85:
        risk_level = "Low"
        risk_class = "risk-low"
        risk_description = "Excellent credit profile with minimal risk"
    elif matching_score >= 70:
        risk_level = "Low-Medium"
        risk_class = "risk-low"
        risk_description = "Strong credit position with manageable risk"
    elif matching_score >= 55:
        risk_level = "Medium"
        risk_class = "risk-medium"
        risk_description = "Moderate risk requiring enhanced documentation"
    elif matching_score >= 40:
        risk_level = "Medium-High"
        risk_class = "risk-medium"
        risk_description = "Elevated risk requiring careful review"
    else:
        risk_level = "High"
        risk_class = "risk-high"
        risk_description = "High risk profile requiring significant improvement"
    
    # Format dates
    report_date = datetime.now().strftime('%d %B %Y')
    report_time = datetime.now().strftime('%H:%M')
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Credit Risk Assessment Report - {report_date}</title>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f8f9fa;
            }}
            
            .report-container {{
                background: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.1);
                margin-bottom: 40px;
            }}
            
            .header {{
                text-align: center;
                padding-bottom: 30px;
                border-bottom: 3px solid #3f51b5;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                color: #1a237e;
                margin-bottom: 10px;
                font-size: 2.5rem;
            }}
            
            .header .subtitle {{
                color: #5c6bc0;
                font-size: 1.2rem;
                margin-bottom: 20px;
            }}
            
            .metadata {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
                padding: 20px;
                background: #f5f7fa;
                border-radius: 8px;
            }}
            
            .metadata-item {{
                flex: 1;
                min-width: 200px;
            }}
            
            .metadata-label {{
                font-weight: 600;
                color: #546e7a;
                margin-bottom: 5px;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .metadata-value {{
                font-size: 1.1rem;
                color: #263238;
            }}
            
            .section {{
                margin-bottom: 40px;
                padding: 25px;
                background: #fff;
                border-radius: 10px;
                border-left: 4px solid #3f51b5;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            
            .section-title {{
                color: #283593;
                margin-bottom: 20px;
                font-size: 1.5rem;
                padding-bottom: 10px;
                border-bottom: 2px solid #e8eaf6;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .metric-card {{
                background: linear-gradient(135deg, #f5f7fa 0%, #e3f2fd 100%);
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #e0e0e0;
                transition: transform 0.3s ease;
            }}
            
            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            }}
            
            .metric-value {{
                font-size: 2.5rem;
                font-weight: 700;
                color: #1a237e;
                margin: 10px 0;
            }}
            
            .metric-label {{
                color: #546e7a;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .chart-container {{
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }}
            
            .recommendation {{
                margin: 15px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #4caf50;
            }}
            
            .warning {{
                margin: 15px 0;
                padding: 20px;
                background: #fff3e0;
                border-radius: 8px;
                border-left: 4px solid #ff9800;
            }}
            
            .risk-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: 600;
                margin: 5px;
            }}
            
            .risk-low {{ background: #e8f5e9; color: #2e7d32; }}
            .risk-medium {{ background: #fff3e0; color: #ef6c00; }}
            .risk-high {{ background: #ffebee; color: #c62828; }}
            
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            
            .table th, .table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }}
            
            .table th {{
                background: #f5f7fa;
                color: #546e7a;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.85rem;
                letter-spacing: 0.5px;
            }}
            
            .table tr:hover {{
                background: #f8f9fa;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
                color: #78909c;
                font-size: 0.9rem;
            }}
            
            @media print {{
                body {{
                    background: white;
                }}
                
                .report-container {{
                    box-shadow: none;
                    padding: 0;
                }}
                
                .no-print {{
                    display: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="header">
                <h1>Credit Risk Assessment Report</h1>
                <div class="subtitle">Professional Risk Evaluation & Recommendation</div>
                <div class="metadata">
                    <div class="metadata-item">
                        <div class="metadata-label">Report Date</div>
                        <div class="metadata-value">{report_date}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Report Time</div>
                        <div class="metadata-value">{report_time}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Reference ID</div>
                        <div class="metadata-value">CR-{datetime.now().strftime('%Y%m%d-%H%M%S')}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Executive Summary</h2>
                <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
                    <div style="flex: 1;">
                        <div class="metric-value">{results['matching_score']}/100</div>
                        <div class="metadata-label">Overall Matching Score</div>
                    </div>
                    <div style="flex: 2;">
                        <span class="risk-badge {risk_class}">{risk_level}</span>
                        <p>{risk_description}</p>
                    </div>
                </div>
                <p><strong>Assessment Status:</strong> {results['status']}</p>
                <p><strong>Model Prediction:</strong> {'Approve' if results['prediction'] == 1 else 'Review Required'}</p>
                <p><strong>Approval Probability:</strong> {results['approval_probability']:.1f}%</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">Applicant Information</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Credit Score</div>
                        <div class="metric-value">{applicant_data['credit_score']}</div>
                        <div>{features['credit_category']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Annual Income</div>
                        <div class="metric-value">£{applicant_data['income_annum']:,}</div>
                        <div>{applicant_data['self_employed']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Loan Request</div>
                        <div class="metric-value">£{applicant_data['loan_amount']:,}</div>
                        <div>{applicant_data['loan_term']} years</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Assets</div>
                        <div class="metric-value">£{applicant_data['total_assets']:,}</div>
                        <div>{results['features']['asset_coverage']:.1f}% coverage</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Key Metrics & Visualizations</h2>
                <div class="chart-container">
                    {gauge_div}
                </div>
                <div class="chart-container">
                    {radar_div}
                </div>
    """
    
    # Add SHAP chart if available
    if shap_div:
        html_content += f"""
                <div class="chart-container">
                    {shap_div}
                </div>
        """
    
    # Add financial metrics table
    html_content += """
                <h3>Financial Health Metrics</h3>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Status</th>
                            <th>Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Define metrics
    metrics_data = [
        ("Monthly Payment Ratio", f"{features['monthly_payment_ratio']:.1f}%", 
         "Good" if features['monthly_payment_ratio'] <= 35 else "Monitor", 
         "≤35% recommended"),
        ("Asset Coverage", f"{features['asset_coverage']:.1f}%", 
         "Strong" if features['asset_coverage'] >= 125 else "Weak", 
         "≥125% recommended"),
        ("Credit Category", features['credit_category'], 
         "Good" if features['credit_category'] in ['Good', 'Very Good', 'Excellent'] else "Needs Improvement", 
         "Maintain or improve"),
        ("Loan to Income Ratio", f"{(applicant_data['loan_amount'] / applicant_data['income_annum']) * 100:.1f}%", 
         "Acceptable" if (applicant_data['loan_amount'] / applicant_data['income_annum']) <= 4 else "High", 
         "≤400% recommended"),
        ("Stability Score", f"{features['stability_score']}/45", 
         "Stable" if features['stability_score'] >= 30 else "Volatile", 
         "≥30 recommended")
    ]
    
    for metric, value, status, recommendation in metrics_data:
        status_class = "risk-low" if status in ["Good", "Strong", "Acceptable", "Stable"] else "risk-high"
        html_content += f"""
                        <tr>
                            <td><strong>{metric}</strong></td>
                            <td>{value}</td>
                            <td><span class="risk-badge {status_class}">{status}</span></td>
                            <td>{recommendation}</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">Recommendations & Action Items</h2>
    """
    
    # Add recommendations
    for i, rec in enumerate(results['recommendations'][:6], 1):
        box_class = "recommendation" if rec['box_type'] == "success" else "warning"
        html_content += f"""
                <div class="{box_class}">
                    <h3>{i}. {rec['title']}</h3>
                    <p>{rec['message']}</p>
                    <ul>
        """
        for action in rec['actions']:
            html_content += f"<li>{action}</li>"
        html_content += """
                    </ul>
                </div>
        """
    
    # Add next steps
    html_content += """
                <h3>Next Steps</h3>
                <div class="recommendation">
                    <ol>
                        <li><strong>Immediate (0-7 days):</strong> Review all recommendations and gather required documentation</li>
                        <li><strong>Short-term (7-30 days):</strong> Implement credit improvement strategies if applicable</li>
                        <li><strong>Medium-term (1-3 months):</strong> Monitor credit score and financial ratios</li>
                        <li><strong>Long-term (3-6 months):</strong> Reassess credit profile and consider reapplication if needed</li>
                    </ol>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Disclaimer</h2>
                <div style="background: #fff3e0; padding: 20px; border-radius: 8px; border-left: 4px solid #ff9800;">
                    <p><strong>Important Notice:</strong> This credit risk assessment report is generated by machine learning models and provides preliminary evaluation only. Final credit decisions are made by individual lending institutions based on comprehensive underwriting criteria.</p>
                    <p>This report does not constitute a loan offer or guarantee of credit approval. All financial decisions should be made in consultation with qualified financial advisors. Model accuracy may vary based on input data quality and completeness.</p>
                    <p style="margin-top: 20px; font-size: 0.9rem; color: #666;">
                        Report generated by: Credit Risk Assessment System<br>
                        © 2024 Smart Solution to tough data. All rights reserved.
                    </p>
                </div>
            </div>
            
            <div class="footer">
                <p>Confidential Report - For Internal Use Only</p>
                <p class="no-print">Page generated: {report_date} at {report_time} | Report ID: CR-{datetime.now().strftime('%Y%m%d-%H%M%S')}</p>
            </div>
        </div>
        
        <script>
            // Add print functionality
            function printReport() {{
                window.print();
            }}
            
            // Add download functionality
            function downloadReport() {{
                const element = document.querySelector('.report-container');
                const opt = {{
                    margin:       1,
                    filename:     'credit_risk_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                    image:        {{ type: 'jpeg', quality: 0.98 }},
                    html2canvas:  {{ scale: 2 }},
                    jsPDF:        {{ unit: 'in', format: 'letter', orientation: 'portrait' }}
                }};
                
                // Using html2pdf library (would need to be included)
                // html2pdf().set(opt).from(element).save();
                
                // Fallback for printing
                printReport();
            }}
            
            // Initialize interactive elements
            document.addEventListener('DOMContentLoaded', function() {{
                // Add print button dynamically
                const header = document.querySelector('.header');
                const printBtn = document.createElement('button');
                printBtn.innerHTML = '🖨️ Print Report';
                printBtn.style.cssText = `
                    background: linear-gradient(135deg, #3f51b5 0%, #283593 100%);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: 600;
                    margin-top: 10px;
                `;
                printBtn.onclick = printReport;
                header.appendChild(printBtn);
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

def generate_pdf_report(results):
    """Generate a PDF report using FPDF"""
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Credit Risk Assessment Report", ln=True, align='C')
    pdf.ln(10)
    
    # Report date
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Executive Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Matching Score: {results['matching_score']}/100", ln=True)
    pdf.cell(200, 10, txt=f"Status: {results['status']}", ln=True)
    pdf.cell(200, 10, txt=f"Approval Probability: {results['approval_probability']:.1f}%", ln=True)
    pdf.ln(10)
    
    # Applicant Information
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Applicant Information", ln=True)
    pdf.set_font("Arial", size=12)
    
    app_data = results['applicant_data']
    pdf.cell(200, 10, txt=f"Credit Score: {app_data['credit_score']}", ln=True)
    pdf.cell(200, 10, txt=f"Annual Income: £{app_data['income_annum']:,}", ln=True)
    pdf.cell(200, 10, txt=f"Loan Amount: £{app_data['loan_amount']:,}", ln=True)
    pdf.cell(200, 10, txt=f"Loan Term: {app_data['loan_term']} years", ln=True)
    pdf.cell(200, 10, txt=f"Employment: {app_data['self_employed']}", ln=True)
    pdf.cell(200, 10, txt=f"Total Assets: £{app_data['total_assets']:,}", ln=True)
    pdf.ln(10)
    
    # Financial Metrics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Financial Health Metrics", ln=True)
    pdf.set_font("Arial", size=12)
    
    features = results['features']
    pdf.cell(200, 10, txt=f"Monthly Payment Ratio: {features['monthly_payment_ratio']:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Asset Coverage: {features['asset_coverage']:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Credit Category: {features['credit_category']}", ln=True)
    pdf.cell(200, 10, txt=f"Stability Score: {features['stability_score']}/45", ln=True)
    pdf.ln(10)
    
    # Key Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Key Recommendations", ln=True)
    pdf.set_font("Arial", size=12)
    
    for i, rec in enumerate(results['recommendations'][:5], 1):
        pdf.multi_cell(0, 10, txt=f"{i}. {rec['title']}: {rec['message']}")
        pdf.ln(5)
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="Disclaimer: This report is generated by machine learning models for preliminary assessment only. Final credit decisions are made by individual lending institutions.")
    
    # Save to buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    
    return buffer

# ---------------------------------------------------------
# EXPORT FUNCTIONS
# ---------------------------------------------------------
def create_download_link(content, filename, mime_type):
    """Create a download link for various file types"""
    b64 = base64.b64encode(content).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{filename}</a>'

def export_to_html(results):
    """Export results to HTML file"""
    html_content = generate_html_report(results)
    return html_content.encode()

def export_to_pdf(results):
    """Export results to PDF file"""
    pdf_buffer = generate_pdf_report(results)
    return pdf_buffer.getvalue()

def export_to_json(results):
    """Export results to JSON file"""
    # Create a clean copy for export
    export_data = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "report_id": f"CR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "system_version": "1.0"
        },
        "assessment_results": {
            "matching_score": results['matching_score'],
            "approval_probability": results['approval_probability'],
            "prediction": "Approve" if results['prediction'] == 1 else "Review",
            "status": results['status'],
            "risk_level": "Low" if results['matching_score'] >= 70 else "Medium" if results['matching_score'] >= 50 else "High"
        },
        "applicant_data": results['applicant_data'],
        "financial_metrics": results['features'],
        "key_recommendations": [
            {
                "title": rec['title'],
                "priority": rec['box_type'],
                "actions": rec['actions']
            }
            for rec in results['recommendations'][:5]
        ],
        "assessment_guidelines": {
            "credit_score_interpretation": {
                "excellent": "≥ 900",
                "very_good": "800-899",
                "good": "700-799",
                "fair": "600-699",
                "needs_improvement": "< 600"
            },
            "payment_affordability_threshold": "≤ 35% of monthly income",
            "asset_coverage_target": "≥ 125% of loan amount"
        }
    }
    
    return json.dumps(export_data, indent=2).encode()

# ---------------------------------------------------------
# ENHANCED FEATURE ENGINEERING FUNCTIONS
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
    df_engineered['income_to_uk_average'] = (df_engineered['income_annum'] / UK_AVERAGE_SALARY) * 100
    
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
    
    # Risk indicators
    df_engineered['high_risk_indicator'] = (
        (df_engineered['credit_score'] < 600) |
        (df_engineered['monthly_payment_ratio'] > 50) |
        (df_engineered['asset_coverage'] < 50)
    ).astype(int)
    
    return df_engineered

def calculate_matching_score(row):
    """Calculate matching score (0-100) based on credit profile"""
    score = 0
    
    # 1. Credit Score Component (40%)
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
    
    # 2. Monthly Payment Affordability (25%)
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
    
    # 3. Asset Security (20%)
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
    
    # 4. Stability Factors (15%)
    score += min(row['stability_score'], 15)
    
    # 5. Risk adjustments
    if row['self_employed'] == 'Yes' and row['income_annum'] < 30000:
        score -= 10
    
    if row['no_of_dependents'] > 3:
        score -= 5
    
    if row['loan_term'] > 7 and row['loan_amount'] < 150000:
        score -= 3
    
    return min(max(score, 0), 100)

def generate_detailed_recommendations(score, features, applicant_data, prediction):
    """Generate personalized, actionable credit risk recommendations"""
    recommendations = []
    
    # Credit Score Analysis
    credit_score = features.get('credit_score', 0)
    if credit_score >= 800:
        recommendations.append({
            "title": "Excellent Credit Standing",
            "message": f"Credit score of {credit_score} is in the top tier for lending.",
            "actions": [
                "Qualify for optimal interest rates",
                "Maintain credit utilization below 25%",
                "Continue current payment patterns"
            ],
            "box_type": "success"
        })
    elif credit_score >= 700:
        recommendations.append({
            "title": "Good Credit Profile",
            "message": f"Score of {credit_score} meets standard lending requirements.",
            "actions": [
                "Aim for 750+ to access premium rates",
                "Review credit report quarterly",
                "Limit new credit applications"
            ],
            "box_type": "success"
        })
    elif credit_score >= 600:
        recommendations.append({
            "title": "Credit Improvement Opportunity",
            "message": f"Score of {credit_score} requires attention for optimal terms.",
            "actions": [
                "Register on electoral roll if applicable",
                "Reduce credit card balances below 30%",
                "Establish consistent payment history"
            ],
            "box_type": "warning"
        })
    else:
        recommendations.append({
            "title": "Credit Building Required",
            "message": f"Score of {credit_score} indicates significant risk factors.",
            "actions": [
                "Obtain comprehensive credit reports",
                "Address any delinquencies immediately",
                "Build 12-month positive payment history"
            ],
            "box_type": "warning"
        })
    
    # Payment Affordability Analysis
    payment_ratio = features.get('monthly_payment_ratio', 0)
    monthly_payment = applicant_data['loan_amount'] / (applicant_data['loan_term'] * 12)
    
    if payment_ratio <= 30:
        recommendations.append({
            "title": "Strong Payment Capacity",
            "message": f"Monthly payment of £{monthly_payment:.0f} represents {payment_ratio:.1f}% of income.",
            "actions": [
                "Maintain current debt-to-income ratio",
                "Consider shorter terms for interest savings",
                "Monitor changes in income stability"
            ],
            "box_type": "success"
        })
    elif payment_ratio <= 40:
        recommendations.append({
            "title": "Acceptable Payment Load",
            "message": f"Payment of £{monthly_payment:.0f} is {payment_ratio:.1f}% of income.",
            "actions": [
                "Maintain emergency fund coverage",
                "Review discretionary expenses quarterly",
                "Consider longer terms for flexibility"
            ],
            "box_type": "info"
        })
    else:
        recommendations.append({
            "title": "Elevated Payment Burden",
            "message": f"At {payment_ratio:.1f}% of income, payment capacity is constrained.",
            "actions": [
                f"Reduce requested amount by £{applicant_data['loan_amount'] * 0.1:.0f}",
                "Extend loan term to improve affordability",
                "Increase income sources or reduce expenses"
            ],
            "box_type": "warning"
        })
    
    # Asset Coverage Analysis
    asset_coverage = features.get('asset_coverage', 0)
    if asset_coverage >= 150:
        recommendations.append({
            "title": "Strong Asset Position",
            "message": f"Asset coverage of {asset_coverage:.1f}% provides excellent security.",
            "actions": [
                "Maintain current asset allocation",
                "Document all assets comprehensively",
                "Consider leveraging assets for better rates"
            ],
            "box_type": "success"
        })
    elif asset_coverage >= 100:
        recommendations.append({
            "title": "Adequate Asset Coverage",
            "message": f"Assets cover {asset_coverage:.1f}% of the requested loan amount.",
            "actions": [
                "Monitor asset values regularly",
                "Maintain insurance coverage on key assets",
                "Consider additional collateral for better terms"
            ],
            "box_type": "info"
        })
    else:
        recommendations.append({
            "title": "Limited Asset Security",
            "message": f"Asset coverage of {asset_coverage:.1f}% may be insufficient.",
            "actions": [
                "Increase asset base before applying",
                "Consider additional guarantors",
                "Provide detailed asset documentation"
            ],
            "box_type": "warning"
        })
    
    # Model Prediction Specific Recommendations
    if prediction == 1:
        recommendations.append({
            "title": "Model Assessment: Low Risk",
            "message": "Machine learning model identifies low default probability.",
            "actions": [
                "Proceed with formal application",
                "Document all income sources thoroughly",
                "Prepare 6 months of bank statements"
            ],
            "box_type": "success"
        })
    else:
        recommendations.append({
            "title": "Model Assessment: Elevated Risk",
            "message": "Model indicates elevated default risk based on profile characteristics.",
            "actions": [
                "Review and improve credit profile factors",
                "Consider reducing requested loan amount",
                "Provide additional collateral if available"
            ],
            "box_type": "warning"
        })
    
    # Overall Score-Based Recommendation
    if score >= 85:
        recommendations.append({
            "title": "Premium Credit Profile",
            "message": f"Matching score of {score}/100 indicates excellent creditworthiness.",
            "actions": [
                "Apply to multiple lenders for competitive terms",
                "Expect accelerated processing timeline",
                "Document all assets comprehensively"
            ],
            "box_type": "success"
        })
    elif score >= 70:
        recommendations.append({
            "title": "Strong Credit Position",
            "message": f"Score of {score}/100 suggests high approval probability.",
            "actions": [
                "Complete application with full documentation",
                "Apply to primary lending institutions",
                "Maintain current financial position"
            ],
            "box_type": "success"
        })
    elif score >= 55:
        recommendations.append({
            "title": "Borderline Credit Profile",
            "message": f"Score of {score}/100 requires enhanced documentation.",
            "actions": [
                "Provide detailed income verification",
                "Include employment stability documentation",
                "Consider additional co-signers if applicable"
            ],
            "box_type": "warning"
        })
    else:
        recommendations.append({
            "title": "Profile Requires Strengthening",
            "message": f"Score of {score}/100 indicates significant improvement needed.",
            "actions": [
                "Focus on credit score improvement for 6-12 months",
                "Increase liquid asset position",
                "Consult with financial advisory services"
            ],
            "box_type": "warning"
        })
    
    return recommendations

def generate_shap_explanation(model, X_input, feature_names):
    """Generate SHAP-based explanation for model decision"""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        
        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get feature importance
        feature_importance = np.abs(shap_values).mean(0)
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': feature_importance[0] if len(feature_importance.shape) > 1 else feature_importance
        }).sort_values('Impact', ascending=False).head(10)
        
        return feature_df
    except Exception as e:
        st.error(f"SHAP explanation error: {str(e)}")
        return None

# ---------------------------------------------------------
# ENHANCED VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def create_score_gauge(score):
    """Create a professional gauge chart for matching score"""
    colors = ['#ef5350', '#ff9800', '#4caf50', '#2e7d32']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Credit Profile Score",
            'font': {'size': 18, 'family': "Arial", 'color': "#1a237e"}
        },
        delta={'reference': 70, 'position': "bottom"},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "#37474f",
                'tickfont': {'size': 11, 'color': "#546e7a"}
            },
            'bar': {'color': "#1a237e", 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#546e7a",
            'steps': [
                {'range': [0, 50], 'color': colors[0]},
                {'range': [50, 70], 'color': colors[1]},
                {'range': [70, 85], 'color': colors[2]},
                {'range': [85, 100], 'color': colors[3]}
            ],
            'threshold': {
                'line': {'color': "#000000", 'width': 3},
                'thickness': 0.85,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#263238', family="Arial"),
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
        annotations=[
            dict(
                text=f"Risk Level: {'Low' if score >= 70 else 'Medium' if score >= 50 else 'High'}",
                x=0.5,
                y=-0.2,
                showarrow=False,
                font=dict(size=14, color="#546e7a")
            )
        ]
    )
    
    return fig

def create_feature_radar(features):
    """Create radar chart for financial health metrics"""
    categories = ['Credit Score', 'Affordability', 'Asset Security', 'Stability', 'Income Level']
    values = [
        min(100, features['credit_score'] / 9.99),
        100 - min(100, features['monthly_payment_ratio']),
        min(100, features['asset_coverage'] / 2.5),
        min(100, features['stability_score'] / 45 * 100),
        min(100, features.get('income_to_uk_average', 100))
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(26, 35, 126, 0.3)',
        line=dict(color='rgb(26, 35, 126)', width=3),
        marker=dict(size=8, color='rgb(26, 35, 126)'),
        name="Current Profile"
    ))
    
    # Add target line
    fig.add_trace(go.Scatterpolar(
        r=[70, 70, 70, 70, 70, 70],
        theta=categories + [categories[0]],
        line=dict(color='rgba(0, 150, 136, 0.5)', width=2, dash='dash'),
        name="Target Minimum"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(96, 125, 139, 0.3)',
                linecolor='#546e7a',
                tickfont=dict(size=10, color="#546e7a")
            ),
            angularaxis=dict(
                gridcolor='rgba(96, 125, 139, 0.3)',
                linecolor='#546e7a',
                rotation=90
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#263238', family="Arial"),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=350,
        margin=dict(l=80, r=80, t=40, b=40)
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
        marker=dict(
            color=plot_data['Impact'],
            colorscale='Blues',
            showscale=False
        ),
        text=plot_data['Impact'].round(4),
        textposition='outside',
        textfont=dict(color='#1a237e', size=10)
    ))
    
    fig.update_layout(
        title={
            'text': 'Top Decision Factors',
            'font': {'size': 16, 'color': "#1a237e", 'family': "Arial"}
        },
        xaxis_title='Impact Magnitude',
        yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#263238', family="Arial"),
        height=350,
        margin=dict(l=150, r=30, t=50, b=30),
        xaxis=dict(
            gridcolor='rgba(96, 125, 139, 0.2)',
            zerolinecolor='rgba(96, 125, 139, 0.3)'
        ),
        yaxis=dict(
            tickfont=dict(size=11)
        )
    )
    
    return fig

# ---------------------------------------------------------
# MAIN APPLICATION WITH EXPORT FUNCTIONALITY
# ---------------------------------------------------------
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Credit Risk Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-container">
    <strong>Professional Risk Evaluation:</strong> This system provides data-driven credit risk assessment using machine learning models. 
    Generate comprehensive reports in HTML, PDF, or JSON format with actionable insights and recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3>📝 Application Information</h3>', unsafe_allow_html=True)
        
        # Personal Information
        st.subheader("Personal Details")
        credit_score = st.slider(
            "Credit Score (UK Scale)",
            min_value=0,
            max_value=999,
            value=750,
            help="UK credit scores range from 0-999"
        )
        
        income_annum = st.number_input(
            "Annual Income (£)",
            min_value=15000,
            value=35000,
            step=5000,
            help="Gross annual income before tax"
        )
        
        loan_amount = st.number_input(
            "Loan Amount (£)",
            min_value=5000,
            value=25000,
            step=5000,
            help="Total borrowing requirement"
        )
        
        loan_term = st.slider(
            "Loan Term (Years)",
            min_value=1,
            max_value=30,
            value=5,
            help="Repayment period in years"
        )
        
        no_of_dependents = st.selectbox(
            "Number of Dependents",
            options=[0, 1, 2, 3, 4, "5 or more"],
            index=1
        )
        
        # Employment & Education
        col1, col2 = st.columns(2)
        with col1:
            self_employed = st.radio(
                "Employment Type",
                ["Employed", "Self-Employed"]
            )
        with col2:
            education = st.radio(
                "Education Level",
                ["Graduate", "Not Graduate"]
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Asset Information
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3>💰 Asset Information</h3>', unsafe_allow_html=True)
        
        total_assets = st.number_input(
            "Total Assets Value (£)",
            min_value=0,
            value=50000,
            step=10000,
            help="Combined value of all assets"
        )
        
        if st.checkbox("View asset allocation model"):
            st.info("""
            **Standard Asset Allocation:**
            - Residential Assets: 50%
            - Commercial Assets: 25%
            - Luxury Assets: 15%
            - Bank Assets: 10%
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Report Export Options
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3>📤 Export Options</h3>', unsafe_allow_html=True)
        
        export_format = st.multiselect(
            "Select export formats",
            ["HTML Report", "PDF Summary", "JSON Data"],
            default=["HTML Report"]
        )
        
        include_charts = st.checkbox("Include interactive charts", value=True)
        include_details = st.checkbox("Include detailed recommendations", value=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_submission
        
        if time_since_last < 3 and st.session_state.results is not None:
            remaining = 3 - time_since_last
            st.warning(f"⏳ Assessment cooling period: {remaining:.1f} seconds")
            calculate_disabled = True
        else:
            calculate_disabled = False
        
        # Calculate Button
        col1, col2 = st.columns(2)
        with col1:
            calculate_btn = st.button(
                "🚀 Execute Risk Assessment",
                type="primary",
                disabled=calculate_disabled,
                use_container_width=True
            )
        with col2:
            if st.session_state.results:
                clear_btn = st.button(
                    "🔄 Clear Results",
                    type="secondary",
                    use_container_width=True
                )
                if clear_btn:
                    st.session_state.results = None
                    st.rerun()
    
    # Main Content Area
    if calculate_btn:
        # Update last submission time
        st.session_state.last_submission = time.time()
        
        # Validate inputs
        validation_errors = []
        if loan_term > 30:
            validation_errors.append("Maximum loan term is 30 years")
        if income_annum <= 0:
            validation_errors.append("Annual income must be positive")
        if not (0 <= credit_score <= 999):
            validation_errors.append("Credit score must be between 0-999")
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ Validation Error: {error}")
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
        
        # Display data
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
            
        except FileNotFoundError as e:
            st.error(f"❌ Model file not found: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Model execution error: {str(e)}")
            st.stop()
        
        # Generate detailed recommendations
        recommendations = generate_detailed_recommendations(
            matching_score, 
            df_features.iloc[0].to_dict(),
            uk_display_data,
            prediction
        )
        
        # Determine status
        if matching_score >= 80:
            status = "Low Risk Profile"
            status_box = "success"
        elif matching_score >= 65:
            status = "Moderate Risk Profile"
            status_box = "success"
        elif matching_score >= 50:
            status = "Elevated Risk Profile"
            status_box = "warning"
        else:
            status = "High Risk Profile"
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
            'shap_data': shap_data,
            'export_formats': export_format,
            'include_charts': include_charts,
            'include_details': include_details
        }
        
        st.session_state.report_generated = True
    
    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        
        # Results Header with Export Buttons
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<h2 class="report-header">📋 Risk Assessment Report</h2>', unsafe_allow_html=True)
        with col2:
            if st.session_state.report_generated:
                st.success("✅ Report Generated Successfully!")
        
        # Export Section
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">📤 Export Report</h3>', unsafe_allow_html=True)
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if "HTML Report" in results['export_formats']:
                html_report = generate_html_report(results)
                b64_html = base64.b64encode(html_report).decode()
                html_filename = f"credit_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                st.markdown(
                    f"""
                    <a href="data:text/html;base64,{b64_html}" download="{html_filename}" 
                       style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #3f51b5 0%, #283593 100%); 
                       color: white; text-decoration: none; border-radius: 6px; font-weight: 600; text-align: center; width: 100%;">
                       📄 Download HTML Report
                    </a>
                    """,
                    unsafe_allow_html=True
                )
        
        with export_col2:
            if "PDF Summary" in results['export_formats']:
                pdf_report = export_to_pdf(results)
                b64_pdf = base64.b64encode(pdf_report).decode()
                pdf_filename = f"credit_risk_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                st.markdown(
                    f"""
                    <a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}" 
                       style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #607d8b 0%, #455a64 100%); 
                       color: white; text-decoration: none; border-radius: 6px; font-weight: 600; text-align: center; width: 100%;">
                       📑 Download PDF Summary
                    </a>
                    """,
                    unsafe_allow_html=True
                )
        
        with export_col3:
            if "JSON Data" in results['export_formats']:
                json_report = export_to_json(results)
                b64_json = base64.b64encode(json_report).decode()
                json_filename = f"credit_risk_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                st.markdown(
                    f"""
                    <a href="data:application/json;base64,{b64_json}" download="{json_filename}" 
                       style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%); 
                       color: white; text-decoration: none; border-radius: 6px; font-weight: 600; text-align: center; width: 100%;">
                       📊 Download JSON Data
                    </a>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key Metrics
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">📈 Key Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Profile Score",
                value=f"{results['matching_score']}/100",
                delta=results['status'],
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="Approval Probability",
                value=f"{results['approval_probability']:.1f}%"
            )
        
        with col3:
            decision = "✅ Approve" if results['prediction'] == 1 else "⚠️ Review"
            st.metric(
                label="Model Decision",
                value=decision
            )
        
        with col4:
            if results['matching_score'] >= 70:
                time_value = "⚡ Standard"
            elif results['matching_score'] >= 50:
                time_value = "⏳ Extended"
            else:
                time_value = "🔄 Manual"
            st.metric(
                label="Processing Time",
                value=time_value
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">📊 Visual Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_score_gauge(results['matching_score']), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_feature_radar(results['features']), use_container_width=True)
        
        # SHAP Explanation Chart
        if results.get('shap_data') is not None and results['include_details']:
            shap_chart = create_shap_chart(results['shap_data'])
            if shap_chart:
                st.plotly_chart(shap_chart, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Financial Health Indicators
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">💼 Financial Health Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            payment_ratio = results['features']['monthly_payment_ratio']
            st.write("**Payment Affordability**")
            progress_value = max(0.0, min(1.0, (35 - min(payment_ratio, 35)) / 35))
            st.progress(
                progress_value,
                text=f"{payment_ratio:.1f}% of income"
            )
            st.caption("🎯 Target: ≤35% of monthly income")
        
        with col2:
            asset_coverage = results['features']['asset_coverage']
            st.write("**Asset Coverage**")
            progress_value = min(1.0, asset_coverage / 250)
            st.progress(
                progress_value,
                text=f"{asset_coverage:.1f}% coverage"
            )
            st.caption("🎯 Ideal: ≥125% of loan amount")
        
        with col3:
            stability = results['features']['stability_score']
            st.write("**Stability Score**")
            progress_value = min(1.0, stability / 60)
            st.progress(
                progress_value,
                text=f"{stability}/60 points"
            )
            st.caption("📊 Employment, education, dependents")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk Recommendations
        if results['include_details']:
            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">🎯 Risk Mitigation Recommendations</h3>', unsafe_allow_html=True)
            
            for i, rec in enumerate(results['recommendations'][:6]):
                if rec['box_type'] == "success":
                    box_class = "success-container"
                elif rec['box_type'] == "warning":
                    box_class = "warning-container"
                else:
                    box_class = "info-container"
                
                with st.expander(f"{i+1}. {rec['title']}", expanded=i<2):
                    st.markdown(f"<div class='{box_class}'>", unsafe_allow_html=True)
                    st.write(f"**📋 {rec['message']}**")
                    st.write("")
                    st.write("**🎯 Action Items:**")
                    for action in rec['actions']:
                        st.write(f"• {action}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Report Preview
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">🔍 Report Preview</h3>', unsafe_allow_html=True)
        
        with st.expander("View Complete Report Summary", expanded=False):
            # Applicant Information
            st.markdown("#### 👤 Applicant Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Credit Score:** {results['applicant_data']['credit_score']}")
                st.markdown(f"**Annual Income:** £{results['applicant_data']['income_annum']:,}")
                st.markdown(f"**Loan Amount:** £{results['applicant_data']['loan_amount']:,}")
            
            with col2:
                st.markdown(f"**Loan Term:** {results['applicant_data']['loan_term']} years")
                st.markdown(f"**Employment:** {results['applicant_data']['self_employed']}")
                st.markdown(f"**Education:** {results['applicant_data']['education']}")
            
            with col3:
                st.markdown(f"**Dependents:** {results['applicant_data']['no_of_dependents']}")
                st.markdown(f"**Total Assets:** £{results['applicant_data']['total_assets']:,}")
                st.markdown(f"**Report Date:** {datetime.now().strftime('%d %B %Y at %H:%M')}")
            
            st.markdown("---")
            
            # Next Steps Timeline
            st.markdown("#### 📅 Recommended Timeline")
            
            timeline_steps = [
                ("Immediate (0-7 days)", "Review recommendations and gather documentation"),
                ("Short-term (7-30 days)", "Implement credit improvement strategies"),
                ("Medium-term (1-3 months)", "Monitor credit score and financial ratios"),
                ("Long-term (3-6 months)", "Reassess profile for reapplication if needed")
            ]
            
            for period, action in timeline_steps:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{period}**")
                with col2:
                    st.markdown(f"• {action}")
            
            st.markdown("---")
            
            # Contact Information
            st.markdown("#### 📞 Additional Resources")
            st.markdown("""
            - **Financial Advisory Services:** Consult with qualified financial advisors
            - **Credit Reference Agencies:** Experian, Equifax, TransUnion
            - **UK Financial Conduct Authority:** [www.fca.org.uk](https://www.fca.org.uk)
            - **Money Advice Service:** [www.moneyadviceservice.org.uk](https://www.moneyadviceservice.org.uk)
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer with Disclaimer
        st.markdown("---")
        st.markdown("""
        <div class="disclaimer-box">
        <p><strong>⚠️ Important Notice:</strong> This credit risk assessment system provides preliminary evaluation based on machine learning models. All assessments are indicative and do not constitute financial advice or credit guarantees. Final lending decisions remain at the discretion of individual financial institutions based on comprehensive underwriting processes.</p>
        <p style="margin-top: 0.5rem; text-align: center; font-weight: 600;">© 2024 Smart Solution to tough data | Professional Credit Risk Assessment System</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Welcome screen with enhanced features
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Enhanced Assessment Methodology
            
            **1. Data Collection & Validation**
            Comprehensive collection of financial and personal information with real-time validation.
            
            **2. Machine Learning Analysis**
            Advanced XGBoost model processes 12+ risk factors using historical lending data.
            
            **3. Risk Scoring & Visualization**
            Generation of 0-100 risk score with interactive visualizations.
            
            **4. SHAP Explanation**
            Transparent explanation of model decisions using Shapley values.
            
            **5. Professional Reporting**
            Multi-format report generation (HTML, PDF, JSON) with actionable insights.
            
            ### 📊 Risk Assessment Framework
            
            • **Credit History**: Analysis of credit score and payment patterns  
            • **Payment Capacity**: Evaluation of debt-to-income ratios  
            • **Asset Security**: Assessment of collateral and liquid assets  
            • **Employment Stability**: Review of income consistency and source  
            • **Demographic Factors**: Consideration of education and dependents  
            
            ### 🚀 System Capabilities
            
            • **Multi-Format Export**: Generate reports in HTML, PDF, and JSON formats  
            • **Interactive Visualizations**: Dynamic charts with detailed tooltips  
            • **Transparent Decisions**: SHAP-based feature importance explanations  
            • **Professional Reporting**: Comprehensive assessment reports with timelines  
            • **Real-time Processing**: Instant assessment with rate limiting  
            • **UK Compliance**: Adherence to UK lending standards and regulations  
            """)
        
        with col2:
            st.markdown("""
            <div class="info-container">
            <h4>🇬🇧 UK Credit Standards</h4>
            <ul style="padding-left: 1.2rem; margin-bottom: 0;">
            <li><strong>Credit Score ≥700</strong>: Preferred lending criteria</li>
            <li><strong>Monthly payments ≤35%</strong>: Recommended affordability threshold</li>
            <li><strong>Asset coverage ≥125%</strong>: Optimal security position</li>
            <li><strong>Employment history ≥2 years</strong>: Stability benchmark</li>
            <li><strong>Debt-to-income ≤40%</strong>: Maximum recommended ratio</li>
            <li><strong>Credit inquiries ≤3</strong>: Annual application limit</li>
            </ul>
            </div>
            
            <div class="success-container" style="margin-top: 1.5rem;">
            <h4>📤 Export Features</h4>
            <p>Generate professional reports in multiple formats:</p>
            <ul style="padding-left: 1.2rem; margin-bottom: 0;">
            <li><strong>HTML Report</strong>: Interactive web report with charts</li>
            <li><strong>PDF Summary</strong>: Printable professional document</li>
            <li><strong>JSON Data</strong>: Structured data for integration</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with System Information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #78909c; font-size: 0.9rem; margin-top: 2rem;">
    <p><strong>Credit Risk Assessment System v2.0</strong> | Professional Credit Risk Evaluation Tool</p>
    <p>🔒 Data Privacy Compliant | 📊 Machine Learning Powered | 📈 Real-time Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
