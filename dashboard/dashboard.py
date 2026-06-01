import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
pipeline = joblib.load(os.path.join(model_path, 'fraud_pipeline.pkl'))
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']

num_features = ['amount', 'account_balance', 'is_new_merchant', 'distance_from_home',
                'is_international', 'is_card_present', 'txns_last_24h', 'round_amount_flag',
                'amount_to_balance_ratio', 'hour_of_day', 'day_of_week']
cat_features = ['merchant_category']
categories = ['dining', 'entertainment', 'gas', 'grocery', 'healthcare', 'retail', 'travel', 'utilities']

st.set_page_config(page_title='Fraud Detection', page_icon='\U0001f6a8', layout='wide')

st.markdown("""
<style>
.app-header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 2rem; }
.app-header h1 { color: white; margin: 0; font-size: 1.8rem; font-weight: 600; }
.app-header p { color: #94a3b8; margin: 0.25rem 0 0 0; font-size: 0.9rem; }
.card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
.card h3 { color: #e2e8f0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 1rem 0; }
.metric-big { font-size: 2.5rem; font-weight: 700; text-align: center; padding: 0.5rem 0; }
.metric-label { text-align: center; color: #94a3b8; font-size: 0.85rem; }
.badge-fraud { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; padding: 0.5rem 1.5rem; border-radius: 8px; font-weight: 700; font-size: 1.2rem; text-align: center; }
.badge-clear { background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 0.5rem 1.5rem; border-radius: 8px; font-weight: 700; font-size: 1.2rem; text-align: center; }
.flags-container { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0; }
.flag-on { background: #dc2626; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
.flag-off { background: #334155; color: #94a3b8; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header"><h1>\U0001f6a8 Fraud Detection System</h1><p>Real-time transaction scoring with XGBoost + SHAP explainability</p></div>', unsafe_allow_html=True)

input_col, result_col = st.columns([1, 1.2])

with input_col:
    st.markdown('<div class="card"><h3>Transaction Details</h3>', unsafe_allow_html=True)

    amount = st.number_input('Transaction Amount ($)', min_value=0.0, max_value=50000.0, value=100.0, step=10.0)
    account_balance = st.number_input('Account Balance ($)', min_value=0.0, max_value=500000.0, value=5000.0, step=100.0)
    merchant_category = st.selectbox('Merchant Category', categories)
    distance_from_home = st.slider('Distance from Home (km)', 0.0, 500.0, 10.0)
    hour_of_day = st.slider('Hour of Day', 0, 23, 14)
    day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    st.markdown('<h3>Flags</h3>', unsafe_allow_html=True)
    flag_cols = st.columns(3)
    with flag_cols[0]:
        is_international = st.checkbox('International', value=False)
        is_new_merchant = st.checkbox('New Merchant', value=False)
    with flag_cols[1]:
        is_card_present = st.checkbox('Card Present', value=True)
        round_amount_flag = st.checkbox('Round Amount', value=False)
    with flag_cols[2]:
        txns_last_24h = st.number_input('Txns (24h)', min_value=0, max_value=100, value=2)

    ratio = amount / account_balance if account_balance > 0 else 1.0

    st.markdown('<h3>Decision Threshold</h3>', unsafe_allow_html=True)
    threshold = st.slider('Threshold', 0.05, 0.95, 0.25, 0.05,
        help='Transactions above this fraud probability are flagged. Lower = catch more fraud, more false positives.')

    st.markdown('</div>', unsafe_allow_html=True)

day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

input_df = pd.DataFrame([[
    amount, account_balance, int(is_new_merchant), distance_from_home,
    int(is_international), int(is_card_present), txns_last_24h, int(round_amount_flag),
    ratio, hour_of_day, day_map[day_of_week], merchant_category
]], columns=num_features + cat_features)

proba = float(pipeline.predict_proba(input_df)[0, 1])
decision = 'FRAUD' if proba >= threshold else 'CLEAR'

with result_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Transaction Assessment</h3>', unsafe_allow_html=True)

    metric_row = st.columns([1, 1, 1])
    with metric_row[0]:
        pct = int(proba * 100)
        st.markdown(f'<div class="metric-big">{pct}%</div><div class="metric-label">Fraud Probability</div>', unsafe_allow_html=True)
        st.progress(proba)
    with metric_row[1]:
        badge = 'badge-fraud' if decision == 'FRAUD' else 'badge-clear'
        st.markdown(f'<div style="margin-top:0.5rem"><div class="{badge}">{decision}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Decision at {threshold:.0%} threshold</div>', unsafe_allow_html=True)
    with metric_row[2]:
        margin = proba - threshold if decision == 'FRAUD' else threshold - proba
        st.markdown(f'<div class="metric-big" style="font-size:1.8rem">{margin:.1%}</div><div class="metric-label">Margin from threshold</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Active Risk Signals</h3>', unsafe_allow_html=True)
    flags = []
    if is_international: flags.append('\U0001f30d International')
    if is_new_merchant: flags.append('\U0001f195 New Merchant')
    if not is_card_present: flags.append('\U0001f50e Card Not Present')
    if round_amount_flag: flags.append('\U0001f4b0 Round Amount')
    if hour_of_day < 6 or hour_of_day > 23: flags.append('\U0001f319 Odd Hour')
    if ratio > 0.5: flags.append('\U000026a1 High Amt/Bal Ratio')
    if txns_last_24h > 10: flags.append('\U0001f4ca High Txns (24h)')

    if flags:
        st.markdown('<div class="flags-container">' + ''.join(f'<span class="flag-on">{f}</span>' for f in flags) + '</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#94a3b8">No significant risk signals detected.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>SHAP Explanation — Why this score?</h3>', unsafe_allow_html=True)

    input_processed = preprocessor.transform(input_df)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_processed)
    feature_names = preprocessor.get_feature_names_out()

    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_processed[0],
        feature_names=feature_names
    )
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(shap_html, height=200, scrolling=True)

    st.caption('Red pushes toward FRAUD. Blue pushes toward CLEAR.')
    st.markdown('</div>', unsafe_allow_html=True)
