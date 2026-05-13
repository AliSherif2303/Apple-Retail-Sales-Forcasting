# -*- coding: utf-8 -*-
"""
Apple Sales Intelligence Platform — Multi-Page Entry Point
Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Apple Sales Intelligence Platform",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 40%, #0a0f1a 100%); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar branding ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0;">
        <span style="font-size:2.5rem;">🍎</span>
        <div style="font-size:1.4rem; font-weight:800; color:#818cf8; margin-top:8px;">
            Apple Sales Intelligence
        </div>
        <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
            Multi-Page Analytics Platform
        </div>
    </div>
    <hr style="border-color: rgba(99,102,241,0.3);">
    """, unsafe_allow_html=True)
    st.markdown("👈 **Select a page** from the sidebar above.")

# ─── Home page content ───────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 60px 0 30px 0;">
    <span style="font-size:4rem;">🍎</span>
    <h1 style="color:#818cf8; margin-top:16px; font-size:2.4rem;">
        Apple Sales Intelligence Platform
    </h1>
    <p style="color:#94a3b8; font-size:1.1rem; max-width:600px; margin:16px auto;">
        A comprehensive analytics suite for Apple retail sales forecasting,
        market expansion, and AI-powered data insights.
    </p>
</div>
""", unsafe_allow_html=True)

# Page cards
cols = st.columns(3)
pages_info = [
    ("🏠", "Dashboard", "Interactive EDA with KPIs, trends, and market analysis."),
    ("📈", "Long-Term Forecasting", "Combined AdaBoost & CatBoost 12-month forecast."),
    ("📊", "Short-Term Forecasting", "Pre-computed model predictions by store."),
    ("🗺️", "Market Clustering", "K-Means city segmentation for expansion."),
    ("🎯", "Target Market Validator", "Score & rank new markets for entry."),
    ("🤖", "SQL RAG Agent", "Ask questions in plain English using AI."),
]

for i, (icon, title, desc) in enumerate(pages_info):
    with cols[i % 3]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.06));
            border: 1px solid rgba(99,102,241,0.25); border-radius: 16px;
            padding: 24px; margin-bottom: 16px; min-height: 160px;">
            <div style="font-size:2rem; margin-bottom:8px;">{icon}</div>
            <div style="font-size:1.05rem; font-weight:700; color:#e2e8f0;">{title}</div>
            <div style="font-size:0.82rem; color:#94a3b8; margin-top:8px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<hr style="border-color: rgba(99,102,241,0.3); margin-top:40px;">
<div style="text-align:center; color:#334155; font-size:0.78rem; padding-bottom:12px;">
    Apple Sales Intelligence Platform — Built with Streamlit & Plotly — Data: 2021-2025
</div>
""", unsafe_allow_html=True)
