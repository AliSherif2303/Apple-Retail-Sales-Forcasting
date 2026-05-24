# -*- coding: utf-8 -*-
"""
Apple Sales Intelligence Platform — Multi-Page Entry Point
Run with:  streamlit run app.py
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from theme import CSS, ACCENT

st.set_page_config(
    page_title="Apple Sales Intelligence Platform",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)

# ─── Sidebar branding ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 16px 0;">
        <span style="font-size:2.5rem; filter:drop-shadow(0 0 18px {ACCENT});">🍎</span>
        <div style="font-size:1.4rem; font-weight:800; color:{ACCENT}; margin-top:8px;">
            Apple Sales Intelligence
        </div>
        <div style="font-size:0.8rem; color:#A0A0A5; margin-top:4px;">
            Multi-Page Analytics Platform
        </div>
    </div>
    <hr style="border-color: rgba(0,240,255,0.3);">
    """, unsafe_allow_html=True)
    st.markdown("👈 **Select a page** from the sidebar above.")

# ─── Home page content ───────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding: 60px 0 36px 0;">
    <span style="font-size:4rem; filter: drop-shadow(0 0 28px {ACCENT});">🍎</span>
    <h1 style="color:{ACCENT}; margin-top:16px; font-size:2.4rem; letter-spacing:1px;">
        Apple Sales Intelligence Platform
    </h1>
    <p style="color:#E1E1E6; font-size:1.1rem; max-width:600px; margin:16px auto;">
        A comprehensive analytics suite for Apple retail sales forecasting,
        market expansion, and AI-powered data insights.
    </p>
    <p style="color:#A0A0A5; font-size:0.85rem; margin-top:8px;">
        ✨ Click any card below to navigate directly to that page
    </p>
</div>
""", unsafe_allow_html=True)

# Page cards
pages_info = [
    ("🏠", "Dashboard",               "Interactive EDA with KPIs, trends, and market analysis.",      "/Dashboard"),
    ("📈", "Long-Term Forecasting",   "Live 12-month forecast using CatBoost & AdaBoost models.",     "/Long_Term_Forecasting"),
    ("📊", "Short-Term Forecasting",  "Live 1-month forecast using CatBoost & AdaBoost models.",      "/Short_Term_Forecasting"),
    ("🗺️", "Market Clustering",       "K-Means city segmentation for expansion.",                      "/Market_Clustering"),
    ("🎯", "Target Market Validator", "Score & rank new markets for entry.",                           "/Target_Market_Validator"),
    ("🤖", "SQL RAG Agent",           "Ask questions in plain English using AI.",                      "/SQL_RAG_Agent"),
    ("🔮", "Live Model Forecast",     "Real-time predictions using trained CatBoost and AdaBoost.",    "/Live_Forecast"),
]

cols = st.columns(3)
for i, (icon, title, desc, href) in enumerate(pages_info):
    with cols[i % 3]:
        st.markdown(f"""
        <a class="nav-card" href="{href}" target="_self">
            <span class="nav-card-icon">{icon}</span>
            <div class="nav-card-title">{title}</div>
            <div class="nav-card-desc">{desc}</div>
            <span class="nav-card-arrow">→</span>
        </a>
        """, unsafe_allow_html=True)

st.markdown(f"""
<hr style="border-color: rgba(0,240,255,0.3); margin-top:40px;">
<div style="text-align:center; color:#A0A0A5; font-size:0.78rem; padding-bottom:12px;">
    Apple Sales Intelligence Platform — Built with Streamlit &amp; Plotly — Data: 2021–2025
</div>
""", unsafe_allow_html=True)
