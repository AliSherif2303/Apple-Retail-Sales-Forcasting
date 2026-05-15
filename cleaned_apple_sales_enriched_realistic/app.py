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

/* ── Interactive nav cards ── */
.nav-card {
    display: block;
    text-decoration: none !important;
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.06));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 28px 24px;
    margin-bottom: 16px;
    min-height: 170px;
    cursor: pointer;
    transition: transform 0.22s ease, box-shadow 0.22s ease,
                border-color 0.22s ease, background 0.22s ease;
    position: relative;
    overflow: hidden;
}
.nav-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 16px;
    background: radial-gradient(circle at 50% 0%, rgba(99,102,241,0.18) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.22s ease;
}
.nav-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 48px rgba(99,102,241,0.30), 0 0 0 1px rgba(129,140,248,0.4);
    border-color: rgba(99,102,241,0.6);
    background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(139,92,246,0.12));
    text-decoration: none !important;
}
.nav-card:hover::before { opacity: 1; }
.nav-card:active { transform: translateY(-2px); }

.nav-card-icon {
    font-size: 2.4rem;
    margin-bottom: 10px;
    display: block;
    filter: drop-shadow(0 0 10px rgba(129,140,248,0.5));
    transition: filter 0.22s ease, transform 0.22s ease;
}
.nav-card:hover .nav-card-icon {
    filter: drop-shadow(0 0 18px rgba(129,140,248,0.9));
    transform: scale(1.12);
}
.nav-card-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 8px;
    transition: color 0.22s ease;
}
.nav-card:hover .nav-card-title { color: #a5b4fc; }
.nav-card-desc {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Arrow indicator on hover */
.nav-card-arrow {
    position: absolute;
    bottom: 16px;
    right: 18px;
    font-size: 1rem;
    color: rgba(99,102,241,0.0);
    transition: color 0.22s ease, transform 0.22s ease;
}
.nav-card:hover .nav-card-arrow {
    color: rgba(129,140,248,0.7);
    transform: translateX(4px);
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
<div style="text-align:center; padding: 60px 0 36px 0;">
    <span style="font-size:4rem; filter: drop-shadow(0 0 24px rgba(129,140,248,0.6));">🍎</span>
    <h1 style="color:#818cf8; margin-top:16px; font-size:2.4rem;">
        Apple Sales Intelligence Platform
    </h1>
    <p style="color:#94a3b8; font-size:1.1rem; max-width:600px; margin:16px auto;">
        A comprehensive analytics suite for Apple retail sales forecasting,
        market expansion, and AI-powered data insights.
    </p>
    <p style="color:#64748b; font-size:0.85rem; margin-top:8px;">
        ✨ Click any card below to navigate directly to that page
    </p>
</div>
""", unsafe_allow_html=True)

# Page cards — href must match Streamlit's URL routing from filenames
pages_info = [
    ("🏠", "Dashboard",               "Interactive EDA with KPIs, trends, and market analysis.",      "/Dashboard"),
    ("📈", "Long-Term Forecasting",   "Combined AdaBoost & CatBoost 12-month forecast.",              "/Long_Term_Forecasting"),
    ("📊", "Short-Term Forecasting",  "Pre-computed model predictions by store.",                      "/Short_Term_Forecasting"),
    ("🗺️", "Market Clustering",       "K-Means city segmentation for expansion.",                      "/Market_Clustering"),
    ("🎯", "Target Market Validator", "Score & rank new markets for entry.",                           "/Target_Market_Validator"),
    ("🤖", "SQL RAG Agent",           "Ask questions in plain English using AI.",                      "/SQL_RAG_Agent"),
    ("🔮", "Live Forecast",           "Pick a store & horizon — predict with CatBoost & AdaBoost.",   "/Live_Forecast"),
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

st.markdown("""
<hr style="border-color: rgba(99,102,241,0.3); margin-top:40px;">
<div style="text-align:center; color:#334155; font-size:0.78rem; padding-bottom:12px;">
    Apple Sales Intelligence Platform — Built with Streamlit &amp; Plotly — Data: 2021-2025
</div>
""", unsafe_allow_html=True)
