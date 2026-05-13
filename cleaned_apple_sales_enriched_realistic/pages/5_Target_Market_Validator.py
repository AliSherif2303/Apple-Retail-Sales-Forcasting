# -*- coding: utf-8 -*-
"""Page 5 — Target Market Validator (from Target Market Validator.py)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT = APP_DIR.parent
PROC = PROJECT / "data" / "processed"

PALETTE = ["#818cf8","#c084fc","#f472b6","#34d399","#fbbf24","#60a5fa"]
def style_fig(fig, h=420):
    fig.update_layout(paper_bgcolor="rgba(10,10,20,0)", plot_bgcolor="rgba(10,10,20,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
        height=h, margin=dict(l=16,r=16,t=42,b=16),
        legend=dict(bgcolor="rgba(15,15,30,0.7)"),
        xaxis=dict(gridcolor="rgba(99,102,241,0.12)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.12)"), colorway=PALETTE)

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 40%, #0a0f1a 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%); }
.kpi-card { background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
  border: 1px solid rgba(99,102,241,0.3); border-radius: 16px; padding: 20px 24px; text-align: center; }
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #818cf8; }
.kpi-label { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }
.section-header { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin: 28px 0 12px 0;
  padding-bottom: 8px; border-bottom: 2px solid rgba(99,102,241,0.3); }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("# 🎯 Target Market Validator")
st.markdown("*Score and rank new markets for Apple retail expansion.*")

def _kpi(v,l): st.markdown(f'<div class="kpi-card"><div class="kpi-value">{v}</div><div class="kpi-label">{l}</div></div>', unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading market data...")
def load_data():
    s = PROC / "merged_city_sales_data.csv"
    c = PROC / "filtered_data.csv"
    if not s.exists() or not c.exists():
        return None, None
    sales = pd.read_csv(s)
    # filtered_data.csv may have non-UTF8 characters
    for enc in ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']:
        try:
            country = pd.read_csv(c, encoding=enc)
            return sales, country
        except UnicodeDecodeError:
            continue
    return None, None

sales_df, country_df = load_data()
if sales_df is None:
    st.error("Required data files not found in data/processed/.")
    st.stop()

# Historical analysis
hist = sales_df.groupby("country_norm_mapped").agg(
    TotalSales=("sales_amount_realistic","sum"),
    GDP=("gdp_per_capita","mean"),
    Population=("Population","mean"),
    InternetUsage=("internet_usage_pct","mean"),
).reset_index()
hist.columns = ["Country","TotalSales","GDP","Population","InternetUsage"]
hist["SalesPerCapita"] = hist["TotalSales"] / hist["Population"]

# Score new markets
pred = country_df.copy()
pred.rename(columns={"Country Name":"Country","GDP per capita (USD)":"GDP",
    "Internet Usage (%)":"Internet","Inflation Rate (%)":"Inflation",
    "Official Exchange Rate":"ExchangeRate"}, inplace=True)
pred["Population"] = pd.to_numeric(pred.get("Population",0), errors="coerce")

def score(row):
    s = 0
    if row["GDP"] > 50000: s += 3
    elif row["GDP"] > 20000: s += 2
    elif row["GDP"] > 5000: s += 1
    if row["Population"] > 100_000_000: s += 3
    elif row["Population"] > 50_000_000: s += 2
    elif row["Population"] > 10_000_000: s += 1
    if row.get("Internet",0) > 80: s += 2
    elif row.get("Internet",0) > 50: s += 1
    return s

pred["Market_Score"] = pred.apply(score, axis=1)
pred["Priority"] = pd.cut(pred["Market_Score"], bins=[-1,2,4,6,9],
    labels=["Very Low","Low","Medium","High"])
pred = pred.sort_values(["Market_Score"], ascending=False).reset_index(drop=True)
pred["Rank"] = pred.index + 1

# KPIs
high = len(pred[pred["Priority"]=="High"])
med = len(pred[pred["Priority"]=="Medium"])
c1,c2,c3,c4 = st.columns(4)
with c1: _kpi(str(len(pred)), "Countries Evaluated")
with c2: _kpi(str(high), "High Priority")
with c3: _kpi(str(med), "Medium Priority")
with c4: _kpi(f"{pred['Market_Score'].mean():.1f}/8", "Avg Score")

st.markdown("")

# Scoring explanation
with st.expander("📘 Scoring System Explained"):
    st.markdown("""
| Criteria | Score |
|----------|-------|
| GDP > $50K | +3 |  GDP $20K-50K | +2 |  GDP $5K-20K | +1 |
| Pop > 100M | +3 |  Pop 50-100M | +2 |  Pop 10-50M | +1 |
| Internet > 80% | +2 |  Internet 50-80% | +1 |
| **Max Score** | **8** |
""")

# Priority distribution
st.markdown('<div class="section-header">📊 Priority Distribution</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    pc = pred["Priority"].value_counts()
    fig = px.pie(values=pc.values, names=pc.index, title="Market Priority Distribution",
        color_discrete_sequence=["#34d399","#fbbf24","#fb923c","#ef4444"])
    style_fig(fig, 380)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    top15 = pred.head(15)
    colors = ["#34d399" if x=="High" else "#fbbf24" if x=="Medium" else "#fb923c" for x in top15["Priority"]]
    fig2 = px.bar(top15, x="Market_Score", y="Country", orientation="h",
        title="Top 15 Markets by Score", color="Priority",
        color_discrete_map={"High":"#34d399","Medium":"#fbbf24","Low":"#fb923c","Very Low":"#ef4444"})
    fig2.update_layout(yaxis=dict(autorange="reversed"))
    style_fig(fig2, 450)
    st.plotly_chart(fig2, use_container_width=True)

# Full table
st.markdown('<div class="section-header">📋 Complete Rankings</div>', unsafe_allow_html=True)
show_cols = [c for c in ["Rank","Country","GDP","Population","Internet","Market_Score","Priority"] if c in pred.columns]
st.dataframe(pred[show_cols], use_container_width=True, height=400)
st.download_button("📥 Download Rankings CSV", pred.to_csv(index=False), "market_rankings.csv")
