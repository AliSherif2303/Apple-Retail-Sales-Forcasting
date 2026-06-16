# -*- coding: utf-8 -*-
"""Page 5 — Target Market Validator (pre-computed results from market_expansion_intelligence.csv)"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Target Market Validator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent.parent

# ── Styling ────────────────────────────────────────────────────────────
PALETTE = ["#818cf8","#c084fc","#f472b6","#34d399","#fbbf24","#60a5fa"]

DECISION_CONFIG = {
    "EXPAND NOW":            {"color": "#10b981", "bg": "rgba(16,185,129,0.15)",  "border": "rgba(16,185,129,0.5)",  "emoji": "🚀"},
    "HIGH PRIORITY":         {"color": "#34d399", "bg": "rgba(52,211,153,0.12)",  "border": "rgba(52,211,153,0.4)",  "emoji": "✅"},
    "PILOT FIRST":           {"color": "#fbbf24", "bg": "rgba(251,191,36,0.12)",  "border": "rgba(251,191,36,0.4)",  "emoji": "🧪"},
    "CONSIDER WITH CAUTION": {"color": "#fb923c", "bg": "rgba(251,146,60,0.12)",  "border": "rgba(251,146,60,0.4)",  "emoji": "⚠️"},
    "MONITOR":               {"color": "#60a5fa", "bg": "rgba(96,165,250,0.12)",  "border": "rgba(96,165,250,0.4)",  "emoji": "👁️"},
    "WAIT":                  {"color": "#94a3b8", "bg": "rgba(148,163,184,0.10)", "border": "rgba(148,163,184,0.3)", "emoji": "⏳"},
    "AVOID":                 {"color": "#ef4444", "bg": "rgba(239,68,68,0.12)",   "border": "rgba(239,68,68,0.4)",   "emoji": "❌"},
}

PRIORITY_COLOR = {
    "Very High": "#10b981", "High": "#34d399",
    "Medium": "#fbbf24",    "Low": "#fb923c",   "Very Low": "#ef4444",
}

def style_fig(fig, h=400):
    fig.update_layout(
        paper_bgcolor="rgba(10,10,20,0)", plot_bgcolor="rgba(10,10,20,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
        height=h, margin=dict(l=16, r=16, t=42, b=16),
        legend=dict(bgcolor="rgba(15,15,30,0.7)"),
        xaxis=dict(gridcolor="rgba(99,102,241,0.12)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.12)"),
        colorway=PALETTE,
    )

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
.decision-banner { border-radius: 20px; padding: 28px 32px; text-align: center; margin: 24px 0; }
.decision-title { font-size: 2.4rem; font-weight: 900; letter-spacing: 2px; }
.decision-sub { font-size: 0.95rem; margin-top: 8px; opacity: 0.8; }
.reason-box { background: rgba(15,15,30,0.7); border-radius: 14px; padding: 20px 24px;
  border: 1px solid rgba(99,102,241,0.2); margin-top: 12px; }
.pro { color: #34d399; font-size: 0.9rem; margin: 4px 0; }
.con { color: #f87171; font-size: 0.9rem; margin: 4px 0; }
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.metric-chip { background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25);
  border-radius: 10px; padding: 10px 18px; min-width: 120px; }
.chip-val { font-size: 1.2rem; font-weight: 700; color: #00F0FF; }
.chip-lab { font-size: 0.72rem; color: #E1E1E6; margin-top: 2px; }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Load data from CSV ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading market expansion data...")
def load_data():
    csv_path = APP_DIR / "market_expansion_intelligence.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "Country" not in df.columns:
        st.error("Data format error: 'Country' column missing.")
        return None
    df["Country"] = df["Country"].str.strip().str.title()
    return df

df = load_data()
if df is None:
    st.error("⚠️ market_expansion_intelligence.csv not found in the app directory.")
    st.stop()

# ── Page header ────────────────────────────────────────────────────────
st.markdown("# 🎯 Target Market Validator")
st.markdown("*Select a country to see its expansion priority, decision, and detailed analysis.*")

# ── Country selector ───────────────────────────────────────────────────
countries = sorted(df["Country"].dropna().unique().tolist())
selected = st.selectbox(
    "🌍 Choose a Country",
    options=countries,
    index=0,
    help="Select any country from the evaluated list to see its full expansion profile.",
)

row = df[df["Country"] == selected].iloc[0]
decision = str(row.get("Expansion_Decision", "MONITOR")).strip().upper()
cfg = DECISION_CONFIG.get(decision, DECISION_CONFIG["MONITOR"])
priority = str(row.get("Priority", "N/A"))
priority_color = PRIORITY_COLOR.get(priority, "#94a3b8")

# ── Decision Banner ────────────────────────────────────────────────────
st.markdown(f"""
<div class="decision-banner" style="background:{cfg['bg']}; border: 2px solid {cfg['border']};">
    <div class="decision-title" style="color:{cfg['color']};">
        {cfg['emoji']} {decision}
    </div>
    <div class="decision-sub" style="color:{cfg['color']};">
        Expansion Decision for <strong>{selected}</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, val, label, color="#818cf8"):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color};">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

rank_val = int(row["Rank"]) if "Rank" in row and pd.notna(row["Rank"]) else "N/A"
score_val = f"{row['Market_Score']:.1f}/100" if "Market_Score" in row else "N/A"
sales_y1 = f"${row['Est_Sales_Year1_M']:.1f}M" if "Est_Sales_Year1_M" in row else "N/A"
sales_y3 = f"${row['Est_Sales_Year3_M']:.1f}M" if "Est_Sales_Year3_M" in row else "N/A"
urgency = str(row.get("Action_Urgency", "N/A"))

kpi(c1, f"#{rank_val}", "Global Rank")
kpi(c2, priority, "Priority Tier", priority_color)
kpi(c3, score_val, "Market Score (0–100)")
kpi(c4, sales_y1, "Est. Year-1 Sales")
kpi(c5, sales_y3, "Est. Year-3 Sales")

# ── Decision Reason ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Decision Analysis</div>', unsafe_allow_html=True)

reason_raw = str(row.get("Decision_Reason", ""))
lines = [l.strip() for l in reason_raw.split("\n") if l.strip()]
primary = lines[0] if lines else ""
pros_items = [l.replace("✓", "").strip() for l in lines if l.startswith("✓")]
cons_items = [l.replace("⚠️", "").strip() for l in lines if l.startswith("⚠️")]

pros_html = "".join(f'<div class="pro">✓ {p}</div>' for p in pros_items)
cons_html = "".join(f'<div class="con">⚠️ {c}</div>' for c in cons_items)

st.markdown(f"""
<div class="reason-box">
    <div style="font-size:1rem; color:#e2e8f0; font-weight:600; margin-bottom:12px;">
        {primary}
    </div>
    {pros_html}
    {cons_html}
</div>
""", unsafe_allow_html=True)

# ── Market Metrics ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Market Metrics</div>', unsafe_allow_html=True)

gdp        = row.get("GDP", 0)
pop        = row.get("Population", 0)
internet   = row.get("Internet", 0)
inflation  = row.get("Inflation", 0)
exchange   = row.get("ExchangeRate", 0)
pen_rate   = row.get("Penetration_Rate", 0)
risk_pen   = row.get("Risk_Penalty", 1.0)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-chip">
        <div class="chip-val">${gdp:,.0f}</div>
        <div class="chip-lab">GDP per Capita</div>
    </div>
    <div class="metric-chip">
        <div class="chip-val">{pop/1e6:.1f}M</div>
        <div class="chip-lab">Population</div>
    </div>
    <div class="metric-chip">
        <div class="chip-val">{internet:.1f}%</div>
        <div class="chip-lab">Internet Usage</div>
    </div>
    <div class="metric-chip">
        <div class="chip-val">{inflation:.1f}%</div>
        <div class="chip-lab">Inflation Rate</div>
    </div>
    <div class="metric-chip">
        <div class="chip-val">{exchange:.2f}</div>
        <div class="chip-lab">Exchange Rate (USD)</div>
    </div>
    <div class="metric-chip">
        <div class="chip-val">{pen_rate:.1f}%</div>
        <div class="chip-lab">Penetration Rate</div>
    </div>
    <div class="metric-chip">
        <div class="chip-val">{risk_pen:.2f}</div>
        <div class="chip-lab">Risk Penalty (1=low)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Comparison Charts ──────────────────────────────────────────────────
st.markdown('<div class="section-header">🌐 Global Comparison</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    chart_df = df.copy().sort_values("Market_Score", ascending=True).tail(20)
    chart_df["_color"] = chart_df["Country"].apply(
        lambda x: cfg["color"] if x == selected else "rgba(99,102,241,0.5)"
    )
    fig1 = go.Figure(go.Bar(
        x=chart_df["Market_Score"],
        y=chart_df["Country"],
        orientation="h",
        marker_color=chart_df["_color"],
        text=chart_df["Market_Score"].round(1),
        textposition="outside",
    ))
    fig1.update_layout(title="Top 20 Markets by Score (highlighted: selected)",
                       xaxis_title="Market Score (0–100)")
    style_fig(fig1, 480)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    pc = df["Priority"].value_counts().reset_index()
    pc.columns = ["Priority", "Count"]
    color_map = {
        "Very High": "#10b981", "High": "#34d399",
        "Medium": "#fbbf24", "Low": "#fb923c", "Very Low": "#ef4444",
    }
    fig2 = px.pie(pc, values="Count", names="Priority",
                  title="Priority Distribution — All Countries",
                  hole=0.5, color="Priority", color_discrete_map=color_map)
    fig2.update_traces(textinfo="label+percent")
    style_fig(fig2, 380)
    st.plotly_chart(fig2, use_container_width=True)

# ── Expansion decision distribution ───────────────────────────────────
st.markdown('<div class="section-header">🗺️ Expansion Decision Overview</div>', unsafe_allow_html=True)
fig3 = px.bar(
    df.sort_values("Est_Sales_Year1_M", ascending=False).head(25),
    x="Country", y="Est_Sales_Year1_M",
    color="Expansion_Decision",
    title="Top 25 Countries by Estimated Year-1 Sales ($M)",
    labels={"Est_Sales_Year1_M": "Est. Year-1 Sales ($M)", "Expansion_Decision": "Decision"},
    color_discrete_map={
        "EXPAND NOW": "#10b981", "HIGH PRIORITY": "#34d399",
        "PILOT FIRST": "#fbbf24", "CONSIDER WITH CAUTION": "#fb923c",
        "MONITOR": "#60a5fa", "WAIT": "#94a3b8", "AVOID": "#ef4444",
    },
)
fig3.update_layout(xaxis_tickangle=-40)
style_fig(fig3, 420)
st.plotly_chart(fig3, use_container_width=True)

# ── Risk vs Reward scatter ─────────────────────────────────────────────
st.markdown('<div class="section-header">⚖️ Risk vs Reward Matrix</div>', unsafe_allow_html=True)
if "Risk_Penalty" in df.columns:
    scatter_df = df.copy()
    scatter_df["Risk_Category"] = scatter_df["Risk_Penalty"].apply(
        lambda p: "Low Risk" if p > 0.85 else ("Medium Risk" if p > 0.7 else "High Risk")
    )
    fig4 = px.scatter(
        scatter_df,
        x="Market_Score",
        y="Est_Sales_Year1_M",
        color="Risk_Category",
        size="Est_Sales_Year1_M",
        hover_name="Country",
        title="Risk vs Reward (bubble = Year-1 sales potential)",
        log_y=True,
        color_discrete_map={"Low Risk": "#10b981", "Medium Risk": "#fbbf24", "High Risk": "#ef4444"},
        labels={"Market_Score": "Market Score (0–100)", "Est_Sales_Year1_M": "Year-1 Sales ($M)"},
    )
    style_fig(fig4, 420)
    st.plotly_chart(fig4, use_container_width=True)

# ── Full table ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 All Countries — Ranked</div>', unsafe_allow_html=True)
show_cols = [c for c in [
    "Rank", "Country", "Market_Score", "Priority", "Expansion_Decision",
    "Action_Urgency", "Est_Sales_Year1_M", "Est_Sales_Year3_M",
    "GDP", "Population", "Internet", "Inflation",
] if c in df.columns]
show = df[show_cols].copy()
if "Est_Sales_Year1_M" in show.columns:
    show["Est_Sales_Year1_M"] = show["Est_Sales_Year1_M"].round(2)
if "Est_Sales_Year3_M" in show.columns:
    show["Est_Sales_Year3_M"] = show["Est_Sales_Year3_M"].round(2)
if "Market_Score" in show.columns:
    show["Market_Score"] = show["Market_Score"].round(1)
st.dataframe(show, use_container_width=True, height=400)
st.download_button(
    "📥 Download Full Rankings CSV",
    df.to_csv(index=False),
    "market_expansion_intelligence.csv",
)
