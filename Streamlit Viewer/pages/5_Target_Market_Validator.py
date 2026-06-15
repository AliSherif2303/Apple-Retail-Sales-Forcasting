# -*- coding: utf-8 -*-
"""Page 5 — Target Market Validator (pre-computed results explorer)"""
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
CHART_BG   = "rgba(28,28,30,0.6)"
PAPER_BG   = "rgba(0,0,0,0)"
FONT_COLOR = "#E1E1E6"
GRID_COLOR = "rgba(0,240,255,0.10)"
PALETTE    = ["#00F0FF","#39FF14","#FF5252","#5E5CE6","#FFFFFF","#00B4FF","#A0A0A5","#FFD60A"]

DECISION_CONFIG = {
    "STRONG YES": {"color": "#39FF14", "bg": "rgba(57,255,20,0.15)", "border": "rgba(57,255,20,0.5)",  "emoji": "🚀"},
    "YES":        {"color": "#34d399", "bg": "rgba(52,211,153,0.12)",  "border": "rgba(52,211,153,0.4)",  "emoji": "✅"},
    "CONSIDER":   {"color": "#fbbf24", "bg": "rgba(251,191,36,0.12)",  "border": "rgba(251,191,36,0.4)",  "emoji": "🤔"},
    "CAUTIOUS":   {"color": "#fb923c", "bg": "rgba(251,146,60,0.12)",  "border": "rgba(251,146,60,0.4)",  "emoji": "⚠️"},
    "NO":         {"color": "#FF5252", "bg": "rgba(255,82,82,0.12)",   "border": "rgba(255,82,82,0.4)",   "emoji": "❌"},
}
PRIORITY_COLOR = {
    "Very High": "#39FF14", "High": "#34d399",
    "Medium": "#fbbf24",    "Low": "#fb923c",
}

def style_fig(fig, h=400):
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
        height=h, margin=dict(l=16, r=16, t=42, b=16),
        legend=dict(bgcolor="rgba(28,28,30,0.85)", bordercolor="rgba(0,240,255,0.3)", borderwidth=1),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="rgba(0,240,255,0.2)"),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="rgba(0,240,255,0.2)"),
        colorway=PALETTE,
    )

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #000000; }
[data-testid="stSidebar"] { background: #1C1C1E; border-right: 1px solid rgba(0,240,255,0.2); }
.kpi-card { background: #1C1C1E;
  border: 1px solid rgba(0,240,255,0.35); border-radius: 16px; padding: 20px 24px; text-align: center;
  transition: transform 0.2s ease; }
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,240,255,0.2); }
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #00F0FF; }
.kpi-label { font-size: 0.82rem; color: #E1E1E6; margin-top: 4px; }
.section-header { font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 28px 0 12px 0;
  padding-bottom: 8px; border-bottom: 2px solid rgba(0,240,255,0.4); }
.decision-banner { border-radius: 20px; padding: 28px 32px; text-align: center; margin: 24px 0; }
.decision-title { font-size: 2.4rem; font-weight: 900; letter-spacing: 2px; }
.decision-sub { font-size: 0.95rem; margin-top: 8px; opacity: 0.8; }
.reason-box { background: #1C1C1E; border-radius: 14px; padding: 20px 24px;
  border: 1px solid rgba(0,240,255,0.3); margin-top: 12px; }
.pro { color: #39FF14; font-size: 0.9rem; margin: 4px 0; }
.con { color: #FF5252; font-size: 0.9rem; margin: 4px 0; }
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.metric-chip { background: #1C1C1E; border: 1px solid rgba(0,240,255,0.3);
  border-radius: 10px; padding: 10px 18px; min-width: 120px; }
.chip-val { font-size: 1.2rem; font-weight: 700; color: #00F0FF; }
.chip-lab { font-size: 0.72rem; color: #E1E1E6; margin-top: 2px; }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Inline data (avoids git-lfs issues on VM) ─────────────────────────
import io

_CSV = """Rank,Country,GDP,Population,Internet,Market_Score,Priority,Estimated_Annual_Sales_M,Confidence,Expansion_Decision,Risk_Level,Decision_Reason,Inflation,ExchangeRate
1,Belgium,56614.57,2150300,95.78,6,High,249.05,High (GDP within known range),YES,Low-Medium,"Strong market opportunity with manageable risks | Pros: Strong score (6/8); $249M potential; High GDP | Cons: Small population (2.2M)",3.14,0.92
2,Saudi Arabia,35121.66,7950000,100.0,5,High,587.77,High (GDP within known range),YES,Low-Medium,"Strong market opportunity with manageable risks | Pros: Strong score (5/8); $588M potential; High GDP | Cons: Small population (8.0M)",1.69,3.75
3,Denmark,71026.48,1409680,99.77,5,High,276.62,High (GDP within known range),YES,Low-Medium,"Strong market opportunity with manageable risks | Pros: Strong score (5/8); $277M potential; High GDP | Cons: Small population (1.4M)",1.37,6.89
4,Sweden,57117.49,1730000,95.53,5,High,201.81,High (GDP within known range),YES,Low-Medium,"Strong market opportunity with manageable risks | Pros: Strong score (5/8); $202M potential; High GDP | Cons: Small population (1.7M)",2.84,10.57
5,Russian Federation,14889.02,12750000,94.37,5,High,167.43,High (GDP within known range),YES,Low-Medium,"Strong market opportunity with manageable risks | Pros: Strong score (5/8); $167M potential; Excellent digital (94%)",8.43,92.55
6,Luxembourg,137781.68,138000,98.76,5,High,79.79,Medium (GDP outside historical range),YES,Low-Medium,"Strong market opportunity with manageable risks | Pros: Strong score (5/8); $80M potential; Very High GDP | Cons: Very small population (0.1M)",2.05,0.92
7,Argentina,13969.78,15890600,89.67,4,Medium,211.58,High (GDP within known range),CONSIDER,Medium,"Moderate opportunity with good sales potential | Pros: $212M potential; Good digital (90%) | Cons: High inflation (219.9%)",219.88,914.69
8,Malaysia,11874.43,8900000,98.02,4,Medium,111.79,High (GDP within known range),CONSIDER,Medium,"Moderate opportunity with good sales potential | Pros: $112M potential; Excellent digital (98%) | Cons: Small population (8.9M)",1.83,4.58
9,Israel,54176.68,1020000,88.18,4,Medium,107.16,High (GDP within known range),CONSIDER,Medium,"Moderate opportunity with good sales potential | Pros: $107M potential; High GDP | Cons: Small population (1.0M)",3.07,3.70
10,Egypt Arab Rep.,3338.47,22600000,72.22,3,Medium,205.47,Medium (GDP outside historical range),CONSIDER,Medium,"Moderate opportunity with good sales potential | Pros: $205M potential; Large population | Cons: Low GDP; High inflation (28.3%)",28.27,45.30
11,Peru,8452.37,11600000,81.96,3,Medium,129.89,High (GDP within known range),CONSIDER,Medium,"Moderate opportunity with good sales potential | Pros: $130M potential; Good digital (82%)",2.01,3.75
12,Indonesia,4925.43,11400000,72.78,3,Medium,118.78,Medium (GDP outside historical range),CONSIDER,Medium,"Moderate opportunity with good sales potential | Pros: $119M potential; Good digital (73%) | Cons: Low GDP",2.18,15855.45
13,Turkiye,15892.72,5500000,87.31,3,Medium,101.35,High (GDP within known range),CONSIDER,Medium,"Moderate opportunity with good sales potential | Pros: $101M potential; Upper-middle income | Cons: High inflation (58.5%)",58.51,32.81
14,Dominican Republic,10875.66,3680000,91.00,3,Medium,44.03,High (GDP within known range),CONSIDER,Medium,"Moderate opportunity with good sales potential | Pros: $44M potential; Good digital (91%) | Cons: Small population (3.7M)",3.30,59.57
15,Portugal,29292.24,3010000,88.49,2,Low,161.33,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $161M potential; Upper-middle income | Cons: Low score (2/8); Small population (3.0M)",2.42,0.92
16,Kenya,2132.43,5650000,34.98,2,Low,90.09,Medium (GDP outside historical range),CAUTIOUS,High,"Low score but decent potential | Pros: $90M potential | Cons: Low score (2/8); Low GDP",4.49,134.82
17,Viet Nam,4717.29,5600000,84.15,2,Low,63.75,Medium (GDP outside historical range),CAUTIOUS,High,"Low score but decent potential | Pros: $64M potential; Good digital (84%) | Cons: Low score (2/8); Low GDP",3.62,24164.89
18,Brazil,10310.55,5040700,84.46,2,Low,57.52,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $58M potential; Good digital (84%) | Cons: Low score (2/8)",4.37,5.39
19,Uganda,1077.91,4200000,8.95,2,Low,46.46,Medium (GDP outside historical range),CAUTIOUS,High,"Low score but decent potential | Pros: $46M potential | Cons: Low score (2/8); Very low GDP; Poor digital (9%)",3.32,3757.26
20,Bahrain,29653.57,759300,100.0,2,Low,44.62,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $45M potential; Upper-middle income; 100% internet | Cons: Low score (2/8); Very small population",0.92,0.38
21,Hungary,23292.33,1780000,93.78,2,Low,37.69,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $38M potential; Good digital (94%) | Cons: Low score (2/8); Small population (1.8M)",3.70,365.69
22,Oman,20285.23,1710000,95.25,2,Low,33.33,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $33M potential; Good digital (95%) | Cons: Low score (2/8); Small population (1.7M)",0.59,0.38
23,Estonia,31428.35,460000,92.24,2,Low,28.86,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $29M potential; High GDP | Cons: Low score (2/8); Very small population",3.52,0.92
24,Belarus,8317.63,2075900,94.26,2,Low,25.41,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $25M potential; Good digital (94%) | Cons: Low score (2/8); Small population",5.79,3.25
25,Slovenia,34301.03,290000,90.76,2,Low,19.65,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $20M potential; High GDP | Cons: Low score (2/8); Very small population",1.97,0.92
26,Kazakhstan,14154.63,1410000,93.39,2,Low,19.27,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $19M potential; Good digital (93%) | Cons: Low score (2/8); Small population",8.84,468.96
27,Latvia,23409.08,610000,92.71,2,Low,12.88,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $13M potential; Good digital (93%) | Cons: Low score (2/8); Small population",1.27,0.92
28,Czechia,31823.31,1335350,87.69,1,Low,82.11,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $82M potential; High GDP | Cons: Low score (1/8); Small population",2.44,23.22
29,Greece,24626.15,3150000,86.27,1,Low,81.33,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $81M potential; Upper-middle income | Cons: Low score (1/8); Small population",2.74,0.92
30,Poland,25103.57,1810000,88.59,1,Low,65.82,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $66M potential; Upper-middle income | Cons: Low score (1/8); Small population",3.78,3.98
31,Uruguay,23906.51,1390000,91.99,1,Low,36.30,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $36M potential; Good digital (92%) | Cons: Low score (1/8); Small population",4.85,40.21
32,Romania,20080.21,1800000,91.29,1,Low,34.13,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $34M potential; Good digital (91%) | Cons: Low score (1/8); Small population",5.72,4.60
33,Slovak Republic,25992.67,445000,89.83,1,Low,21.37,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $21M potential; Upper-middle income | Cons: Low score (1/8); Very small population",2.76,0.92
34,Serbia,13679.21,1410000,87.69,1,Low,18.51,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $19M potential; Good digital (88%) | Cons: Low score (1/8); Small population",4.67,108.21
35,Montenegro,13263.33,185000,88.88,1,Low,2.18,High (GDP within known range),NO,Very High,"Unfavorable market conditions | Pros: Good digital (89%) | Cons: Low score (1/8); $2.2M potential only; Very small population",3.34,0.92
36,Bulgaria,17596.02,1290000,82.44,0,Low,22.93,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $23M potential; Good digital (82%) | Cons: Low score (0/8); Small population",2.45,1.81
37,Ecuador,6874.71,2050000,77.17,0,Low,22.12,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $22M potential; Good digital (77%) | Cons: Low score (0/8); Small population",1.55,1.0
38,Croatia,24050.44,805000,83.63,0,Low,19.90,High (GDP within known range),CAUTIOUS,High,"Low score but decent potential | Pros: $20M potential; Good digital (84%) | Cons: Low score (0/8); Small population",2.97,0.92
39,Paraguay,6416.10,535000,81.58,0,Low,5.97,Medium (GDP outside historical range),NO,Very High,"Unfavorable market conditions | Pros: Good digital (82%) | Cons: Low score (0/8); Very small population",3.84,7560.25
40,Bosnia And Herzegovina,9358.79,345000,86.10,0,Low,3.98,High (GDP within known range),NO,Very High,"Unfavorable market conditions | Pros: Good digital (86%) | Cons: Low score (0/8); Very small population",1.69,1.81
"""

@st.cache_data(show_spinner="Loading market data...")
def load_data():
    df = pd.read_csv(io.StringIO(_CSV))
    # Defensive: ensure Country column exists (handles any edge case)
    if "Country" not in df.columns:
        st.error("Data format error: 'Country' column missing.")
        return None
    df["Country"] = df["Country"].str.strip().str.title()
    return df

df = load_data()
if df is None:
    st.error("⚠️ market_expansion_priorities_adjusted.csv not found in the app directory.")
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
decision = str(row["Expansion_Decision"]).strip()
cfg = DECISION_CONFIG.get(decision, DECISION_CONFIG["CONSIDER"])
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

def kpi(col, val, label, color="#00F0FF"):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color};">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(c1, f"#{int(row['Rank'])}", "Global Rank")
kpi(c2, priority, "Priority Tier", priority_color)
kpi(c3, f"{int(row['Market_Score'])}/8", "Market Score")
kpi(c4, f"${row['Estimated_Annual_Sales_M']:.0f}M", "Est. Annual Sales")
risk = str(row.get("Risk_Level", "N/A"))
risk_color = {"Low": "#34d399", "Low-Medium": "#34d399", "Medium": "#fbbf24",
              "Medium-High": "#fb923c", "High": "#fb923c", "Very High": "#ef4444"}.get(risk, "#94a3b8")
kpi(c5, risk, "Risk Level", risk_color)

# ── Decision Reason ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Decision Analysis</div>', unsafe_allow_html=True)

reason_raw = str(row.get("Decision_Reason", ""))
lines = reason_raw.split("\n")
primary = lines[0].strip() if lines else ""
pros_line = next((l for l in lines if "✓ Pros:" in l), "")
cons_line = next((l for l in lines if "✗ Cons:" in l), "")

pros_items = pros_line.replace("✓ Pros:", "").strip().split(";") if pros_line else []
cons_items = cons_line.replace("✗ Cons:", "").strip().split(";") if cons_line else []

pros_html = "".join(f'<div class="pro">✓ {p.strip()}</div>' for p in pros_items if p.strip())
cons_html = "".join(f'<div class="con">✗ {c.strip()}</div>' for c in cons_items if c.strip())

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

gdp = row.get("GDP", 0)
pop = row.get("Population", 0)
internet = row.get("Internet", 0)
inflation = row.get("Inflation", 0)
exchange = row.get("ExchangeRate", 0)
confidence = row.get("Confidence", "N/A")

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
        <div class="chip-val" style="font-size:0.95rem;">{confidence.split(" ")[0]}</div>
        <div class="chip-lab">Forecast Confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Comparison Charts ──────────────────────────────────────────────────
st.markdown('<div class="section-header">🌐 Global Comparison</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # All countries bar chart — highlight selected
    chart_df = df.copy().sort_values("Market_Score", ascending=True).tail(20)
    chart_df["_color"] = chart_df["Country"].apply(
        lambda x: cfg["color"] if x == selected else "rgba(0,240,255,0.2)"
    )
    fig1 = go.Figure(go.Bar(
        x=chart_df["Market_Score"],
        y=chart_df["Country"],
        orientation="h",
        marker_color=chart_df["_color"],
        text=chart_df["Market_Score"],
        textposition="outside",
    ))
    fig1.update_layout(title="Top 20 Markets by Score (highlighted: selected)",
                       xaxis_title="Market Score (0–8)")
    style_fig(fig1, 480)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Priority distribution donut
    pc = df["Priority"].value_counts().reset_index()
    pc.columns = ["Priority", "Count"]
    color_map = {"Very High": "#10b981", "High": "#34d399", "Medium": "#fbbf24", "Low": "#fb923c"}
    fig2 = px.pie(pc, values="Count", names="Priority",
                  title="Priority Distribution — All Countries",
                  hole=0.5,
                  color="Priority",
                  color_discrete_map=color_map)
    fig2.update_traces(textinfo="label+percent")
    style_fig(fig2, 380)
    st.plotly_chart(fig2, use_container_width=True)

# ── Expansion decision distribution ───────────────────────────────────
st.markdown('<div class="section-header">🗺️ Expansion Decision Overview</div>', unsafe_allow_html=True)
dec_df = df.copy()
dec_df["Country_display"] = dec_df["Country"] + " (" + dec_df["Expansion_Decision"] + ")"
fig3 = px.bar(
    dec_df.sort_values("Estimated_Annual_Sales_M", ascending=False).head(25),
    x="Country", y="Estimated_Annual_Sales_M",
    color="Expansion_Decision",
    title="Top 25 Countries by Estimated Annual Sales ($M)",
    labels={"Estimated_Annual_Sales_M": "Est. Sales ($M)", "Expansion_Decision": "Decision"},
    color_discrete_map={
        "STRONG YES": "#10b981", "YES": "#34d399",
        "CONSIDER": "#fbbf24", "CAUTIOUS": "#fb923c", "NO": "#ef4444",
    },
)
fig3.update_layout(xaxis_tickangle=-40)
style_fig(fig3, 420)
st.plotly_chart(fig3, use_container_width=True)

# ── Full table ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 All Countries — Ranked</div>', unsafe_allow_html=True)
show = df[["Rank","Country","Market_Score","Priority","Expansion_Decision",
           "Risk_Level","Estimated_Annual_Sales_M","GDP","Internet"]].copy()
show["Estimated_Annual_Sales_M"] = show["Estimated_Annual_Sales_M"].round(1)
show["GDP"] = show["GDP"].round(0).astype(int)
show["Internet"] = show["Internet"].round(1)
st.dataframe(show, use_container_width=True, height=380)
st.download_button("📥 Download Full Rankings CSV",
                   df.to_csv(index=False),
                   "market_expansion_priorities_adjusted.csv")
