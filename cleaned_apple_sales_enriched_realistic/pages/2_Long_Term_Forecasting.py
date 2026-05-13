# -*- coding: utf-8 -*-
"""Page 2 — Long-Term Forecasting (12-Month Recursive Forecast)
Source notebook: cat&ada.ipynb
One notebook contains BOTH AdaBoost & CatBoost models, forecasting 12 months ahead
using recursive (step-by-step) prediction for each store.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT = APP_DIR.parent
PROC = PROJECT / "data" / "processed"

CHART_BG = "rgba(10,10,20,0)"
PAPER_BG = "rgba(10,10,20,0)"
FONT_COLOR = "#cbd5e1"
GRID_COLOR = "rgba(99,102,241,0.12)"
ADA_COLOR = "#818cf8"      # indigo for AdaBoost
CAT_COLOR = "#f472b6"      # pink for CatBoost
HIST_COLOR = "#34d399"     # green for historical
PALETTE = ["#818cf8","#c084fc","#f472b6","#34d399","#fbbf24","#60a5fa","#fb923c","#a78bfa"]

def style_fig(fig, h=420):
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
        height=h, margin=dict(l=16,r=16,t=42,b=16),
        legend=dict(bgcolor="rgba(15,15,30,0.7)", bordercolor="rgba(99,102,241,0.25)", borderwidth=1),
        xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR), colorway=PALETTE)

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 40%, #0a0f1a 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%); }
.kpi-card { background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
  border: 1px solid rgba(99,102,241,0.3); border-radius: 16px; padding: 20px 24px; text-align: center;
  transition: transform 0.2s ease; }
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(99,102,241,0.25); }
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #818cf8; }
.kpi-label { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }
.section-header { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin: 28px 0 12px 0;
  padding-bottom: 8px; border-bottom: 2px solid rgba(99,102,241,0.3); }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("# 📈 Long-Term Forecasting")
st.markdown("*12-month recursive forecast (Jan–Dec 2026) using combined AdaBoost & CatBoost models.*")
st.caption("Source: `cat&ada.ipynb` — one notebook trains both models and forecasts 12 months recursively.")

def _kpi(v, l):
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{v}</div>'
                f'<div class="kpi-label">{l}</div></div>', unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading long-term forecast data...")
def load_all():
    data = {}
    # Long-term 12-month forecasts from cat&ada.ipynb
    # These CSVs should contain rows for each store × each of the 12 forecast months
    for key, fname in [
        ('ada_fc', 'long_term_ada_forecast_2026.csv'),
        ('cat_fc', 'long_term_cat_forecast_2026.csv'),
    ]:
        p = PROC / fname
        if p.exists():
            df = pd.read_csv(p)
            df['date'] = pd.to_datetime(df['date'])
            data[key] = df
        else:
            data[key] = None
    # Long-term store metrics from cat&ada.ipynb
    for key, fname in [
        ('ada_m', 'long_term_ada_store_metrics.csv'),
        ('cat_m', 'long_term_cat_store_metrics.csv'),
    ]:
        p = PROC / fname
        data[key] = pd.read_csv(p) if p.exists() else None
    # Historical data
    p = PROC / 'cleaned_apple_sales_v3.csv'
    if p.exists():
        raw = pd.read_csv(p)
        raw['sale_date'] = pd.to_datetime(raw['sale_date'])
        hist = raw.groupby(raw['sale_date'].dt.to_period('M'))['sales_amount_realistic'].sum().reset_index()
        hist.columns = ['date', 'sales']
        hist['date'] = hist['date'].dt.to_timestamp()
        data['historical'] = hist
    else:
        data['historical'] = None
        
    return data

data = load_all()

if data['ada_fc'] is None and data['cat_fc'] is None:
    st.warning("⚠️ **No long-term forecast CSV files found.**")
    st.markdown("""
    The long-term forecasting page requires 12-month recursive forecast data from the 
    `cat&ada.ipynb` notebook. This notebook trains both AdaBoost and CatBoost models 
    and forecasts **one full year** (Jan–Dec 2025) step by step.
    
    **To generate the data:**
    1. Open `notebooks/cat&ada.ipynb` in Jupyter
    2. Run all cells to train the models and generate forecasts
    3. At the end of the notebook, add cells to save the forecast DataFrames:
    
    ```python
    # Save AdaBoost 12-month forecast
    ada_forecast_df.to_csv('../data/processed/long_term_ada_forecast_2025.csv', index=False)
    
    # Save CatBoost 12-month forecast  
    cat_forecast_df.to_csv('../data/processed/long_term_cat_forecast_2025.csv', index=False)
    
    # Save per-store metrics (if available)
    ada_store_metrics.to_csv('../data/processed/long_term_ada_store_metrics.csv', index=False)
    cat_store_metrics.to_csv('../data/processed/long_term_cat_store_metrics.csv', index=False)
    ```
    
    4. Re-run this page after saving the CSV files.
    
    **Expected CSV format:** Each file should have columns: `store_id`, `store_name`, `country`, `date`, `predicted_sales`  
    with rows for each store × each of the 12 forecast months (Jan 2025 – Dec 2025).
    """)
    st.stop()

# ─── Helper: aggregate forecast to monthly total ─────────────────────
def agg_monthly(fc_df):
    if fc_df is None:
        return None
    return fc_df.groupby('date')['predicted_sales'].sum().reset_index()

ada_monthly = agg_monthly(data['ada_fc'])
cat_monthly = agg_monthly(data['cat_fc'])

# ═══════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════
def format_currency(val):
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    elif val >= 1e6:
        return f"${val/1e6:.2f}M"
    else:
        return f"${val:,.0f}"

ada_total = format_currency(data['ada_fc']['predicted_sales'].sum()) if data['ada_fc'] is not None else "N/A"
cat_total = format_currency(data['cat_fc']['predicted_sales'].sum()) if data['cat_fc'] is not None else "N/A"
n_stores = str(data['ada_fc']['store_id'].nunique()) if data['ada_fc'] is not None else (
    str(data['cat_fc']['store_id'].nunique()) if data['cat_fc'] is not None else "N/A")
n_months = str(data['ada_fc']['date'].nunique()) if data['ada_fc'] is not None else (
    str(data['cat_fc']['date'].nunique()) if data['cat_fc'] is not None else "N/A")



c1, c2, c3, c4 = st.columns(4)
with c1: _kpi(n_stores, "Stores Forecasted")
with c2: _kpi(f"{n_months} Months", "Forecast Horizon")
with c3: _kpi(ada_total, "AdaBoost Total (Full Year)")
with c4: _kpi(cat_total, "CatBoost Total (Full Year)")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab_overview, tab_stores, tab_compare = st.tabs([
    "📊 Historical + Forecast Trend", "🏪 Store-Level Analysis", "⚖️ Model Comparison"
])

# ── TAB 1: Historical + 12-Month Forecast Trend ──
with tab_overview:
    st.markdown('<div class="section-header">📊 Total Monthly Sales: Historical + 12-Month Forecast</div>', unsafe_allow_html=True)
    st.info("💡 **Long-term forecasting** predicts **12 months ahead** (Jan–Dec 2026) using "
            "recursive step-by-step prediction. Both models are trained in a single notebook (`cat&ada.ipynb`).")
    fig = go.Figure()
    if data['historical'] is not None:
        hist = data['historical']
        fig.add_trace(go.Scatter(x=hist['date'], y=hist['sales'], mode='lines+markers',
                                  name='Historical', line=dict(color=HIST_COLOR, width=2),
                                  marker=dict(size=4)))
    # Show 12-month forecast as TREND LINES with text annotations
    if ada_monthly is not None:
        ada_text = [f"{v/1e6:.1f}M" for v in ada_monthly['predicted_sales']]
        fig.add_trace(go.Scatter(x=ada_monthly['date'], y=ada_monthly['predicted_sales'],
                                  mode='lines+markers+text', name='AdaBoost 12-Month Forecast',
                                  line=dict(color=ADA_COLOR, width=3, dash='dash'),
                                  marker=dict(size=7, symbol='square'),
                                  text=ada_text, textposition="top center", textfont=dict(size=10)))
    if cat_monthly is not None:
        cat_text = [f"{v/1e6:.1f}M" for v in cat_monthly['predicted_sales']]
        fig.add_trace(go.Scatter(x=cat_monthly['date'], y=cat_monthly['predicted_sales'],
                                  mode='lines+markers+text', name='CatBoost 12-Month Forecast',
                                  line=dict(color=CAT_COLOR, width=3, dash='dot'),
                                  marker=dict(size=7, symbol='triangle-up'),
                                  text=cat_text, textposition="bottom center", textfont=dict(size=10)))
    # Shade forecast region with subtle color
    if ada_monthly is not None and len(ada_monthly) > 1:
        fig.add_vrect(x0=ada_monthly['date'].min(), x1=ada_monthly['date'].max(),
                      fillcolor="orange", opacity=0.03, layer="below", line_width=0,
                      annotation_text="12-Month Forecast", annotation_position="top left")
    elif cat_monthly is not None and len(cat_monthly) > 1:
        fig.add_vrect(x0=cat_monthly['date'].min(), x1=cat_monthly['date'].max(),
                      fillcolor="orange", opacity=0.03, layer="below", line_width=0,
                      annotation_text="12-Month Forecast", annotation_position="top left")
    fig.update_layout(
        title="Historical vs 12-Month Recursive Forecast — Total Monthly Sales",
        xaxis=dict(
            range=[hist['date'].min() if data['historical'] is not None else '2021-01-01', '2027-01-15']
        )
    )
    style_fig(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly Breakdown Bars
    st.markdown('<div class="section-header">📅 Monthly Predicted Sales (2026) — AdaBoost vs CatBoost</div>', unsafe_allow_html=True)
    bar_data = []
    if ada_monthly is not None:
        for _, row in ada_monthly.iterrows():
            bar_data.append({'Month': row['date'].strftime('%Y-%m'), 'Sales': row['predicted_sales'], 'Model': 'AdaBoost'})
    if cat_monthly is not None:
        for _, row in cat_monthly.iterrows():
            bar_data.append({'Month': row['date'].strftime('%Y-%m'), 'Sales': row['predicted_sales'], 'Model': 'CatBoost'})
    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        fig2 = px.bar(bar_df, x='Month', y='Sales', color='Model', barmode='group',
                       title="Monthly Total Predicted Sales (2026) — AdaBoost vs CatBoost",
                       color_discrete_map={'AdaBoost': ADA_COLOR, 'CatBoost': CAT_COLOR},
                       text_auto='.3s')
        fig2.update_traces(textposition='outside', textfont_size=9)
        fig2.update_layout(xaxis=dict(tickangle=-45, tickformat="%b %Y", dtick="M1"))
        style_fig(fig2, 420)
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2: Store-Level Analysis ──
with tab_stores:
    st.markdown('<div class="section-header">🏪 Top & Bottom Stores by Predicted Annual Sales (2026)</div>', unsafe_allow_html=True)
    model_choice = st.radio("Select Model:", ["AdaBoost", "CatBoost"], horizontal=True, key="lt_model")
    fc = data['ada_fc'] if model_choice == "AdaBoost" else data['cat_fc']
    metrics = data['ada_m'] if model_choice == "AdaBoost" else data['cat_m']
    color = ADA_COLOR if model_choice == "AdaBoost" else CAT_COLOR

    if fc is not None:
        name_col = 'store_name' if 'store_name' in fc.columns else 'store_id'
        store_totals = fc.groupby(name_col)['predicted_sales'].sum().sort_values(ascending=False)
        top10 = store_totals.head(10)
        bot10 = store_totals.tail(10)

        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.bar(x=top10.values, y=top10.index, orientation='h',
                           title=f"Top 10 Stores — {model_choice} (Full Year 2026)",
                           text=[f"${v/1e6:.2f}M" for v in top10.values],
                           color_discrete_sequence=[color])
            fig3.update_traces(textposition='outside')
            fig3.update_layout(yaxis=dict(autorange="reversed"))
            style_fig(fig3, 420)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.bar(x=bot10.values, y=bot10.index, orientation='h',
                           title=f"Bottom 10 Stores — {model_choice} (Full Year 2026)",
                           text=[f"${v/1e6:.2f}M" for v in bot10.values],
                           color_discrete_sequence=["#fb923c"])
            fig4.update_traces(textposition='outside')
            fig4.update_layout(yaxis=dict(autorange="reversed"))
            style_fig(fig4, 420)
            st.plotly_chart(fig4, use_container_width=True)

        # Store-level monthly trend (12 months)
        st.markdown('<div class="section-header">📈 12-Month Forecast Trend for Selected Stores</div>', unsafe_allow_html=True)
        all_stores = sorted(fc[name_col].unique())
        selected_stores = st.multiselect("Select stores to compare monthly trends:", all_stores,
                                          default=all_stores[:3] if len(all_stores) >= 3 else all_stores, key="lt_stores")
        if selected_stores:
            sub = fc[fc[name_col].isin(selected_stores)]
            fig5 = px.line(sub, x='date', y='predicted_sales', color=name_col,
                            markers=True, title=f"{model_choice} — Monthly Forecast Trend by Store (2026)")
            fig5.update_layout(xaxis=dict(tickangle=-45, tickformat="%b %Y", dtick="M1"))
            style_fig(fig5, 400)
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning(f"No {model_choice} long-term forecast data available. "
                   f"Please run the cat&ada.ipynb notebook and export the forecast CSV.")

    # Store metrics table
    if metrics is not None:
        st.markdown('<div class="section-header">📋 Per-Store Test Metrics</div>', unsafe_allow_html=True)
        st.dataframe(metrics.sort_values('R2', ascending=False), use_container_width=True, height=400)

    # Download button
    if fc is not None:
        st.download_button(f"📥 Download {model_choice} Long-Term Forecast CSV", fc.to_csv(index=False),
                           f"long_term_{model_choice.lower()}_forecast.csv")

# ── TAB 3: Model Comparison ──
with tab_compare:
    st.markdown('<div class="section-header">⚖️ AdaBoost vs CatBoost — 12-Month Forecast Comparison</div>', unsafe_allow_html=True)
    if data['ada_m'] is not None and data['cat_m'] is not None:
        ada_m = data['ada_m'].rename(columns={'MAE': 'AdaBoost_MAE', 'R2': 'AdaBoost_R2'})
        cat_m = data['cat_m'].rename(columns={'MAE': 'CatBoost_MAE', 'R2': 'CatBoost_R2'})
        merged = ada_m[['store_id', 'store_name', 'AdaBoost_MAE', 'AdaBoost_R2']].merge(
            cat_m[['store_id', 'CatBoost_MAE', 'CatBoost_R2']], on='store_id', how='outer')
        merged['Better_Model'] = merged.apply(
            lambda r: 'AdaBoost' if r.get('AdaBoost_R2', 0) >= r.get('CatBoost_R2', 0) else 'CatBoost', axis=1)
        st.dataframe(merged.sort_values('AdaBoost_R2', ascending=False), use_container_width=True, height=400)

        # R² distribution comparison
        c1, c2 = st.columns(2)
        with c1:
            fig6 = go.Figure()
            fig6.add_trace(go.Histogram(x=data['ada_m']['R2'], nbinsx=25, name='AdaBoost R²',
                                         marker_color=ADA_COLOR, opacity=0.7))
            fig6.add_trace(go.Histogram(x=data['cat_m']['R2'], nbinsx=25, name='CatBoost R²',
                                         marker_color=CAT_COLOR, opacity=0.7))
            fig6.update_layout(title="R² Distribution by Store", barmode='overlay')
            style_fig(fig6, 350)
            st.plotly_chart(fig6, use_container_width=True)
        with c2:
            fig7 = go.Figure()
            fig7.add_trace(go.Histogram(x=data['ada_m']['MAE'], nbinsx=25, name='AdaBoost MAE',
                                         marker_color=ADA_COLOR, opacity=0.7))
            fig7.add_trace(go.Histogram(x=data['cat_m']['MAE'], nbinsx=25, name='CatBoost MAE',
                                         marker_color=CAT_COLOR, opacity=0.7))
            fig7.update_layout(title="MAE Distribution by Store", barmode='overlay')
            style_fig(fig7, 350)
            st.plotly_chart(fig7, use_container_width=True)

        # Summary KPIs
        st.markdown('<div class="section-header">📊 Overall Summary (Test Set)</div>', unsafe_allow_html=True)
        ada_wins = (merged['Better_Model'] == 'AdaBoost').sum()
        cat_wins = (merged['Better_Model'] == 'CatBoost').sum()
        
        # Using exact tuned overall metrics from cat&ada.ipynb notebook
        ada_r2_val = 0.9032
        ada_mae_val = 94796
        cat_r2_val = 0.9784
        cat_mae_val = 39896
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: _kpi(f"{ada_r2_val:.4f}", "AdaBoost Overall R²")
        with c2: _kpi(f"${ada_mae_val:,.0f}", "AdaBoost Overall MAE")
        with c3: _kpi(f"{cat_r2_val:.4f}", "CatBoost Overall R²")
        with c4: _kpi(f"${cat_mae_val:,.0f}", "CatBoost Overall MAE")
        st.markdown("")
        c1, c2 = st.columns(2)
        with c1: _kpi(str(ada_wins), "Stores Where AdaBoost Wins")
        with c2: _kpi(str(cat_wins), "Stores Where CatBoost Wins")
    elif data['ada_fc'] is not None or data['cat_fc'] is not None:
        # If we have forecast data but no metrics, show a basic comparison
        st.info("Per-store metrics are not yet available. Run cat&ada.ipynb and export store metrics to see detailed comparisons.")
        if data['ada_fc'] is not None and data['cat_fc'] is not None:
            ada_store_totals = data['ada_fc'].groupby('store_id')['predicted_sales'].sum().reset_index()
            ada_store_totals.columns = ['store_id', 'AdaBoost_Annual']
            cat_store_totals = data['cat_fc'].groupby('store_id')['predicted_sales'].sum().reset_index()
            cat_store_totals.columns = ['store_id', 'CatBoost_Annual']
            comp = ada_store_totals.merge(cat_store_totals, on='store_id', how='outer')
            comp['Difference'] = comp['AdaBoost_Annual'] - comp['CatBoost_Annual']
            comp['Higher_Model'] = comp.apply(
                lambda r: 'AdaBoost' if r['AdaBoost_Annual'] >= r['CatBoost_Annual'] else 'CatBoost', axis=1)
            st.dataframe(comp.sort_values('AdaBoost_Annual', ascending=False), use_container_width=True, height=400)
    else:
        st.warning("Both AdaBoost and CatBoost data are needed for comparison. "
                   "Please run the cat&ada.ipynb notebook and export forecast data.")
