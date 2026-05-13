# -*- coding: utf-8 -*-
"""Page 3 — Short-Term Forecasting (One-Month Ahead)
Source notebooks: ADABOOST.ipynb & CATBOOST.ipynb (separate models)
Each notebook independently trains its own model and forecasts one month (Jan 2026).
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

st.markdown("# 📊 Short-Term Forecasting")
st.markdown("*One-month-ahead forecast (Jan 2026) using independent AdaBoost & CatBoost models.*")
st.caption("Source: `ADABOOST.ipynb` and `CATBOOST.ipynb` — each notebook trains its own model separately.")

def _kpi(v, l):
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{v}</div>'
                f'<div class="kpi-label">{l}</div></div>', unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading short-term forecast data...")
def load_all():
    data = {}
    # Short-term forecasts (1 month ahead — Jan 2026)
    for key, fname in [
        ('ada_fc', 'adaboost_forecast_jan2025.csv'),
        ('cat_fc', 'catboost_forecast_jan2026.csv'),
    ]:
        p = PROC / fname
        if p.exists():
            df = pd.read_csv(p)
            df['date'] = pd.to_datetime(df['date'])
            data[key] = df
        else:
            data[key] = None
    # Store metrics
    for key, fname in [
        ('ada_m', 'adaboost_store_metrics.csv'),
        ('cat_m', 'catboost_store_metrics.csv'),
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

        store_hist = raw.groupby(['store_name', raw['sale_date'].dt.to_period('M')])['sales_amount_realistic'].sum().reset_index()
        store_hist.columns = ['store_name', 'date', 'sales']
        store_hist['date'] = store_hist['date'].dt.to_timestamp()
        data['store_historical'] = store_hist
    else:
        data['historical'] = None
        data['store_historical'] = None
    return data

data = load_all()

if data['ada_fc'] is None and data['cat_fc'] is None:
    st.error("No short-term forecast CSV files found in data/processed/. "
             "Please run the ADABOOST.ipynb and CATBOOST.ipynb notebooks first.")
    st.stop()

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
forecast_period = "1 Month"

c1, c2, c3, c4 = st.columns(4)
with c1: _kpi(n_stores, "Stores Forecasted")
with c2: _kpi(forecast_period, "Forecast Horizon")
with c3: _kpi(ada_total, "AdaBoost Predicted (Jan 2026)")
with c4: _kpi(cat_total, "CatBoost Predicted (Jan 2026)")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab_overview, tab_stores, tab_compare = st.tabs([
    "📊 Historical + Forecast Point", "🏪 Store-Level Analysis", "⚖️ Model Comparison"
])

# ── TAB 1: Historical + Single-Point Forecast ──
with tab_overview:
    st.markdown('<div class="section-header">📊 Total Monthly Sales: Historical + One-Month Forecast</div>', unsafe_allow_html=True)
    st.info("💡 **Short-term forecasting** predicts **one month ahead** (January 2026). "
            "Each model (AdaBoost & CatBoost) is trained in its own separate notebook.")
    fig = go.Figure()
    if data['historical'] is not None:
        hist = data['historical']
        fig.add_trace(go.Scatter(x=hist['date'], y=hist['sales'], mode='lines+markers',
                                  name='Historical', line=dict(color=HIST_COLOR, width=2),
                                  marker=dict(size=4)))
    # Show forecast as single large point(s) at the end
    if data['ada_fc'] is not None:
        ada_date = data['ada_fc']['date'].iloc[0]
        ada_sum = data['ada_fc']['predicted_sales'].sum()
        fig.add_trace(go.Scatter(x=[ada_date], y=[ada_sum],
                                  mode='markers+text', name='AdaBoost Forecast (Jan 2026)',
                                  marker=dict(color=ADA_COLOR, size=16, symbol='square',
                                              line=dict(width=2, color='white')),
                                  text=[f"${ada_sum/1e6:.1f}M"], textposition='top center',
                                  textfont=dict(color=ADA_COLOR, size=12, family='Inter')))
    if data['cat_fc'] is not None:
        cat_date = data['cat_fc']['date'].iloc[0]
        cat_sum = data['cat_fc']['predicted_sales'].sum()
        fig.add_trace(go.Scatter(x=[cat_date], y=[cat_sum],
                                  mode='markers+text', name='CatBoost Forecast (Jan 2026)',
                                  marker=dict(color=CAT_COLOR, size=16, symbol='triangle-up',
                                              line=dict(width=2, color='white')),
                                  text=[f"${cat_sum/1e6:.1f}M"], textposition='bottom center',
                                  textfont=dict(color=CAT_COLOR, size=12, family='Inter')))
    # Add a vertical line at the forecast date
    forecast_date = None
    if data['ada_fc'] is not None:
        forecast_date = data['ada_fc']['date'].iloc[0]
    elif data['cat_fc'] is not None:
        forecast_date = data['cat_fc']['date'].iloc[0]
    if forecast_date is not None:
        fc_str = str(forecast_date)
        fig.add_shape(type="line", x0=fc_str, x1=fc_str, y0=0, y1=1,
                      yref="paper", line=dict(color="orange", dash="dash", width=1), opacity=0.6)
        fig.add_annotation(x=fc_str, y=1, yref="paper", text="Forecast Point",
                           showarrow=False, font=dict(color="orange"), yanchor="bottom")
    fig.update_layout(title="Historical Monthly Sales + One-Month-Ahead Forecast (Jan 2026)")
    style_fig(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    # Per-store comparison bar chart
    st.markdown('<div class="section-header">🏬 Per-Store Predicted Sales — AdaBoost vs CatBoost</div>', unsafe_allow_html=True)
    bar_data = []
    if data['ada_fc'] is not None:
        for _, row in data['ada_fc'].iterrows():
            bar_data.append({'Store': row.get('store_name', row.get('store_id', '')),
                             'Predicted Sales': row['predicted_sales'], 'Model': 'AdaBoost'})
    if data['cat_fc'] is not None:
        for _, row in data['cat_fc'].iterrows():
            bar_data.append({'Store': row.get('store_name', row.get('store_id', '')),
                             'Predicted Sales': row['predicted_sales'], 'Model': 'CatBoost'})
    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        # Sort by predicted sales (AdaBoost) for better readability
        store_order = bar_df[bar_df['Model'] == (
            'AdaBoost' if data['ada_fc'] is not None else 'CatBoost')].sort_values(
            'Predicted Sales', ascending=True)['Store'].tolist()
        fig2 = px.bar(bar_df, x='Predicted Sales', y='Store', color='Model', barmode='group',
                       orientation='h',
                       title="Per-Store Predicted Sales — Jan 2026 (AdaBoost vs CatBoost)",
                       color_discrete_map={'AdaBoost': ADA_COLOR, 'CatBoost': CAT_COLOR},
                       category_orders={'Store': store_order})
        style_fig(fig2, max(500, len(store_order) * 18))
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2: Store-Level Analysis ──
with tab_stores:
    st.markdown('<div class="section-header">🏪 Top & Bottom Stores by Predicted Sales (Jan 2026)</div>', unsafe_allow_html=True)
    model_choice = st.radio("Select Model:", ["AdaBoost", "CatBoost"], horizontal=True, key="st_model")
    fc = data['ada_fc'] if model_choice == "AdaBoost" else data['cat_fc']
    metrics = data['ada_m'] if model_choice == "AdaBoost" else data['cat_m']
    color = ADA_COLOR if model_choice == "AdaBoost" else CAT_COLOR

    if fc is not None:
        top10_df = fc.nlargest(10, 'predicted_sales').sort_values('predicted_sales', ascending=True)
        bot10_df = fc.nsmallest(10, 'predicted_sales').sort_values('predicted_sales', ascending=True)

        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.bar(top10_df, x='predicted_sales', y='store_name', orientation='h',
                           title=f"Top 10 Stores — {model_choice} (Jan 2026)",
                           text=[f"${v/1e6:.2f}M" for v in top10_df['predicted_sales']],
                           color='predicted_sales', color_continuous_scale="viridis")
            fig3.update_traces(textposition='outside')
            fig3.update_layout(coloraxis_showscale=False)
            style_fig(fig3, 420)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.bar(bot10_df, x='predicted_sales', y='store_name', orientation='h',
                           title=f"Bottom 10 Stores — {model_choice} (Jan 2026)",
                           text=[f"${v/1e6:.2f}M" for v in bot10_df['predicted_sales']],
                           color='predicted_sales', color_continuous_scale="viridis")
            fig4.update_traces(textposition='outside')
            fig4.update_layout(coloraxis_showscale=False)
            style_fig(fig4, 420)
            st.plotly_chart(fig4, use_container_width=True)

        # Store-level monthly trend (Historical + 1-Month)
        st.markdown('<div class="section-header">📈 Historical + 1-Month Forecast Trend for Selected Stores</div>', unsafe_allow_html=True)
        store_hist = data.get('store_historical')
        if store_hist is not None:
            all_stores = sorted(store_hist['store_name'].unique())
            selected_stores = st.multiselect("Select stores to compare monthly trends:", all_stores,
                                              default=all_stores[:3] if len(all_stores) >= 3 else all_stores, key="st_stores_trend")
            if selected_stores:
                fig5 = go.Figure()
                for s in selected_stores:
                    s_hist = store_hist[store_hist['store_name'] == s].sort_values('date')
                    fig5.add_trace(go.Scatter(x=s_hist['date'], y=s_hist['sales'], mode='lines+markers', name=f"{s} (Actual)", marker=dict(size=4)))
                    # Add forecast point
                    s_fc = fc[fc['store_name'] == s] if 'store_name' in fc.columns else fc[fc['store_id'] == s]
                    if not s_fc.empty:
                        fig5.add_trace(go.Scatter(x=[s_hist['date'].max(), s_fc['date'].iloc[0]], 
                                                  y=[s_hist['sales'].iloc[-1], s_fc['predicted_sales'].iloc[0]], 
                                                  mode='lines+markers+text', name=f"{s} (Forecast)", line=dict(dash='dash'),
                                                  text=["", f"{s_fc['predicted_sales'].iloc[0]/1e6:.2f}M"], textposition="top center"))
                fig5.update_layout(title="Historical vs 1-Month Forecast Trend", xaxis_title="Month", yaxis_title="Sales")
                style_fig(fig5, 450)
                st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning(f"No {model_choice} short-term forecast data available. "
                   f"Please run the {model_choice.upper()}.ipynb notebook first.")

    # Store metrics table
    if metrics is not None:
        st.markdown('<div class="section-header">📋 Per-Store Test Metrics (from notebook evaluation)</div>', unsafe_allow_html=True)
        st.dataframe(metrics.sort_values('R2', ascending=False), use_container_width=True, height=400)

    # Download button
    if fc is not None:
        st.download_button(f"📥 Download {model_choice} Short-Term Forecast CSV", fc.to_csv(index=False),
                           f"short_term_{model_choice.lower()}_forecast.csv")

# ── TAB 3: Model Comparison ──
with tab_compare:
    st.markdown('<div class="section-header">⚖️ AdaBoost vs CatBoost — One-Month Forecast Comparison</div>', unsafe_allow_html=True)
    if data['ada_m'] is not None and data['cat_m'] is not None:
        ada_m = data['ada_m'].rename(columns={'MAE': 'AdaBoost_MAE', 'R2': 'AdaBoost_R2'})
        cat_m = data['cat_m'].rename(columns={'MAE': 'CatBoost_MAE', 'R2': 'CatBoost_R2'})
        merged = ada_m[['store_id', 'store_name', 'AdaBoost_MAE', 'AdaBoost_R2']].merge(
            cat_m[['store_id', 'CatBoost_MAE', 'CatBoost_R2']], on='store_id', how='outer')
        merged['Better_Model'] = merged.apply(
            lambda r: 'AdaBoost' if r.get('AdaBoost_R2', 0) >= r.get('CatBoost_R2', 0) else 'CatBoost', axis=1)
        st.dataframe(merged.sort_values('AdaBoost_R2', ascending=False), use_container_width=True, height=400)

        # R² and MAE distribution comparison
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
        
        # Using overall metrics from ADABOOST.ipynb and CATBOOST.ipynb tuning results
        ada_r2_overall = 0.9017
        ada_mae_overall = 93293
        cat_r2_overall = 0.9778
        cat_mae_overall = 40264
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: _kpi(f"{ada_r2_overall:.4f}", "AdaBoost Overall R²")
        with c2: _kpi(f"${ada_mae_overall:,.0f}", "AdaBoost Overall MAE")
        with c3: _kpi(f"{cat_r2_overall:.4f}", "CatBoost Overall R²")
        with c4: _kpi(f"${cat_mae_overall:,.0f}", "CatBoost Overall MAE")
        st.markdown("")
        c1, c2 = st.columns(2)
        with c1: _kpi(str(ada_wins), "Stores Where AdaBoost Wins")
        with c2: _kpi(str(cat_wins), "Stores Where CatBoost Wins")
    else:
        st.warning("Both AdaBoost and CatBoost metrics are needed for comparison. "
                   "Please run both ADABOOST.ipynb and CATBOOST.ipynb notebooks.")
