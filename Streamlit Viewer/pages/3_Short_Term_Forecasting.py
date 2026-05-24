# -*- coding: utf-8 -*-
"""Page 3 — Short-Term Forecasting (1-Month Ahead)
Source models: CAT_LIVE.cbm & ADA_LIVE.joblib
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings, joblib
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Short-Term Forecasting", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT = APP_DIR.parent
PROC = PROJECT / "data" / "processed"

def _find(fn):
    for p in [Path("/notebooks")/fn, APP_DIR/fn, APP_DIR/"notebooks"/fn, PROJECT/"notebooks"/fn]:
        if p.exists(): return p
    return None

CAT_PATH = _find("CAT_LIVE.cbm")
ADA_PATH = _find("ADA_LIVE.joblib")
DATA_PATH = PROC / "cleaned_apple_sales_v3.csv"

CHART_BG="rgba(28,28,30,0.6)"; PAPER_BG="rgba(0,0,0,0)"
FONT_COLOR="#E1E1E6"; GRID_COLOR="rgba(0,240,255,0.10)"
ADA_COLOR="#00F0FF"; CAT_COLOR="#39FF14"; HIST_COLOR="#A0A0A5"
PALETTE=["#00F0FF","#39FF14","#FF5252","#5E5CE6","#FFFFFF","#00B4FF","#A0A0A5","#FFD60A"]
TARGET="sales_amount_realistic"
SAFE_LAGS=[1,2,3,6,12]; ROLL_WINDOWS=[3,6]
FEATURES=[
    'sales_lag_1','sales_lag_2','sales_lag_3','sales_lag_6','sales_lag_12',
    'sales_roll_mean_3','sales_roll_mean_6','sales_mom_pct','sales_lag1_vs_roll6',
    'price_realistic','promo_flag','month_sin','month_cos',
    'is_holiday_season','is_launch_season',
    'gdp_per_capita','inflation_rate','exchange_rate','internet_usage_pct',
    'gdp_change','inflation_change','exchange_change','internet_usage_change',
    'store_encoded','num_transactions','num_unique_products','num_categories','year',
]

def style_fig(fig, h=420):
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
        height=h, margin=dict(l=16,r=16,t=42,b=16),
        legend=dict(bgcolor="rgba(28,28,30,0.85)", bordercolor="rgba(0,240,255,0.3)", borderwidth=1),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="rgba(0,240,255,0.2)"),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="rgba(0,240,255,0.2)"), colorway=PALETTE)

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
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("# 📊 Short-Term Forecasting")
st.markdown("*One-month-ahead forecast using live AdaBoost & CatBoost models (no overfitting).*")
st.caption("Source: `CAT_LIVE.ipynb` & `ADA_LIVE.ipynb` — trained models with healthy generalization.")

def _kpi(v, l):
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{v}</div>'
                f'<div class="kpi-label">{l}</div></div>', unsafe_allow_html=True)

def format_currency(val):
    if val >= 1e9: return f"${val/1e9:.2f}B"
    elif val >= 1e6: return f"${val/1e6:.2f}M"
    elif val >= 1e3: return f"${val/1e3:.1f}K"
    return f"${val:,.0f}"

# ─── Load Models ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {}
    if CAT_PATH and CAT_PATH.exists():
        from catboost import CatBoostRegressor
        cb = CatBoostRegressor(); cb.load_model(str(CAT_PATH))
        models["CatBoost"] = cb
    if ADA_PATH and ADA_PATH.exists():
        models["AdaBoost"] = joblib.load(ADA_PATH)
    return models

@st.cache_data(show_spinner="Loading & preprocessing data...")
def load_and_preprocess():
    from sklearn.preprocessing import LabelEncoder
    df_raw = pd.read_csv(DATA_PATH)
    df_raw["sale_date"] = pd.to_datetime(df_raw["sale_date"])
    df_raw["year"] = df_raw["sale_date"].dt.year
    df_raw["month"] = df_raw["sale_date"].dt.month
    if "country_norm_mapped" not in df_raw.columns:
        df_raw["country_norm_mapped"] = df_raw["country"].str.lower().str.strip()
    monthly_aggs = {"sales_amount_realistic":"sum","quantity_realistic":"sum",
        "price_realistic":"mean","store_name":"first","country_norm_mapped":"first","promo_flag":"mean"}
    for ecol in ["gdp_per_capita","inflation_rate","exchange_rate","internet_usage_pct"]:
        if ecol in df_raw.columns: monthly_aggs[ecol] = "first"
    df_monthly = df_raw.groupby(["store_id","year","month"]).agg(monthly_aggs).reset_index()
    grp = df_raw.groupby(["store_id","year","month"])
    df_monthly["num_transactions"] = grp.size().values
    df_monthly["num_unique_products"] = grp["product_id"].nunique().values
    if "category_id" in df_raw.columns:
        df_monthly["num_categories"] = grp["category_id"].nunique().values
    df_monthly["date"] = pd.to_datetime(df_monthly[["year","month"]].assign(day=1))
    df_monthly.sort_values(["store_id","date"], inplace=True)
    df_monthly.reset_index(drop=True, inplace=True)
    for lag in SAFE_LAGS:
        df_monthly[f"sales_lag_{lag}"] = df_monthly.groupby("store_id")[TARGET].shift(lag)
    for w in ROLL_WINDOWS:
        rolled = df_monthly.groupby("store_id")[TARGET].shift(1).rolling(w, min_periods=1)
        df_monthly[f"sales_roll_mean_{w}"] = rolled.mean().reset_index(0, drop=True)
    df_monthly["sales_mom_pct"] = df_monthly.groupby("store_id")[TARGET].pct_change()
    df_monthly["sales_lag1_vs_roll6"] = (
        df_monthly["sales_lag_1"] / df_monthly["sales_roll_mean_6"].replace(0, np.nan))
    df_monthly["month_sin"] = np.sin(2*np.pi*df_monthly["month"]/12)
    df_monthly["month_cos"] = np.cos(2*np.pi*df_monthly["month"]/12)
    df_monthly["is_holiday_season"] = df_monthly["month"].isin([11,12]).astype(int)
    df_monthly["is_launch_season"] = df_monthly["month"].isin([9,10]).astype(int)
    for col in ["gdp_per_capita","inflation_rate","exchange_rate","internet_usage_pct"]:
        if col in df_monthly.columns:
            new = col.replace("_per_capita","").replace("_rate","").replace("_pct","") + "_change"
            df_monthly[new] = df_monthly.groupby("store_id")[col].pct_change().fillna(0)
    le = LabelEncoder()
    df_monthly["store_encoded"] = le.fit_transform(df_monthly["store_id"])
    df_monthly["row_num"] = df_monthly.groupby("store_id").cumcount()
    df_clean = df_monthly[df_monthly["row_num"] >= 12].copy()
    df_clean.drop(columns="row_num", inplace=True)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    # Historical
    hist = df_raw.groupby(df_raw['sale_date'].dt.to_period('M'))['sales_amount_realistic'].sum().reset_index()
    hist.columns = ['date','sales']; hist['date'] = hist['date'].dt.to_timestamp()
    # Store-level historical
    store_hist = df_raw.groupby(['store_name', df_raw['sale_date'].dt.to_period('M')])['sales_amount_realistic'].sum().reset_index()
    store_hist.columns = ['store_name','date','sales']; store_hist['date'] = store_hist['date'].dt.to_timestamp()
    return df_clean, hist, store_hist

# ─── Recursive forecast ──────────────────────────────────────────────
def recursive_forecast(model, store_history, n_months, features):
    history = store_history.copy(); predictions = []
    for step in range(n_months):
        last_row = history.iloc[-1]
        next_date = last_row["date"] + pd.DateOffset(months=1)
        new_row = {}; sales_series = history[TARGET].values
        for lag in SAFE_LAGS:
            new_row[f"sales_lag_{lag}"] = sales_series[-lag] if lag <= len(sales_series) else 0.0
        for w in ROLL_WINDOWS:
            wd = sales_series[-w:] if len(sales_series) >= w else sales_series
            new_row[f"sales_roll_mean_{w}"] = float(np.mean(wd))
        new_row["sales_mom_pct"] = ((sales_series[-1]-sales_series[-2])/sales_series[-2]
            if len(sales_series)>=2 and sales_series[-2]!=0 else 0.0)
        r6 = new_row.get("sales_roll_mean_6",1)
        new_row["sales_lag1_vs_roll6"] = new_row["sales_lag_1"]/r6 if r6!=0 else 0.0
        new_row["price_realistic"] = last_row.get("price_realistic",0)
        new_row["promo_flag"] = last_row.get("promo_flag",0)
        nm = next_date.month
        new_row["month_sin"]=np.sin(2*np.pi*nm/12); new_row["month_cos"]=np.cos(2*np.pi*nm/12)
        new_row["is_holiday_season"]=int(nm in [11,12]); new_row["is_launch_season"]=int(nm in [9,10])
        for col in ["gdp_per_capita","inflation_rate","exchange_rate","internet_usage_pct",
                     "gdp_change","inflation_change","exchange_change","internet_usage_change"]:
            new_row[col] = last_row.get(col,0)
        new_row["store_encoded"]=last_row.get("store_encoded",0)
        new_row["num_transactions"]=last_row.get("num_transactions",0)
        new_row["num_unique_products"]=last_row.get("num_unique_products",0)
        new_row["num_categories"]=last_row.get("num_categories",0)
        new_row["year"]=next_date.year
        valid_f=[f for f in features if f in new_row]
        X_new=pd.DataFrame([{f:new_row[f] for f in valid_f}])
        pred=max(float(model.predict(X_new)[0]),0)
        predictions.append({"date":next_date,"predicted_sales":pred})
        next_full=last_row.copy(); next_full["date"]=next_date
        next_full["year"]=next_date.year; next_full["month"]=nm; next_full[TARGET]=pred
        for k,v in new_row.items():
            if k in next_full.index: next_full[k]=v
        history=pd.concat([history,pd.DataFrame([next_full])],ignore_index=True)
    return pd.DataFrame(predictions)

@st.cache_data(show_spinner="Running 1-month forecasts for all stores...")
def forecast_all_stores(_model, df_clean, n_months, features, model_name):
    all_fc = []
    for sid in df_clean["store_id"].unique():
        sdata = df_clean[df_clean["store_id"]==sid].copy()
        if len(sdata)<13: continue
        fc = recursive_forecast(_model, sdata, n_months, features)
        fc["store_id"]=sid
        fc["store_name"]=sdata["store_name"].iloc[0] if "store_name" in sdata.columns else sid
        fc["country"]=sdata["country_norm_mapped"].iloc[0] if "country_norm_mapped" in sdata.columns else ""
        all_fc.append(fc)
    return pd.concat(all_fc, ignore_index=True) if all_fc else None

# ═══════════════════════════════════════════════════════════════════════
models = load_models()
if not models:
    st.warning("⚠️ **Model files not found.** Please ensure `CAT_LIVE.cbm` and `ADA_LIVE.joblib` "
               "are in the `notebooks/` folder.")
    st.stop()

df_clean, historical, store_historical = load_and_preprocess()
HORIZON = 1

ada_fc = forecast_all_stores(models["AdaBoost"], df_clean, HORIZON, FEATURES, "AdaBoost") if "AdaBoost" in models else None
cat_fc = forecast_all_stores(models["CatBoost"], df_clean, HORIZON, FEATURES, "CatBoost") if "CatBoost" in models else None

if ada_fc is None and cat_fc is None:
    st.error("No forecasts could be generated."); st.stop()

# Determine forecast month label
fc_ref = ada_fc if ada_fc is not None else cat_fc
forecast_month_label = fc_ref['date'].iloc[0].strftime('%b %Y')

# ═══════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════
ada_total = format_currency(ada_fc['predicted_sales'].sum()) if ada_fc is not None else "N/A"
cat_total = format_currency(cat_fc['predicted_sales'].sum()) if cat_fc is not None else "N/A"
n_stores = str(ada_fc['store_id'].nunique()) if ada_fc is not None else (
    str(cat_fc['store_id'].nunique()) if cat_fc is not None else "N/A")

c1,c2,c3,c4 = st.columns(4)
with c1: _kpi(n_stores, "Stores Forecasted")
with c2: _kpi("1 Month", "Forecast Horizon")
with c3: _kpi(ada_total, f"AdaBoost ({forecast_month_label})")
with c4: _kpi(cat_total, f"CatBoost ({forecast_month_label})")
st.markdown("")

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab_overview, tab_stores, tab_compare = st.tabs([
    "📊 Historical + Forecast Point", "🏪 Store-Level Analysis", "⚖️ Model Comparison"
])

# ── TAB 1 ──
with tab_overview:
    st.markdown('<div class="section-header">📊 Total Monthly Sales: Historical + One-Month Forecast</div>', unsafe_allow_html=True)
    st.info("💡 **Short-term forecasting** predicts **one month ahead**. "
            "Models from `CAT_LIVE.ipynb` & `ADA_LIVE.ipynb` (no overfitting).")
    fig = go.Figure()
    if historical is not None:
        fig.add_trace(go.Scatter(x=historical['date'], y=historical['sales'], mode='lines+markers',
                                  name='Historical', line=dict(color=HIST_COLOR, width=2), marker=dict(size=4)))
    if ada_fc is not None:
        ada_date = ada_fc['date'].iloc[0]; ada_sum = ada_fc['predicted_sales'].sum()
        fig.add_trace(go.Scatter(x=[ada_date], y=[ada_sum],
            mode='markers+text', name=f'AdaBoost ({forecast_month_label})',
            marker=dict(color=ADA_COLOR, size=16, symbol='square', line=dict(width=2, color='white')),
            text=[f"${ada_sum/1e6:.1f}M"], textposition='top center',
            textfont=dict(color=ADA_COLOR, size=12, family='Inter')))
    if cat_fc is not None:
        cat_date = cat_fc['date'].iloc[0]; cat_sum = cat_fc['predicted_sales'].sum()
        fig.add_trace(go.Scatter(x=[cat_date], y=[cat_sum],
            mode='markers+text', name=f'CatBoost ({forecast_month_label})',
            marker=dict(color=CAT_COLOR, size=16, symbol='triangle-up', line=dict(width=2, color='white')),
            text=[f"${cat_sum/1e6:.1f}M"], textposition='bottom center',
            textfont=dict(color=CAT_COLOR, size=12, family='Inter')))
    forecast_date = ada_fc['date'].iloc[0] if ada_fc is not None else cat_fc['date'].iloc[0]
    fc_str = str(forecast_date)
    fig.add_shape(type="line", x0=fc_str, x1=fc_str, y0=0, y1=1,
                  yref="paper", line=dict(color="orange", dash="dash", width=1), opacity=0.6)
    fig.add_annotation(x=fc_str, y=1, yref="paper", text="Forecast Point",
                       showarrow=False, font=dict(color="orange"), yanchor="bottom")
    fig.update_layout(title=f"Historical Monthly Sales + One-Month-Ahead Forecast ({forecast_month_label})")
    style_fig(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    # Per-store bar
    st.markdown('<div class="section-header">🏬 Per-Store Predicted Sales — AdaBoost vs CatBoost</div>', unsafe_allow_html=True)
    bar_data = []
    if ada_fc is not None:
        for _, row in ada_fc.iterrows():
            bar_data.append({'Store': row['store_name'], 'Predicted Sales': row['predicted_sales'], 'Model': 'AdaBoost'})
    if cat_fc is not None:
        for _, row in cat_fc.iterrows():
            bar_data.append({'Store': row['store_name'], 'Predicted Sales': row['predicted_sales'], 'Model': 'CatBoost'})
    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        ref_model = 'AdaBoost' if ada_fc is not None else 'CatBoost'
        store_order = bar_df[bar_df['Model']==ref_model].sort_values('Predicted Sales', ascending=True)['Store'].tolist()
        fig2 = px.bar(bar_df, x='Predicted Sales', y='Store', color='Model', barmode='group',
                       orientation='h', title=f"Per-Store Predicted Sales — {forecast_month_label}",
                       color_discrete_map={'AdaBoost': ADA_COLOR, 'CatBoost': CAT_COLOR},
                       category_orders={'Store': store_order})
        style_fig(fig2, max(500, len(store_order)*18))
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2 ──
with tab_stores:
    st.markdown(f'<div class="section-header">🏪 Top & Bottom Stores ({forecast_month_label})</div>', unsafe_allow_html=True)
    model_choice = st.radio("Select Model:", ["AdaBoost","CatBoost"], horizontal=True, key="st_model")
    fc = ada_fc if model_choice=="AdaBoost" else cat_fc
    color = ADA_COLOR if model_choice=="AdaBoost" else CAT_COLOR

    if fc is not None:
        top10 = fc.nlargest(10,'predicted_sales').sort_values('predicted_sales', ascending=True)
        bot10 = fc.nsmallest(10,'predicted_sales').sort_values('predicted_sales', ascending=True)
        c1,c2 = st.columns(2)
        with c1:
            fig3 = px.bar(top10, x='predicted_sales', y='store_name', orientation='h',
                title=f"Top 10 — {model_choice}", text=[f"${v/1e6:.2f}M" for v in top10['predicted_sales']],
                color='predicted_sales', color_continuous_scale="viridis")
            fig3.update_traces(textposition='outside'); fig3.update_layout(coloraxis_showscale=False)
            style_fig(fig3, 420); st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.bar(bot10, x='predicted_sales', y='store_name', orientation='h',
                title=f"Bottom 10 — {model_choice}", text=[f"${v/1e6:.2f}M" for v in bot10['predicted_sales']],
                color='predicted_sales', color_continuous_scale="viridis")
            fig4.update_traces(textposition='outside'); fig4.update_layout(coloraxis_showscale=False)
            style_fig(fig4, 420); st.plotly_chart(fig4, use_container_width=True)

        # Historical + 1-month trend
        st.markdown('<div class="section-header">📈 Historical + 1-Month Forecast Trend</div>', unsafe_allow_html=True)
        if store_historical is not None:
            all_stores = sorted(store_historical['store_name'].unique())
            selected_stores = st.multiselect("Select stores:", all_stores,
                default=all_stores[:3] if len(all_stores)>=3 else all_stores, key="st_stores_trend")
            if selected_stores:
                fig5 = go.Figure()
                for s in selected_stores:
                    s_hist = store_historical[store_historical['store_name']==s].sort_values('date')
                    fig5.add_trace(go.Scatter(x=s_hist['date'], y=s_hist['sales'], mode='lines+markers',
                        name=f"{s} (Actual)", marker=dict(size=4)))
                    s_fc = fc[fc['store_name']==s]
                    if not s_fc.empty:
                        fig5.add_trace(go.Scatter(
                            x=[s_hist['date'].max(), s_fc['date'].iloc[0]],
                            y=[s_hist['sales'].iloc[-1], s_fc['predicted_sales'].iloc[0]],
                            mode='lines+markers+text', name=f"{s} (Forecast)", line=dict(dash='dash'),
                            text=["", f"{s_fc['predicted_sales'].iloc[0]/1e6:.2f}M"], textposition="top center"))
                fig5.update_layout(title="Historical vs 1-Month Forecast Trend")
                style_fig(fig5, 450); st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning(f"No {model_choice} forecast available.")

    if fc is not None:
        st.download_button(f"📥 Download {model_choice} Short-Term Forecast CSV", fc.to_csv(index=False),
                           f"short_term_{model_choice.lower()}_forecast.csv")

# ── TAB 3 ──
with tab_compare:
    st.markdown('<div class="section-header">⚖️ AdaBoost vs CatBoost — One-Month Comparison</div>', unsafe_allow_html=True)
    if ada_fc is not None and cat_fc is not None:
        ada_s = ada_fc[['store_id','store_name','predicted_sales']].rename(columns={'predicted_sales':'AdaBoost_Sales'})
        cat_s = cat_fc[['store_id','predicted_sales']].rename(columns={'predicted_sales':'CatBoost_Sales'})
        merged = ada_s.merge(cat_s, on='store_id', how='outer')
        merged['Higher_Model'] = merged.apply(
            lambda r: 'AdaBoost' if r.get('AdaBoost_Sales',0) >= r.get('CatBoost_Sales',0) else 'CatBoost', axis=1)
        st.dataframe(merged.sort_values('CatBoost_Sales', ascending=False), use_container_width=True, height=400)

        st.markdown('<div class="section-header">📊 Overall Model Metrics (Test Set — No Overfitting)</div>', unsafe_allow_html=True)
        ada_r2=0.8944; ada_mae=114310; cat_r2=0.9766; cat_mae=41447
        c1,c2,c3,c4 = st.columns(4)
        with c1: _kpi(f"{ada_r2:.4f}", "AdaBoost Overall R²")
        with c2: _kpi(f"${ada_mae:,.0f}", "AdaBoost Overall MAE")
        with c3: _kpi(f"{cat_r2:.4f}", "CatBoost Overall R²")
        with c4: _kpi(f"${cat_mae:,.0f}", "CatBoost Overall MAE")
        st.markdown("")
        ada_wins = (merged['Higher_Model']=='AdaBoost').sum()
        cat_wins = (merged['Higher_Model']=='CatBoost').sum()
        c1,c2 = st.columns(2)
        with c1: _kpi(str(ada_wins), "Stores Where AdaBoost Predicts Higher")
        with c2: _kpi(str(cat_wins), "Stores Where CatBoost Predicts Higher")
        st.info("💡 **CatBoost** Generalization Loss: **2.20%** (Very Healthy) | "
                "**AdaBoost** Generalization Loss: **5.19%** (Acceptable)")
    else:
        st.warning("Both models needed for comparison.")
