# -*- coding: utf-8 -*-
"""Page 2 — Long-Term Forecasting (12-Month Recursive Forecast)
Source models: CAT_LIVE.cbm & ADA_LIVE.joblib
Loads trained models and runs live 12-month recursive forecasts for all stores.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings, joblib
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Long-Term Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT = APP_DIR.parent
PROC = PROJECT / "data" / "processed"

# ─── Model paths ──────────────────────────────────────────────────────
def _find(fn):
    for p in [Path("/notebooks")/fn, APP_DIR/fn, APP_DIR/"notebooks"/fn, PROJECT/"notebooks"/fn]:
        if p.exists(): return p
    return None

CAT_PATH = _find("CAT_LIVE.cbm")
ADA_PATH = _find("ADA_LIVE.joblib")
DATA_PATH = PROC / "cleaned_apple_sales_v3.csv"

CHART_BG = "rgba(28,28,30,0.6)"
PAPER_BG = "rgba(0,0,0,0)"
FONT_COLOR = "#E1E1E6"
GRID_COLOR = "rgba(0,240,255,0.10)"
ADA_COLOR = "#00F0FF"
CAT_COLOR = "#39FF14"
HIST_COLOR = "#A0A0A5"
PALETTE = ["#00F0FF","#39FF14","#FF5252","#5E5CE6","#FFFFFF","#00B4FF","#A0A0A5","#FFD60A"]

TARGET = "sales_amount_realistic"
SAFE_LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6]
FEATURES = [
    'sales_lag_1','sales_lag_2','sales_lag_3','sales_lag_6','sales_lag_12',
    'sales_roll_mean_3','sales_roll_mean_6',
    'sales_mom_pct','sales_lag1_vs_roll6',
    'price_realistic','promo_flag',
    'month_sin','month_cos','is_holiday_season','is_launch_season',
    'gdp_per_capita','inflation_rate','exchange_rate','internet_usage_pct',
    'gdp_change','inflation_change','exchange_change','internet_usage_change',
    'store_encoded','num_transactions','num_unique_products','num_categories',
    'year',
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

st.markdown("# 📈 Long-Term Forecasting")
st.markdown("*12-month recursive forecast using live AdaBoost & CatBoost models (no overfitting).*")
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
    df_raw["year"]  = df_raw["sale_date"].dt.year
    df_raw["month"] = df_raw["sale_date"].dt.month
    if "country_norm_mapped" not in df_raw.columns:
        df_raw["country_norm_mapped"] = df_raw["country"].str.lower().str.strip()
    monthly_aggs = {
        "sales_amount_realistic":"sum","quantity_realistic":"sum",
        "price_realistic":"mean","store_name":"first",
        "country_norm_mapped":"first","promo_flag":"mean",
    }
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
    df_monthly["is_launch_season"]  = df_monthly["month"].isin([9,10]).astype(int)
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
    # Historical monthly totals
    hist = df_raw.groupby(df_raw['sale_date'].dt.to_period('M'))['sales_amount_realistic'].sum().reset_index()
    hist.columns = ['date','sales']
    hist['date'] = hist['date'].dt.to_timestamp()
    return df_clean, hist

# ─── Recursive forecast engine ───────────────────────────────────────
def recursive_forecast(model, store_history, n_months, features):
    history = store_history.copy()
    predictions = []
    for step in range(n_months):
        last_row = history.iloc[-1]
        last_date = last_row["date"]
        next_date = last_date + pd.DateOffset(months=1)
        new_row = {}
        sales_series = history[TARGET].values
        for lag in SAFE_LAGS:
            new_row[f"sales_lag_{lag}"] = sales_series[-lag] if lag <= len(sales_series) else 0.0
        for w in ROLL_WINDOWS:
            wd = sales_series[-w:] if len(sales_series) >= w else sales_series
            new_row[f"sales_roll_mean_{w}"] = float(np.mean(wd))
        if len(sales_series) >= 2 and sales_series[-2] != 0:
            new_row["sales_mom_pct"] = (sales_series[-1] - sales_series[-2]) / sales_series[-2]
        else:
            new_row["sales_mom_pct"] = 0.0
        r6 = new_row.get("sales_roll_mean_6", 1)
        new_row["sales_lag1_vs_roll6"] = new_row["sales_lag_1"] / r6 if r6 != 0 else 0.0
        new_row["price_realistic"] = last_row.get("price_realistic", 0)
        new_row["promo_flag"] = last_row.get("promo_flag", 0)
        nm = next_date.month
        new_row["month_sin"] = np.sin(2*np.pi*nm/12)
        new_row["month_cos"] = np.cos(2*np.pi*nm/12)
        new_row["is_holiday_season"] = int(nm in [11,12])
        new_row["is_launch_season"]  = int(nm in [9,10])
        for col in ["gdp_per_capita","inflation_rate","exchange_rate","internet_usage_pct"]:
            new_row[col] = last_row.get(col, 0)
        for col in ["gdp_change","inflation_change","exchange_change","internet_usage_change"]:
            new_row[col] = last_row.get(col, 0)
        new_row["store_encoded"] = last_row.get("store_encoded", 0)
        new_row["num_transactions"] = last_row.get("num_transactions", 0)
        new_row["num_unique_products"] = last_row.get("num_unique_products", 0)
        new_row["num_categories"] = last_row.get("num_categories", 0)
        new_row["year"] = next_date.year
        valid_f = [f for f in features if f in new_row]
        X_new = pd.DataFrame([{f: new_row[f] for f in valid_f}])
        pred = max(float(model.predict(X_new)[0]), 0)
        predictions.append({"date": next_date, "predicted_sales": pred})
        next_full = last_row.copy()
        next_full["date"] = next_date
        next_full["year"] = next_date.year
        next_full["month"] = nm
        next_full[TARGET] = pred
        for k, v in new_row.items():
            if k in next_full.index: next_full[k] = v
        history = pd.concat([history, pd.DataFrame([next_full])], ignore_index=True)
    return pd.DataFrame(predictions)

# ─── Forecast all stores ─────────────────────────────────────────────
@st.cache_data(show_spinner="Running 12-month forecasts for all stores...")
def forecast_all_stores(_model, df_clean, n_months, features, model_name):
    stores = df_clean["store_id"].unique()
    all_fc = []
    for sid in stores:
        sdata = df_clean[df_clean["store_id"] == sid].copy()
        if len(sdata) < 13:
            continue
        fc = recursive_forecast(_model, sdata, n_months, features)
        sname = sdata["store_name"].iloc[0] if "store_name" in sdata.columns else sid
        country = sdata["country_norm_mapped"].iloc[0] if "country_norm_mapped" in sdata.columns else ""
        fc["store_id"] = sid
        fc["store_name"] = sname
        fc["country"] = country
        all_fc.append(fc)
    if all_fc:
        return pd.concat(all_fc, ignore_index=True)
    return None

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
models = load_models()
if not models:
    st.warning("⚠️ **Model files not found.** Please ensure `CAT_LIVE.cbm` and `ADA_LIVE.joblib` "
               "are in the `notebooks/` folder or copied into the app directory.")
    st.stop()

df_clean, historical = load_and_preprocess()

HORIZON = 12

ada_fc = None
cat_fc = None
if "AdaBoost" in models:
    ada_fc = forecast_all_stores(models["AdaBoost"], df_clean, HORIZON, FEATURES, "AdaBoost")
if "CatBoost" in models:
    cat_fc = forecast_all_stores(models["CatBoost"], df_clean, HORIZON, FEATURES, "CatBoost")

if ada_fc is None and cat_fc is None:
    st.error("No forecasts could be generated. Check model files and data.")
    st.stop()

# ─── Aggregates ───────────────────────────────────────────────────────
def agg_monthly(fc):
    if fc is None: return None
    return fc.groupby("date")["predicted_sales"].sum().reset_index()

ada_monthly = agg_monthly(ada_fc)
cat_monthly = agg_monthly(cat_fc)

# ═══════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════
ada_total = format_currency(ada_fc["predicted_sales"].sum()) if ada_fc is not None else "N/A"
cat_total = format_currency(cat_fc["predicted_sales"].sum()) if cat_fc is not None else "N/A"
n_stores = str(ada_fc["store_id"].nunique()) if ada_fc is not None else (
    str(cat_fc["store_id"].nunique()) if cat_fc is not None else "N/A")

c1, c2, c3, c4 = st.columns(4)
with c1: _kpi(n_stores, "Stores Forecasted")
with c2: _kpi(f"{HORIZON} Months", "Forecast Horizon")
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
    st.info("💡 **Long-term forecasting** predicts **12 months ahead** using "
            "recursive step-by-step prediction. Models from `CAT_LIVE.ipynb` & `ADA_LIVE.ipynb` (no overfitting).")
    fig = go.Figure()
    if historical is not None:
        fig.add_trace(go.Scatter(x=historical['date'], y=historical['sales'], mode='lines+markers',
                                  name='Historical', line=dict(color=HIST_COLOR, width=2),
                                  marker=dict(size=4)))
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
    ref = ada_monthly if ada_monthly is not None else cat_monthly
    if ref is not None and len(ref) > 1:
        fig.add_vrect(x0=ref['date'].min(), x1=ref['date'].max(),
                      fillcolor="orange", opacity=0.03, layer="below", line_width=0,
                      annotation_text="12-Month Forecast", annotation_position="top left")
    fig.update_layout(title="Historical vs 12-Month Recursive Forecast — Total Monthly Sales",
        xaxis=dict(range=[historical['date'].min() if historical is not None else '2021-01-01', ref['date'].max() + pd.DateOffset(months=1) if ref is not None else '2027-01-15']))
    style_fig(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly Breakdown Bars
    st.markdown('<div class="section-header">📅 Monthly Predicted Sales — AdaBoost vs CatBoost</div>', unsafe_allow_html=True)
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
                       title="Monthly Total Predicted Sales — AdaBoost vs CatBoost",
                       color_discrete_map={'AdaBoost': ADA_COLOR, 'CatBoost': CAT_COLOR},
                       text_auto='.3s')
        fig2.update_traces(textposition='outside', textfont_size=9)
        fig2.update_layout(xaxis=dict(tickangle=-45, tickformat="%b %Y", dtick="M1"))
        style_fig(fig2, 420)
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2: Store-Level Analysis ──
with tab_stores:
    st.markdown('<div class="section-header">🏪 Top & Bottom Stores by Predicted Annual Sales</div>', unsafe_allow_html=True)
    model_choice = st.radio("Select Model:", ["AdaBoost", "CatBoost"], horizontal=True, key="lt_model")
    fc = ada_fc if model_choice == "AdaBoost" else cat_fc
    color = ADA_COLOR if model_choice == "AdaBoost" else CAT_COLOR

    if fc is not None:
        store_totals = fc.groupby('store_name')['predicted_sales'].sum().sort_values(ascending=False)
        top10 = store_totals.head(10)
        bot10 = store_totals.tail(10)

        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.bar(x=top10.values, y=top10.index, orientation='h',
                           title=f"Top 10 Stores — {model_choice} (Full Year)",
                           text=[f"${v/1e6:.2f}M" for v in top10.values],
                           color_discrete_sequence=[color])
            fig3.update_traces(textposition='outside')
            fig3.update_layout(yaxis=dict(autorange="reversed"))
            style_fig(fig3, 420)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.bar(x=bot10.values, y=bot10.index, orientation='h',
                           title=f"Bottom 10 Stores — {model_choice} (Full Year)",
                           text=[f"${v/1e6:.2f}M" for v in bot10.values],
                           color_discrete_sequence=["#fb923c"])
            fig4.update_traces(textposition='outside')
            fig4.update_layout(yaxis=dict(autorange="reversed"))
            style_fig(fig4, 420)
            st.plotly_chart(fig4, use_container_width=True)

        # Store-level monthly trend
        st.markdown('<div class="section-header">📈 12-Month Forecast Trend for Selected Stores</div>', unsafe_allow_html=True)
        all_stores = sorted(fc['store_name'].unique())
        selected_stores = st.multiselect("Select stores to compare monthly trends:", all_stores,
                                          default=all_stores[:3] if len(all_stores) >= 3 else all_stores, key="lt_stores")
        if selected_stores:
            sub = fc[fc['store_name'].isin(selected_stores)]
            fig5 = px.line(sub, x='date', y='predicted_sales', color='store_name',
                            markers=True, title=f"{model_choice} — Monthly Forecast Trend by Store")
            fig5.update_layout(xaxis=dict(tickangle=-45, tickformat="%b %Y", dtick="M1"))
            style_fig(fig5, 400)
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning(f"No {model_choice} forecast available.")

    if fc is not None:
        st.download_button(f"📥 Download {model_choice} Long-Term Forecast CSV", fc.to_csv(index=False),
                           f"long_term_{model_choice.lower()}_forecast.csv")

# ── TAB 3: Model Comparison ──
with tab_compare:
    st.markdown('<div class="section-header">⚖️ AdaBoost vs CatBoost — 12-Month Forecast Comparison</div>', unsafe_allow_html=True)
    if ada_fc is not None and cat_fc is not None:
        ada_store_totals = ada_fc.groupby('store_id')['predicted_sales'].sum().reset_index()
        ada_store_totals.columns = ['store_id', 'AdaBoost_Annual']
        cat_store_totals = cat_fc.groupby('store_id')['predicted_sales'].sum().reset_index()
        cat_store_totals.columns = ['store_id', 'CatBoost_Annual']
        # Add store names
        name_map = ada_fc[['store_id','store_name']].drop_duplicates().set_index('store_id')['store_name']
        comp = ada_store_totals.merge(cat_store_totals, on='store_id', how='outer')
        comp['store_name'] = comp['store_id'].map(name_map)
        comp['Difference'] = comp['AdaBoost_Annual'] - comp['CatBoost_Annual']
        comp['Higher_Model'] = comp.apply(
            lambda r: 'AdaBoost' if r['AdaBoost_Annual'] >= r['CatBoost_Annual'] else 'CatBoost', axis=1)
        st.dataframe(comp.sort_values('CatBoost_Annual', ascending=False), use_container_width=True, height=400)

        # Summary KPIs — exact metrics from CAT_LIVE.ipynb and ADA_LIVE.ipynb
        st.markdown('<div class="section-header">📊 Overall Model Metrics (Test Set — No Overfitting)</div>', unsafe_allow_html=True)
        cat_r2_val = 0.9766
        cat_mae_val = 41447
        ada_r2_val = 0.8944
        ada_mae_val = 114310

        c1, c2, c3, c4 = st.columns(4)
        with c1: _kpi(f"{ada_r2_val:.4f}", "AdaBoost Overall R²")
        with c2: _kpi(f"${ada_mae_val:,.0f}", "AdaBoost Overall MAE")
        with c3: _kpi(f"{cat_r2_val:.4f}", "CatBoost Overall R²")
        with c4: _kpi(f"${cat_mae_val:,.0f}", "CatBoost Overall MAE")
        st.markdown("")

        ada_wins = (comp['Higher_Model'] == 'AdaBoost').sum()
        cat_wins = (comp['Higher_Model'] == 'CatBoost').sum()
        c1, c2 = st.columns(2)
        with c1: _kpi(str(ada_wins), "Stores Where AdaBoost Predicts Higher")
        with c2: _kpi(str(cat_wins), "Stores Where CatBoost Predicts Higher")

        st.markdown("")
        st.info("💡 **CatBoost** R² Generalization Loss: **2.20%** (Very Healthy) | "
                "**AdaBoost** R² Generalization Loss: **5.19%** (Acceptable)")
    else:
        st.warning("Both models are needed for comparison.")
