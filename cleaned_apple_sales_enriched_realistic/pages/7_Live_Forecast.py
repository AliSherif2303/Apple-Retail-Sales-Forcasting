# -*- coding: utf-8 -*-
"""
Page 7 — Live Model Forecast
Load the trained CatBoost (.cbm) and AdaBoost (.joblib) models,
let the user pick Country → City → Store and a forecast horizon,
then run a recursive forecast and display interactive charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings, joblib

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────
APP_DIR   = Path(__file__).resolve().parent.parent          # cleaned_apple_sales_enriched_realistic/
PROJECT   = APP_DIR.parent                                  # Apple-Retail-Sales-Forcasting/
PROC      = PROJECT / "data" / "processed"
NB_DIR    = PROJECT / "notebooks"
ADA_PATH  = NB_DIR / "ADA_LIVE.joblib"
CAT_PATH  = NB_DIR / "CAT_LIVE.cbm"
DATA_PATH = PROC / "cleaned_apple_sales_v3.csv"

# ─── Visual Constants ─────────────────────────────────────────────────
CHART_BG   = "rgba(28,28,30,0.6)"
PAPER_BG   = "rgba(0,0,0,0)"
FONT_COLOR = "#E1E1E6"
GRID_COLOR = "rgba(0,240,255,0.10)"
ADA_COLOR  = "#00F0FF"
CAT_COLOR  = "#39FF14"
HIST_COLOR = "#A0A0A5"
PALETTE    = ["#00F0FF","#39FF14","#FF5252","#5E5CE6","#FFFFFF","#00B4FF","#A0A0A5","#FFD60A"]

# ─── Feature configuration (must match notebooks) ────────────────────
TARGET    = "sales_amount_realistic"
SAFE_LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6]

FEATURES = [
    'sales_lag_1',  'sales_lag_2',  'sales_lag_3',
    'sales_lag_6',  'sales_lag_12',
    'sales_roll_mean_3', 'sales_roll_mean_6',
    'sales_mom_pct', 'sales_lag1_vs_roll6',
    'price_realistic', 'promo_flag',
    'month_sin', 'month_cos',
    'is_holiday_season', 'is_launch_season',
    'gdp_per_capita', 'inflation_rate', 'exchange_rate', 'internet_usage_pct',
    'gdp_change', 'inflation_change', 'exchange_change', 'internet_usage_change',
    'store_encoded', 'num_transactions', 'num_unique_products', 'num_categories',
    'year',
]

# ─── Helpers ──────────────────────────────────────────────────────────
def style_fig(fig, h=420):
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
        height=h, margin=dict(l=16, r=16, t=42, b=16),
        legend=dict(bgcolor="rgba(28,28,30,0.85)",
                    bordercolor="rgba(0,240,255,0.3)", borderwidth=1),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="rgba(0,240,255,0.2)"),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="rgba(0,240,255,0.2)"),
        colorway=PALETTE,
    )

def _kpi(v, l):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{v}</div>'
        f'<div class="kpi-label">{l}</div></div>',
        unsafe_allow_html=True,
    )

def format_currency(val):
    if val >= 1e9:  return f"${val/1e9:.2f}B"
    if val >= 1e6:  return f"${val/1e6:.2f}M"
    if val >= 1e3:  return f"${val/1e3:.1f}K"
    return f"${val:,.0f}"

# ─── CSS (matches rest of app) ────────────────────────────────────────
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

# ─── Page header ──────────────────────────────────────────────────────
st.markdown("# 🔮 Live Model Forecast")
st.markdown("*Select a store, choose a horizon, and get real-time predictions from both CatBoost & AdaBoost.*")

# ═══════════════════════════════════════════════════════════════════════
# LOAD MODELS & DATA
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {}
    # CatBoost
    if CAT_PATH.exists():
        from catboost import CatBoostRegressor
        cb = CatBoostRegressor()
        cb.load_model(str(CAT_PATH))
        models["CatBoost"] = cb
    # AdaBoost
    if ADA_PATH.exists():
        models["AdaBoost"] = joblib.load(ADA_PATH)
    return models

@st.cache_data(show_spinner="Loading & preprocessing data...")
def load_and_preprocess():
    """Replicate the exact pipeline from CAT_LIVE / ADA_LIVE notebooks."""
    from sklearn.preprocessing import LabelEncoder

    df_raw = pd.read_csv(DATA_PATH)
    df_raw["sale_date"] = pd.to_datetime(df_raw["sale_date"])
    df_raw["year"]  = df_raw["sale_date"].dt.year
    df_raw["month"] = df_raw["sale_date"].dt.month

    if "country_norm_mapped" not in df_raw.columns:
        df_raw["country_norm_mapped"] = df_raw["country"].str.lower().str.strip()

    # ── Monthly aggregation ──
    monthly_aggs = {
        "sales_amount_realistic": "sum",
        "quantity_realistic": "sum",
        "price_realistic": "mean",
        "store_name": "first",
        "country_norm_mapped": "first",
        "promo_flag": "mean",
    }
    for ecol in ["gdp_per_capita","inflation_rate","exchange_rate","internet_usage_pct"]:
        if ecol in df_raw.columns:
            monthly_aggs[ecol] = "first"

    df_monthly = (
        df_raw
        .groupby(["store_id","year","month"])
        .agg(monthly_aggs)
        .reset_index()
    )

    # Extra monthly features
    grp = df_raw.groupby(["store_id","year","month"])
    df_monthly["num_transactions"]    = grp.size().values
    df_monthly["num_unique_products"] = grp["product_id"].nunique().values
    if "category_id" in df_raw.columns:
        df_monthly["num_categories"]  = grp["category_id"].nunique().values

    # Date
    df_monthly["date"] = pd.to_datetime(df_monthly[["year","month"]].assign(day=1))
    df_monthly.sort_values(["store_id","date"], inplace=True)
    df_monthly.reset_index(drop=True, inplace=True)

    # ── Lag features ──
    for lag in SAFE_LAGS:
        df_monthly[f"sales_lag_{lag}"] = df_monthly.groupby("store_id")[TARGET].shift(lag)

    # ── Rolling features ──
    for w in ROLL_WINDOWS:
        rolled = df_monthly.groupby("store_id")[TARGET].shift(1).rolling(w, min_periods=1)
        df_monthly[f"sales_roll_mean_{w}"] = rolled.mean().reset_index(0, drop=True)

    # ── Momentum ──
    df_monthly["sales_mom_pct"]       = df_monthly.groupby("store_id")[TARGET].pct_change()
    df_monthly["sales_lag1_vs_roll6"] = (
        df_monthly["sales_lag_1"] / df_monthly["sales_roll_mean_6"].replace(0, np.nan)
    )

    # ── Cyclical time ──
    df_monthly["month_sin"]         = np.sin(2*np.pi*df_monthly["month"]/12)
    df_monthly["month_cos"]         = np.cos(2*np.pi*df_monthly["month"]/12)
    df_monthly["is_holiday_season"] = df_monthly["month"].isin([11,12]).astype(int)
    df_monthly["is_launch_season"]  = df_monthly["month"].isin([9,10]).astype(int)

    # ── Economic changes ──
    for col in ["gdp_per_capita","inflation_rate","exchange_rate","internet_usage_pct"]:
        if col in df_monthly.columns:
            new = col.replace("_per_capita","").replace("_rate","").replace("_pct","") + "_change"
            df_monthly[new] = df_monthly.groupby("store_id")[col].pct_change().fillna(0)

    # ── Store encoding ──
    le = LabelEncoder()
    df_monthly["store_encoded"] = le.fit_transform(df_monthly["store_id"])

    # ── Clean NaNs (drop first 12 months per store) ──
    df_monthly["row_num"] = df_monthly.groupby("store_id").cumcount()
    df_clean = df_monthly[df_monthly["row_num"] >= 12].copy()
    df_clean.drop(columns="row_num", inplace=True)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

    # Build store lookup: store_id → (store_name, city, country)
    store_lookup = (
        df_raw[["store_id","store_name","city","country_norm_mapped"]]
        .drop_duplicates("store_id")
        .set_index("store_id")
    )

    return df_clean, store_lookup, le


models = load_models()
if not models:
    st.info(
        "### ⚙️ Model Files Not Found\n\n"
        "This page requires trained model files that are generated by running the live forecast notebooks.\n\n"
        "**To generate the models, SSH into the VM and run:**\n"
        "```bash\n"
        f"# Model files expected at:\n"
        f"# {ADA_PATH}\n"
        f"# {CAT_PATH}\n"
        "```\n\n"
        "**Steps:**\n"
        "1. Open `notebooks/CAT_LIVE.ipynb` and run all cells\n"
        "2. Open `notebooks/ADA_LIVE.ipynb` and run all cells\n"
        "3. Confirm both `.cbm` and `.joblib` files appear in the `notebooks/` directory\n"
        "4. Refresh this page\n\n"
        "_The other pages (Dashboard, Forecasting, Clustering) work without these files._"
    )
    st.stop()

df_clean, store_lookup, label_encoder = load_and_preprocess()

# ═══════════════════════════════════════════════════════════════════════
# USER CONTROLS
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎛️ Configure Your Forecast</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

# Build cascading dropdowns
countries = sorted(store_lookup["country_norm_mapped"].str.title().unique())
with col_a:
    sel_country = st.selectbox("🌍 Country", countries, key="fc_country")

# Filter cities for selected country
cities_for_country = sorted(
    store_lookup[store_lookup["country_norm_mapped"] == sel_country.lower()]["city"].unique()
)
with col_b:
    sel_city = st.selectbox("🏙️ City", cities_for_country, key="fc_city")

# Filter stores for selected city
stores_for_city = store_lookup[
    (store_lookup["country_norm_mapped"] == sel_country.lower()) &
    (store_lookup["city"] == sel_city)
]
store_options = {row["store_name"]: sid for sid, row in stores_for_city.iterrows()}
with col_c:
    sel_store_name = st.selectbox("🏪 Store", list(store_options.keys()), key="fc_store")

sel_store_id = store_options[sel_store_name]

col_d, col_e = st.columns(2)
with col_d:
    horizon = st.selectbox(
        "📅 Forecast Horizon",
        [1, 3, 6, 12],
        index=1,
        format_func=lambda x: f"{x} Month{'s' if x > 1 else ''}",
        key="fc_horizon",
    )
with col_e:
    available_models = list(models.keys())
    selected_models = st.multiselect(
        "🤖 Models to Run",
        available_models,
        default=available_models,
        key="fc_models",
    )

run_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# RECURSIVE FORECASTING ENGINE
# ═══════════════════════════════════════════════════════════════════════

def recursive_forecast(model, store_history, n_months, features):
    """
    Recursively predict n_months ahead for a single store.
    `store_history` is a DataFrame of all historical monthly rows for the store,
    already fully feature-engineered.
    Returns a list of dicts: [{date, predicted_sales}, ...]
    """
    history = store_history.copy()
    predictions = []

    for step in range(n_months):
        last_row = history.iloc[-1]

        # Build the next month's date
        last_date = last_row["date"]
        next_date = last_date + pd.DateOffset(months=1)
        next_year  = next_date.year
        next_month = next_date.month

        # Build feature row
        new_row = {}

        # Lag features — use the last N values of TARGET in history
        sales_series = history[TARGET].values
        for lag in SAFE_LAGS:
            if lag <= len(sales_series):
                new_row[f"sales_lag_{lag}"] = sales_series[-lag]
            else:
                new_row[f"sales_lag_{lag}"] = 0.0

        # Rolling means (shifted by 1 already since lags start at 1)
        for w in ROLL_WINDOWS:
            window_data = sales_series[-w:] if len(sales_series) >= w else sales_series
            new_row[f"sales_roll_mean_{w}"] = float(np.mean(window_data))

        # Momentum
        if len(sales_series) >= 2 and sales_series[-2] != 0:
            new_row["sales_mom_pct"] = (sales_series[-1] - sales_series[-2]) / sales_series[-2]
        else:
            new_row["sales_mom_pct"] = 0.0

        roll6_val = new_row.get("sales_roll_mean_6", 1)
        new_row["sales_lag1_vs_roll6"] = (
            new_row["sales_lag_1"] / roll6_val if roll6_val != 0 else 0.0
        )

        # Pricing / promo (carry forward last known)
        new_row["price_realistic"] = last_row.get("price_realistic", 0)
        new_row["promo_flag"]      = last_row.get("promo_flag", 0)

        # Cyclical time
        new_row["month_sin"]         = np.sin(2 * np.pi * next_month / 12)
        new_row["month_cos"]         = np.cos(2 * np.pi * next_month / 12)
        new_row["is_holiday_season"] = int(next_month in [11, 12])
        new_row["is_launch_season"]  = int(next_month in [9, 10])

        # Economics (carry forward)
        for col in ["gdp_per_capita","inflation_rate","exchange_rate","internet_usage_pct"]:
            new_row[col] = last_row.get(col, 0)
        for col in ["gdp_change","inflation_change","exchange_change","internet_usage_change"]:
            new_row[col] = last_row.get(col, 0)

        # Metadata (carry forward)
        new_row["store_encoded"]      = last_row.get("store_encoded", 0)
        new_row["num_transactions"]   = last_row.get("num_transactions", 0)
        new_row["num_unique_products"]= last_row.get("num_unique_products", 0)
        new_row["num_categories"]     = last_row.get("num_categories", 0)
        new_row["year"]               = next_year

        # Build DataFrame row for prediction
        valid_features = [f for f in features if f in new_row]
        X_new = pd.DataFrame([{f: new_row[f] for f in valid_features}])

        # Predict
        pred = float(model.predict(X_new)[0])
        pred = max(pred, 0)   # sales can't be negative

        predictions.append({"date": next_date, "predicted_sales": pred})

        # Append prediction to history for the next step
        next_full = last_row.copy()
        next_full["date"]   = next_date
        next_full["year"]   = next_year
        next_full["month"]  = next_month
        next_full[TARGET]   = pred
        for k, v in new_row.items():
            if k in next_full.index:
                next_full[k] = v
        history = pd.concat([history, pd.DataFrame([next_full])], ignore_index=True)

    return pd.DataFrame(predictions)


# ═══════════════════════════════════════════════════════════════════════
# RUN FORECAST
# ═══════════════════════════════════════════════════════════════════════
if run_btn and selected_models:
    store_data = df_clean[df_clean["store_id"] == sel_store_id].copy()

    if len(store_data) < 13:
        st.error(f"Not enough historical data for **{sel_store_name}** (need at least 13 months).")
        st.stop()

    results = {}
    with st.spinner(f"Forecasting {horizon} month(s) for **{sel_store_name}**..."):
        for mname in selected_models:
            results[mname] = recursive_forecast(
                models[mname], store_data, horizon, FEATURES
            )

    # ═══════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📊 Forecast Results</div>', unsafe_allow_html=True)

    # ── KPI Cards ──
    kpi_cols = st.columns(len(selected_models) * 2)
    idx = 0
    for mname in selected_models:
        fc_df = results[mname]
        total = fc_df["predicted_sales"].sum()
        avg   = fc_df["predicted_sales"].mean()
        with kpi_cols[idx]:
            _kpi(format_currency(total), f"{mname} — Total ({horizon}mo)")
        with kpi_cols[idx+1]:
            _kpi(format_currency(avg), f"{mname} — Monthly Avg")
        idx += 2

    st.markdown("")

    # ── Historical + Forecast Chart ──
    st.markdown('<div class="section-header">📈 Historical + Forecast Trend</div>', unsafe_allow_html=True)

    fig = go.Figure()

    # Historical
    hist = store_data[["date", TARGET]].copy()
    hist = hist.sort_values("date")
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist[TARGET],
        mode="lines+markers", name="Historical",
        line=dict(color=HIST_COLOR, width=2),
        marker=dict(size=4),
    ))

    # Forecast lines
    model_colors = {"CatBoost": CAT_COLOR, "AdaBoost": ADA_COLOR}
    model_dashes = {"CatBoost": "dot", "AdaBoost": "dash"}
    model_symbols = {"CatBoost": "triangle-up", "AdaBoost": "square"}

    for mname in selected_models:
        fc_df = results[mname]
        color = model_colors.get(mname, "#fbbf24")
        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df["predicted_sales"],
            mode="lines+markers+text", name=f"{mname} Forecast",
            line=dict(color=color, width=3, dash=model_dashes.get(mname, "solid")),
            marker=dict(size=8, symbol=model_symbols.get(mname, "circle")),
            text=[format_currency(v) for v in fc_df["predicted_sales"]],
            textposition="top center",
            textfont=dict(size=10),
        ))

    # Shade forecast region
    all_dates = pd.concat([r for r in results.values()])["date"]
    if len(all_dates) > 0:
        fig.add_vrect(
            x0=all_dates.min(), x1=all_dates.max(),
            fillcolor="orange", opacity=0.04, layer="below", line_width=0,
            annotation_text=f"{horizon}-Month Forecast",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"{sel_store_name} — Historical vs {horizon}-Month Forecast",
        xaxis_title="Date", yaxis_title="Sales ($)",
    )
    style_fig(fig, 520)
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly Breakdown Bar Chart ──
    if len(selected_models) == 2:
        st.markdown('<div class="section-header">📊 Model Comparison — Monthly Breakdown</div>', unsafe_allow_html=True)
        bar_data = []
        for mname in selected_models:
            for _, row in results[mname].iterrows():
                bar_data.append({
                    "Month": row["date"].strftime("%b %Y"),
                    "Sales": row["predicted_sales"],
                    "Model": mname,
                })
        bar_df = pd.DataFrame(bar_data)
        fig2 = px.bar(
            bar_df, x="Month", y="Sales", color="Model", barmode="group",
            title=f"Monthly Predicted Sales — {sel_store_name}",
            color_discrete_map={"AdaBoost": ADA_COLOR, "CatBoost": CAT_COLOR},
            text_auto=".3s",
        )
        fig2.update_traces(textposition="outside", textfont_size=9)
        style_fig(fig2, 420)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Forecast Data Table ──
    st.markdown('<div class="section-header">📋 Detailed Forecast Data</div>', unsafe_allow_html=True)

    for mname in selected_models:
        fc_df = results[mname].copy()
        fc_df["date"] = fc_df["date"].dt.strftime("%B %Y")
        fc_df["predicted_sales"] = fc_df["predicted_sales"].apply(lambda x: f"${x:,.2f}")
        fc_df.columns = ["Month", f"{mname} Predicted Sales"]
        st.markdown(f"**{mname}**")
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # ── Download ──
    st.markdown("")
    for mname in selected_models:
        csv = results[mname].to_csv(index=False)
        st.download_button(
            f"📥 Download {mname} Forecast CSV",
            csv,
            f"{sel_store_name.replace(' ','_')}_{mname}_{horizon}mo_forecast.csv",
            key=f"dl_{mname}",
        )

elif run_btn:
    st.warning("Please select at least one model to run.")
