# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple Sales Intelligence Dashboard",
    page_icon="\U0001F34E",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 40%, #0a0f1a 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    backdrop-filter: blur(10px);
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.25);
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #94a3b8;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.kpi-delta {
    font-size: 0.85rem;
    color: #34d399;
    margin-top: 4px;
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,0.2);
    margin: 24px 0;
}

/* Main title */
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.sub-title {
    font-size: 1rem;
    color: #64748b;
    margin-top: 4px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15,15,30,0.6);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid rgba(99,102,241,0.2);
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: #64748b;
    border-radius: 8px;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.4), rgba(139,92,246,0.3)) !important;
    color: #c4b5fd !important;
}

/* Metric containers */
[data-testid="stMetricValue"] { color: #818cf8 !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-weight: 600; }

/* Plotly charts background */
.js-plotly-plot .plotly .modebar { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────────
# Use a path relative to this script so it works locally AND on Streamlit Cloud
DATA_PATH = Path(__file__).parent

@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    dim_economic = pd.read_csv(f"{DATA_PATH}/Dim_Macroeconomics.csv", on_bad_lines="skip")
    dim_product  = pd.read_csv(f"{DATA_PATH}/Dim_Product.csv",  on_bad_lines="skip")
    dim_store    = pd.read_csv(f"{DATA_PATH}/Dim_Store.csv",    on_bad_lines="skip")
    fact_sales   = pd.read_csv(f"{DATA_PATH}/Fact_Sales.csv",   on_bad_lines="skip")

    # Parse dates
    fact_sales["sale_date"]    = pd.to_datetime(fact_sales["sale_date"])
    # dim_economic["sale_date"]  = pd.to_datetime(dim_economic["sale_date"])  # removed: column not present

    # Deduplicate dim_product (each product_id appears twice with different launch_dates)
    dim_product_dedup = dim_product.drop_duplicates(subset="product_id", keep="first")

    # Merge into one analytical table
    # Note: both fact_sales and dim_economic have 'year', so use suffixes
    df = (fact_sales
          .merge(dim_store,    on="store_id",   how="left", suffixes=("", "_store"))
          .merge(dim_product_dedup,  on="product_id", how="left", suffixes=("", "_prod"))
          .merge(dim_economic, on=["year", "country_norm_mapped"], how="left", suffixes=("", "_econ")))

    df["year"]  = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month
    df["month_name"] = df["sale_date"].dt.strftime("%b")
    df["yearmonth"] = df["sale_date"].dt.to_period("M").astype(str)
    df["quarter"] = df["sale_date"].dt.quarter.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
    df["country"] = df["country_norm_mapped"].str.title()

    # Memory optimisation (cuts RAM by ~65%)
    # Convert low-cardinality string/object columns -> category
    for col in df.select_dtypes("object").columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")
    # float64 -> float32  (halves float memory)
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    # int64 -> smallest fitting int type
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    # Same optimisations on dim_economic (used directly in Tab 5)
    for col in dim_economic.select_dtypes("object").columns:
        if dim_economic[col].nunique() / len(dim_economic) < 0.5:
            dim_economic[col] = dim_economic[col].astype("category")
    for col in dim_economic.select_dtypes("float64").columns:
        dim_economic[col] = dim_economic[col].astype("float32")

    return df, dim_economic, dim_product, dim_store

df, dim_economic, dim_product, dim_store = load_data()

# ─── Plotly theme helper ─────────────────────────────────────────────
CHART_BG   = "rgba(10,10,20,0)"
PAPER_BG   = "rgba(10,10,20,0)"
FONT_COLOR = "#cbd5e1"
GRID_COLOR = "rgba(99,102,241,0.12)"
PURPLE_SEQ = px.colors.sequential.Purp
PALETTE    = ["#818cf8","#c084fc","#f472b6","#34d399","#fbbf24","#60a5fa","#fb923c","#a78bfa","#38bdf8","#4ade80"]

def style_fig(fig, height=420):
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
        height=height,
        margin=dict(l=16, r=16, t=36, b=16),
        legend=dict(
            bgcolor="rgba(15,15,30,0.7)",
            bordercolor="rgba(99,102,241,0.25)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(size=11)),
        colorway=PALETTE,
    )
    return fig


# ─── Sidebar Filters ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.6rem;">\U0001F34E Apple EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Sales Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("### \U0001F50D Filters")

    years = sorted(df["year"].unique())
    sel_years = st.multiselect("Year(s)", years, default=years, key="yr")

    all_countries = sorted(df["country"].dropna().unique())
    sel_countries = st.multiselect("Country/Market", all_countries, default=all_countries, key="ct")

    all_categories = sorted(df["category_name"].dropna().unique())
    sel_categories = st.multiselect("Product Category", all_categories, default=all_categories, key="cat")

    promo_opt = st.radio("Promotion Filter", ["All", "Promo Only", "No Promo"], horizontal=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    n_txns = f"{len(df):,}"
    n_markets = df["country_norm_mapped"].nunique()
    n_products = df["product_id"].nunique()
    n_stores = df["store_id"].nunique()
    st.markdown(f"""
    <div style='color:#475569;font-size:0.75rem;'>
    Data covers <b style='color:#818cf8'>{n_txns}</b> transactions across
    <b style='color:#818cf8'>{n_markets} markets</b>, <b style='color:#818cf8'>{n_products} products</b>
    and <b style='color:#818cf8'>{n_stores} stores</b>.<br><br>
    Date range: Jan 2021 - Dec 2025
    </div>
    """, unsafe_allow_html=True)

# Apply filters
mask = (
    df["year"].isin(sel_years)
    & df["country"].isin(sel_countries)
    & df["category_name"].isin(sel_categories)
)
if promo_opt == "Promo Only":
    mask &= df["promo_flag"] == 1
elif promo_opt == "No Promo":
    mask &= df["promo_flag"] == 0

fdf = df[mask].copy()


# ─── Header ──────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Apple Sales Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-market EDA - 2021-2025 - Powered by Streamlit & Plotly</div>', unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─── KPI Row ─────────────────────────────────────────────────────────
total_revenue   = fdf["sales_amount_realistic"].sum()
total_units     = fdf["quantity"].sum()
total_txns      = len(fdf)
avg_order_value = fdf["sales_amount_realistic"].mean()
promo_rate      = fdf["promo_flag"].mean() * 100
unique_markets  = fdf["country"].nunique()

def fmt_million(v):
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

kpis = [
    ("\U0001F4B0 Total Revenue",      fmt_million(total_revenue),   "Realistic adj. sales"),
    ("\U0001F4E6 Total Units Sold",   f"{total_units:,}",            "Across all products"),
    ("\U0001F9FE Transactions",       f"{total_txns:,}",             "Filtered records"),
    ("\U0001F6D2 Avg Order Value",    f"${avg_order_value:,.0f}",    "Per transaction"),
    ("\U0001F3AF Promo Rate",         f"{promo_rate:.1f}%",          "Transactions on promo"),
    ("\U0001F30D Active Markets",     f"{unique_markets}",           "Countries represented"),
]

cols = st.columns(6)
for col, (label, value, sub) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Tabs ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001F4C8 Sales Trends",
    "\U0001F30D Market Analysis",
    "\U0001F4F1 Product Insights",
    "\U0001FAF6 Store Performance",
    "\U0001F4CA Economic Factors",
    "🔮 Sales Forecasting",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 - SALES TRENDS
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">\U0001F4C8 Revenue Over Time</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_b:
        trend_gran = st.selectbox("Granularity", ["Monthly", "Quarterly", "Yearly"], key="tg")

    if trend_gran == "Monthly":
        ts = fdf.groupby("yearmonth")["sales_amount_realistic"].sum().reset_index()
        ts.columns = ["Period", "Revenue"]
    elif trend_gran == "Quarterly":
        ts = fdf.groupby(["year","quarter"])["sales_amount_realistic"].sum().reset_index()
        ts["Period"] = ts["year"].astype(str) + " " + ts["quarter"].astype(str)
        ts = ts[["Period","sales_amount_realistic"]].rename(columns={"sales_amount_realistic":"Revenue"})
    else:
        ts = fdf.groupby("year")["sales_amount_realistic"].sum().reset_index()
        ts.columns = ["Period", "Revenue"]
        ts["Period"] = ts["Period"].astype(str)

    fig_trend = px.area(
        ts, x="Period", y="Revenue",
        title="Sales Revenue Trend",
        color_discrete_sequence=["#818cf8"],
    )
    fig_trend.update_traces(
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.15)",
        line=dict(width=2.5, color="#818cf8"),
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    )
    style_fig(fig_trend, 380)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Revenue by category over time
    st.markdown('<div class="section-header">\U0001F4F1 Category Revenue by Year</div>', unsafe_allow_html=True)
    cat_yr = fdf.groupby(["year","category_name"])["sales_amount_realistic"].sum().reset_index()
    cat_yr.columns = ["Year","Category","Revenue"]
    cat_yr["Year"] = cat_yr["Year"].astype(str)

    fig_cat_yr = px.bar(
        cat_yr, x="Year", y="Revenue", color="Category",
        title="Revenue by Category & Year",
        barmode="group",
        color_discrete_sequence=PALETTE,
    )
    fig_cat_yr.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: $%{y:,.0f}<extra></extra>")
    style_fig(fig_cat_yr, 380)
    st.plotly_chart(fig_cat_yr, use_container_width=True)

    # Monthly seasonality
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">\U0001F4C5 Monthly Seasonality</div>', unsafe_allow_html=True)
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        # Average monthly revenue = total revenue per month name / number of years
        seas_total = fdf.groupby("month_name")["sales_amount_realistic"].sum().reindex(month_order)
        n_years = fdf["year"].nunique()
        seas = (seas_total / n_years).reset_index()
        seas.columns = ["Month", "Avg Revenue"]
        fig_seas = px.bar(
            seas, x="Month", y="Avg Revenue",
            title="Avg Monthly Revenue (Seasonality)",
            color="Avg Revenue", color_continuous_scale="Purples",
        )
        fig_seas.update_traces(hovertemplate="<b>%{x}</b><br>Avg: $%{y:,.0f}<extra></extra>")
        style_fig(fig_seas, 360)
        st.plotly_chart(fig_seas, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">\U0001F3AF Promo vs Non-Promo Revenue</div>', unsafe_allow_html=True)
        promo_ts = fdf.groupby(["yearmonth","promo_flag"])["sales_amount_realistic"].sum().reset_index()
        promo_ts["Promo"] = promo_ts["promo_flag"].map({0:"No Promo", 1:"Promo"})
        fig_promo = px.line(
            promo_ts, x="yearmonth", y="sales_amount_realistic", color="Promo",
            title="Promo vs. Non-Promo Revenue Over Time",
            color_discrete_map={"Promo":"#f472b6","No Promo":"#818cf8"},
        )
        fig_promo.update_traces(hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>")
        style_fig(fig_promo, 360)
        st.plotly_chart(fig_promo, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 - MARKET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">\U0001F30D Revenue by Country</div>', unsafe_allow_html=True)
        country_rev = (fdf.groupby("country")["sales_amount_realistic"]
                       .sum().sort_values(ascending=False).reset_index())
        country_rev.columns = ["Country","Revenue"]
        fig_country = px.bar(
            country_rev, x="Revenue", y="Country", orientation="h",
            title="Total Revenue by Country",
            color="Revenue", color_continuous_scale="Purples",
        )
        fig_country.update_traces(hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>")
        fig_country.update_layout(yaxis=dict(autorange="reversed"))
        style_fig(fig_country, 500)
        st.plotly_chart(fig_country, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">\U0001F5FA Market Share (Treemap)</div>', unsafe_allow_html=True)
        fig_tree = px.treemap(
            country_rev, path=["Country"], values="Revenue",
            title="Market Share by Country",
            color="Revenue", color_continuous_scale="Purples",
        )
        fig_tree.update_traces(hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>")
        style_fig(fig_tree, 500)
        st.plotly_chart(fig_tree, use_container_width=True)

    # Country growth YoY
    st.markdown('<div class="section-header">\U0001F4CA Country Revenue - Year-over-Year</div>', unsafe_allow_html=True)
    cty_yr = fdf.groupby(["country","year"])["sales_amount_realistic"].sum().reset_index()
    cty_yr.columns = ["Country","Year","Revenue"]
    cty_yr["Year"] = cty_yr["Year"].astype(str)
    top10_ct = country_rev["Country"].head(10).tolist()
    fig_cty_yr = px.line(
        cty_yr[cty_yr["Country"].isin(top10_ct)],
        x="Year", y="Revenue", color="Country",
        title="Top-10 Country Revenue Trend",
        markers=True,
        color_discrete_sequence=PALETTE,
    )
    fig_cty_yr.update_traces(hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>")
    style_fig(fig_cty_yr, 380)
    st.plotly_chart(fig_cty_yr, use_container_width=True)

    # Units sold by country
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">\U0001F4E6 Units Sold by Country</div>', unsafe_allow_html=True)
        units_ct = fdf.groupby("country")["quantity"].sum().sort_values(ascending=False).reset_index()
        units_ct.columns = ["Country","Units"]
        fig_units = px.bar(
            units_ct.head(15), x="Country", y="Units",
            title="Units Sold - Top 15 Countries",
            color="Units", color_continuous_scale="Blues",
        )
        style_fig(fig_units, 360)
        st.plotly_chart(fig_units, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">\U0001F4B8 Avg Order Value by Country</div>', unsafe_allow_html=True)
        aov_ct = fdf.groupby("country")["sales_amount_realistic"].mean().sort_values(ascending=False).reset_index()
        aov_ct.columns = ["Country","AOV"]
        fig_aov = px.bar(
            aov_ct.head(15), x="Country", y="AOV",
            title="Avg Order Value - Top 15 Countries",
            color="AOV", color_continuous_scale="Pinkyl",
        )
        fig_aov.update_traces(hovertemplate="<b>%{x}</b><br>AOV: $%{y:,.0f}<extra></extra>")
        style_fig(fig_aov, 360)
        st.plotly_chart(fig_aov, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 - PRODUCT INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">\U0001F3C6 Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = (fdf.groupby("category_name")["sales_amount_realistic"]
                   .sum().sort_values(ascending=False).reset_index())
        cat_rev.columns = ["Category","Revenue"]
        fig_cat = px.pie(
            cat_rev, names="Category", values="Revenue",
            title="Revenue Share by Category",
            color_discrete_sequence=PALETTE,
            hole=0.45,
        )
        fig_cat.update_traces(hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>")
        style_fig(fig_cat, 400)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">\U0001F4CA Units Sold by Category</div>', unsafe_allow_html=True)
        cat_units = (fdf.groupby("category_name")["quantity"]
                     .sum().sort_values(ascending=False).reset_index())
        cat_units.columns = ["Category","Units"]
        fig_cu = px.bar(
            cat_units, x="Category", y="Units",
            title="Units Sold by Category",
            color="Units", color_continuous_scale="Purples",
        )
        fig_cu.update_xaxes(tickangle=30)
        style_fig(fig_cu, 400)
        st.plotly_chart(fig_cu, use_container_width=True)

    # Top products
    st.markdown('<div class="section-header">\U0001F31F Top Products by Revenue</div>', unsafe_allow_html=True)
    n_top = st.slider("Number of top products to show", 5, 30, 15, key="ntp")
    top_prod = (fdf.groupby(["product_name","category_name"])["sales_amount_realistic"]
                .sum().sort_values(ascending=False).head(n_top).reset_index())
    top_prod.columns = ["Product","Category","Revenue"]

    fig_tp = px.bar(
        top_prod, x="Revenue", y="Product", color="Category",
        orientation="h", title=f"Top {n_top} Products by Revenue",
        color_discrete_sequence=PALETTE,
    )
    fig_tp.update_layout(yaxis=dict(autorange="reversed"))
    fig_tp.update_traces(hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>")
    style_fig(fig_tp, max(380, n_top*22))
    st.plotly_chart(fig_tp, use_container_width=True)

    # Price distribution
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">\U0001F4B2 Price Distribution by Category</div>', unsafe_allow_html=True)
        fig_price = px.box(
            fdf.dropna(subset=["category_name","price"]),
            x="category_name", y="price", color="category_name",
            title="Price Distribution per Category",
            color_discrete_sequence=PALETTE,
        )
        fig_price.update_xaxes(tickangle=30)
        fig_price.update_layout(showlegend=False)
        style_fig(fig_price, 400)
        st.plotly_chart(fig_price, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">\U0001F4E6 Quantity Distribution</div>', unsafe_allow_html=True)
        fig_qty = px.histogram(
            fdf, x="quantity", nbins=10,
            title="Quantity per Transaction",
            color_discrete_sequence=["#818cf8"],
        )
        fig_qty.update_traces(hovertemplate="Qty: %{x}<br>Count: %{y}<extra></extra>")
        style_fig(fig_qty, 400)
        st.plotly_chart(fig_qty, use_container_width=True)

    # Promo effect per category
    st.markdown('<div class="section-header">\U0001F3AF Promo Impact by Category</div>', unsafe_allow_html=True)
    promo_cat = (fdf.groupby(["category_name","promo_flag"])["sales_amount_realistic"]
                 .mean().reset_index())
    promo_cat["Promo"] = promo_cat["promo_flag"].map({0:"No Promo", 1:"Promo"})
    fig_pi = px.bar(
        promo_cat, x="category_name", y="sales_amount_realistic", color="Promo",
        barmode="group",
        title="Avg Transaction Value: Promo vs No-Promo",
        color_discrete_map={"Promo":"#f472b6", "No Promo":"#818cf8"},
    )
    fig_pi.update_xaxes(tickangle=30)
    style_fig(fig_pi, 380)
    st.plotly_chart(fig_pi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 4 - STORE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">\U0001FAF6 Top Stores by Revenue</div>', unsafe_allow_html=True)
        store_rev = (fdf.groupby(["store_id","store_name","city","country"])["sales_amount_realistic"]
                     .sum().sort_values(ascending=False).reset_index().head(20))
        store_rev.columns = ["Store ID","Store Name","City","Country","Revenue"]
        fig_st = px.bar(
            store_rev, x="Revenue", y="Store Name",
            orientation="h", color="Country",
            title="Top 20 Stores by Revenue",
            color_discrete_sequence=PALETTE,
        )
        fig_st.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        fig_st.update_traces(hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>")
        style_fig(fig_st, 560)
        st.plotly_chart(fig_st, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">\U0001F4CA Store Count by Country</div>', unsafe_allow_html=True)
        store_ct = (dim_store.groupby("country_norm_mapped").size()
                    .sort_values(ascending=False).reset_index())
        store_ct.columns = ["Country","Store Count"]
        store_ct["Country"] = store_ct["Country"].str.title()
        fig_sc = px.bar(
            store_ct, x="Country", y="Store Count",
            title="Number of Stores per Country",
            color="Store Count", color_continuous_scale="Purples",
        )
        fig_sc.update_xaxes(tickangle=40)
        style_fig(fig_sc, 380)
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown('<div class="section-header">\U0001F3F7 Revenue per Store by Country</div>', unsafe_allow_html=True)
        rev_per_store = (fdf.groupby("country")["sales_amount_realistic"].sum()
                         / fdf.groupby("country")["store_id"].nunique())
        rev_per_store = rev_per_store.sort_values(ascending=False).reset_index()
        rev_per_store.columns = ["Country","Rev per Store"]
        fig_rps = px.bar(
            rev_per_store, x="Country", y="Rev per Store",
            title="Avg Revenue per Store",
            color="Rev per Store", color_continuous_scale="Pinkyl",
        )
        fig_rps.update_xaxes(tickangle=40)
        fig_rps.update_traces(hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>")
        style_fig(fig_rps, 340)
        st.plotly_chart(fig_rps, use_container_width=True)

    # Store revenue heatmap (store x year)
    st.markdown('<div class="section-header">\U0001F525 Store Revenue Heatmap (Top 20 Stores x Year)</div>', unsafe_allow_html=True)
    top20_stores = (fdf.groupby("store_name")["sales_amount_realistic"]
                    .sum().sort_values(ascending=False).head(20).index.tolist())
    heat_df = (fdf[fdf["store_name"].isin(top20_stores)]
               .groupby(["store_name","year"])["sales_amount_realistic"]
               .sum().reset_index())
    heat_piv = heat_df.pivot(index="store_name", columns="year", values="sales_amount_realistic").fillna(0)
    fig_hm = px.imshow(
        heat_piv,
        color_continuous_scale="Purples",
        title="Revenue Heatmap - Top 20 Stores by Year",
        labels=dict(color="Revenue ($)"),
        aspect="auto",
    )
    fig_hm.update_traces(hovertemplate="Store: <b>%{y}</b><br>Year: %{x}<br>Revenue: $%{z:,.0f}<extra></extra>")
    style_fig(fig_hm, 520)
    st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 5 - ECONOMIC FACTORS
# ═══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">\U0001F30D Macroeconomic Context by Country</div>', unsafe_allow_html=True)

    econ_agg = (dim_economic.groupby("country_norm_mapped")
                .agg(avg_fx=("exchange_rate","mean"),
                     avg_inflation=("inflation_rate","mean"),
                     avg_gdp=("gdp_per_capita","mean"))
                .reset_index())
    econ_agg["country"] = econ_agg["country_norm_mapped"].str.title()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">\U0001F4B9 GDP per Capita by Country</div>', unsafe_allow_html=True)
        fig_gdp = px.bar(
            econ_agg.sort_values("avg_gdp", ascending=False),
            x="country", y="avg_gdp",
            title="Avg GDP per Capita (USD)",
            color="avg_gdp", color_continuous_scale="Purples",
        )
        fig_gdp.update_xaxes(tickangle=40)
        fig_gdp.update_traces(hovertemplate="<b>%{x}</b><br>GDP: $%{y:,.0f}<extra></extra>")
        style_fig(fig_gdp, 380)
        st.plotly_chart(fig_gdp, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">\U0001F4C9 Avg Inflation Rate by Country</div>', unsafe_allow_html=True)
        fig_inf = px.bar(
            econ_agg.sort_values("avg_inflation", ascending=False),
            x="country", y="avg_inflation",
            title="Avg Inflation Rate (%)",
            color="avg_inflation", color_continuous_scale="RdPu",
        )
        fig_inf.update_xaxes(tickangle=40)
        fig_inf.update_traces(hovertemplate="<b>%{x}</b><br>Inflation: %{y:.2f}%<extra></extra>")
        style_fig(fig_inf, 380)
        st.plotly_chart(fig_inf, use_container_width=True)

    # GDP vs Revenue scatter
    st.markdown('<div class="section-header">\U0001F517 GDP vs. Sales Revenue Correlation</div>', unsafe_allow_html=True)
    country_rev_econ = (fdf.groupby("country")["sales_amount_realistic"]
                        .sum().reset_index().rename(columns={"sales_amount_realistic": "Revenue"}))
    econ_scatter = econ_agg.merge(country_rev_econ, on="country", how="inner")

    fig_scatter = px.scatter(
        econ_scatter, x="avg_gdp", y="Revenue",
        size="Revenue", color="country",
        hover_name="country",
        title="GDP per Capita vs. Total Revenue by Country",
        labels={"avg_gdp":"Avg GDP per Capita ($)", "Revenue":"Total Revenue ($)"},
        color_discrete_sequence=PALETTE,
    )
    fig_scatter.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>GDP: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra></extra>"
    )
    style_fig(fig_scatter, 420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Exchange rate & Inflation trends by Year
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">\U0001F4B1 Exchange Rate by Year (Selected Countries)</div>', unsafe_allow_html=True)
        sel_econ_ct = st.multiselect(
            "Countries for FX chart",
            sorted(dim_economic["country_norm_mapped"].str.title().unique()),
            default=["Japan","United Kingdom","Australia","Canada"],
            key="fx_ct",
        )
        econ_ts = dim_economic.copy()
        econ_ts["country"] = econ_ts["country_norm_mapped"].str.title()
        econ_ts["year"] = econ_ts["year"].astype(str)
        econ_ts_f = econ_ts[econ_ts["country"].isin(sel_econ_ct)]
        fig_fx = px.line(
            econ_ts_f, x="year", y="exchange_rate", color="country",
            title="Exchange Rate vs USD by Year",
            markers=True,
            color_discrete_sequence=PALETTE,
        )
        fig_fx.update_traces(hovertemplate="<b>%{x}</b><br>FX: %{y:.3f}<extra></extra>")
        style_fig(fig_fx, 380)
        st.plotly_chart(fig_fx, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">\U0001F4C8 Inflation Rate by Year</div>', unsafe_allow_html=True)
        fig_inf_ts = px.line(
            econ_ts_f.dropna(subset=["inflation_rate"]),
            x="year", y="inflation_rate", color="country",
            title="Inflation Rate by Year",
            markers=True,
            color_discrete_sequence=PALETTE,
        )
        fig_inf_ts.update_traces(hovertemplate="<b>%{x}</b><br>Inflation: %{y:.2f}%<extra></extra>")
        style_fig(fig_inf_ts, 380)
        st.plotly_chart(fig_inf_ts, use_container_width=True)

    # Internet usage by country
    st.markdown('<div class="section-header">\U0001F310 Internet Usage by Country</div>', unsafe_allow_html=True)
    fig_iu = px.bar(
        econ_agg.sort_values("avg_fx", ascending=False),
        x="country", y="avg_fx",
        title="Avg Exchange Rate vs USD by Country",
        color="avg_fx", color_continuous_scale="Purples",
    )
    fig_iu.update_xaxes(tickangle=40)
    fig_iu.update_traces(hovertemplate="<b>%{x}</b><br>Avg FX: %{y:.3f}<extra></extra>")
    style_fig(fig_iu, 320)
    st.plotly_chart(fig_iu, use_container_width=True)



# ═══════════════════════════════════════════════════════════════════════
# TAB 6 - SALES FORECASTING (CatBoost)
# ═══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">\U0001F52E CatBoost Sales Forecasting Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#94a3b8;font-size:0.85rem;margin-bottom:16px;'>
    This tab trains a <b style='color:#c084fc'>CatBoost Regressor</b> on monthly store-level sales data
    with 50+ engineered features (lags, rolling windows, cyclical encoding, economic indicators)
    to forecast future sales. The model is trained once and cached for performance.
    </div>
    """, unsafe_allow_html=True)

    # ── Cached training pipeline ──────────────────────────────────────
    @st.cache_data(show_spinner="Training CatBoost model — this may take a moment...")
    def train_catboost_model(_df_input):
        """Full CatBoost pipeline: aggregate → engineer → train → evaluate."""
        from catboost import CatBoostRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

        RANDOM_STATE = 42
        df_raw = _df_input.copy()

        # --- 1. Monthly aggregation per store ---
        df_raw["sale_date"] = pd.to_datetime(df_raw["sale_date"])
        df_raw["year"]  = df_raw["sale_date"].dt.year
        df_raw["month"] = df_raw["sale_date"].dt.month

        agg_dict = {
            "sales_amount_realistic": ("sales_amount_realistic", "sum"),
            "quantity_realistic":     ("quantity_realistic", "sum") if "quantity_realistic" in df_raw.columns else ("quantity", "sum"),
            "price_realistic":        ("price_realistic", "mean") if "price_realistic" in df_raw.columns else ("sales_amount_realistic", "mean"),
            "num_transactions":       ("sale_date", "count"),
            "promo_flag":             ("promo_flag", "mean"),
        }
        # Add optional columns
        for col in ["gdp_per_capita", "inflation_rate", "exchange_rate", "internet_usage_pct"]:
            if col in df_raw.columns:
                agg_dict[col] = (col, "mean")
        if "product_id" in df_raw.columns:
            agg_dict["num_unique_products"] = ("product_id", "nunique")
        if "category_name" in df_raw.columns:
            agg_dict["num_categories"] = ("category_name", "nunique")

        group_cols = ["store_id", "year", "month"]
        if "store_name" in df_raw.columns:
            group_cols.insert(1, "store_name")
        if "country_norm_mapped" in df_raw.columns:
            group_cols.insert(2, "country_norm_mapped")

        dm = df_raw.groupby(group_cols).agg(**agg_dict).reset_index()
        dm["date"] = pd.to_datetime(dm[["year", "month"]].assign(day=1))
        dm = dm.sort_values(["store_id", "date"]).reset_index(drop=True)

        # Fix quantity column name
        if "quantity_realistic" not in dm.columns and "quantity" in df_raw.columns:
            dm.rename(columns={"quantity": "quantity_realistic"}, inplace=True, errors="ignore")
        if "quantity_realistic" not in dm.columns:
            dm["quantity_realistic"] = dm["sales_amount_realistic"]

        # --- 2. Feature engineering ---
        # Cyclical
        dm["month_sin"] = np.sin(2 * np.pi * dm["month"] / 12)
        dm["month_cos"] = np.cos(2 * np.pi * dm["month"] / 12)
        dm["quarter"] = dm["month"].apply(lambda m: (m - 1) // 3 + 1)
        dm["quarter_sin"] = np.sin(2 * np.pi * dm["quarter"] / 4)
        dm["quarter_cos"] = np.cos(2 * np.pi * dm["quarter"] / 4)
        dm["is_holiday_season"] = dm["month"].isin([11, 12]).astype(int)
        dm["is_launch_season"]  = dm["month"].isin([9, 10]).astype(int)

        # Economic rates of change
        if "country_norm_mapped" in dm.columns:
            for col in ["gdp_per_capita", "inflation_rate", "exchange_rate", "internet_usage_pct"]:
                if col in dm.columns:
                    change_col = col.replace("_per_capita", "").replace("_rate", "").replace("_pct", "") + "_change"
                    dm[change_col] = (
                        dm.groupby("country_norm_mapped")[col]
                        .pct_change()
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0)
                    )

        # Lag features
        LAG_MONTHS = [1, 2, 3, 4, 5, 6]
        for lag in LAG_MONTHS:
            dm[f"sales_lag_{lag}"] = dm.groupby("store_id")["sales_amount_realistic"].shift(lag)
            dm[f"qty_lag_{lag}"]   = dm.groupby("store_id")["quantity_realistic"].shift(lag)
        dm["sales_lag_12"] = dm.groupby("store_id")["sales_amount_realistic"].shift(12)
        dm["qty_lag_12"]   = dm.groupby("store_id")["quantity_realistic"].shift(12)

        # Rolling windows
        WINDOWS = [3, 6, 12]
        for w in WINDOWS:
            grp = dm.groupby("store_id")["sales_amount_realistic"]
            dm[f"sales_roll_mean_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            dm[f"sales_roll_std_{w}"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
            dm[f"sales_roll_min_{w}"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).min())
            dm[f"sales_roll_max_{w}"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        for w in [3, 6]:
            dm[f"qty_roll_mean_{w}"] = dm.groupby("store_id")["quantity_realistic"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

        # Trend & momentum
        dm["sales_mom_pct"] = (
            dm.groupby("store_id")["sales_amount_realistic"]
            .pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        dm["sales_expanding_mean"] = dm.groupby("store_id")["sales_amount_realistic"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        dm["sales_lag1_vs_roll6"] = (
            dm["sales_lag_1"] / dm["sales_roll_mean_6"].replace(0, np.nan)
        ).fillna(1)

        # Encoding
        le_store = LabelEncoder()
        dm["store_encoded"] = le_store.fit_transform(dm["store_id"].astype(str))
        if "country_norm_mapped" in dm.columns:
            le_country = LabelEncoder()
            dm["country_encoded"] = le_country.fit_transform(dm["country_norm_mapped"].astype(str))
        else:
            dm["country_encoded"] = 0
        dm["store_target_enc"] = dm.groupby("store_id")["sales_amount_realistic"].transform("mean")

        # Ensure columns exist
        for col in ["num_transactions", "num_unique_products", "num_categories",
                     "price_realistic", "gdp_per_capita", "inflation_rate",
                     "exchange_rate", "internet_usage_pct",
                     "gdp_change", "inflation_change", "exchange_change", "internet_usage_change"]:
            if col not in dm.columns:
                dm[col] = 0

        # --- 3. Cleanup ---
        dm["row_num"] = dm.groupby("store_id").cumcount()
        df_clean = dm[dm["row_num"] >= 6].copy()
        df_clean.drop(columns="row_num", inplace=True)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[num_cols] = df_clean[num_cols].fillna(0)

        # --- 4. Features & split ---
        TARGET = "sales_amount_realistic"
        FEATURES = [
            "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_4", "sales_lag_5", "sales_lag_6", "sales_lag_12",
            "qty_lag_1", "qty_lag_2", "qty_lag_3", "qty_lag_4", "qty_lag_5", "qty_lag_6", "qty_lag_12",
            "sales_roll_mean_3", "sales_roll_std_3", "sales_roll_min_3", "sales_roll_max_3",
            "sales_roll_mean_6", "sales_roll_std_6", "sales_roll_min_6", "sales_roll_max_6",
            "sales_roll_mean_12", "sales_roll_std_12", "sales_roll_min_12", "sales_roll_max_12",
            "qty_roll_mean_3", "qty_roll_mean_6",
            "sales_mom_pct", "sales_expanding_mean", "sales_lag1_vs_roll6",
            "price_realistic", "promo_flag",
            "month_sin", "month_cos", "quarter_sin", "quarter_cos", "year",
            "is_holiday_season", "is_launch_season",
            "gdp_per_capita", "inflation_rate", "exchange_rate", "internet_usage_pct",
            "gdp_change", "inflation_change", "exchange_change", "internet_usage_change",
            "store_encoded", "country_encoded", "store_target_enc",
            "num_transactions", "num_unique_products", "num_categories",
        ]
        available = [f for f in FEATURES if f in df_clean.columns]

        CUTOFF = df_clean["date"].max() - pd.DateOffset(months=5)
        train_m = df_clean["date"] < CUTOFF
        test_m  = df_clean["date"] >= CUTOFF

        X_train, y_train = df_clean.loc[train_m, available], df_clean.loc[train_m, TARGET]
        X_test, y_test   = df_clean.loc[test_m, available],  df_clean.loc[test_m, TARGET]

        # --- 5. Train tuned CatBoost ---
        model = CatBoostRegressor(
            iterations=800, learning_rate=0.1, depth=4,
            loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)
        y_pred = model.predict(X_test)

        metrics = {
            "MAE":  mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100,
            "R²":   r2_score(y_test, y_pred),
        }
        importances = pd.Series(model.feature_importances_, index=available)

        # --- 6. Future forecast ---
        FORECAST_DATE = df_clean["date"].max() + pd.DateOffset(months=1)
        stores = df_clean["store_id"].unique()
        forecasts = []
        for store in stores:
            hist = df_clean[df_clean["store_id"] == store].sort_values("date").tail(12).copy()
            if hist.empty:
                continue
            row = {}
            recent_sales = hist["sales_amount_realistic"].tolist()
            recent_qty   = hist["quantity_realistic"].tolist()
            for lag in LAG_MONTHS:
                row[f"sales_lag_{lag}"] = recent_sales[-lag] if lag <= len(recent_sales) else 0
                row[f"qty_lag_{lag}"]   = recent_qty[-lag]   if lag <= len(recent_qty)   else 0
            row["sales_lag_12"] = recent_sales[-12] if len(recent_sales) >= 12 else 0
            row["qty_lag_12"]   = recent_qty[-12]   if len(recent_qty) >= 12   else 0
            for w in WINDOWS:
                window = recent_sales[-w:] if len(recent_sales) >= w else recent_sales
                row[f"sales_roll_mean_{w}"] = np.mean(window)
                row[f"sales_roll_std_{w}"]  = np.std(window) if len(window) > 1 else 0
                row[f"sales_roll_min_{w}"]  = np.min(window)
                row[f"sales_roll_max_{w}"]  = np.max(window)
            for w in [3, 6]:
                wq = recent_qty[-w:] if len(recent_qty) >= w else recent_qty
                row[f"qty_roll_mean_{w}"] = np.mean(wq)
            row["sales_mom_pct"] = (recent_sales[-1] - recent_sales[-2]) / max(recent_sales[-2], 1) if len(recent_sales) >= 2 else 0
            row["sales_expanding_mean"] = np.mean(recent_sales)
            r6m = np.mean(recent_sales[-6:]) if len(recent_sales) >= 6 else np.mean(recent_sales)
            row["sales_lag1_vs_roll6"] = recent_sales[-1] / max(r6m, 1)
            row["price_realistic"] = hist["price_realistic"].iloc[-1]
            row["promo_flag"]      = hist["promo_flag"].iloc[-1]
            m = FORECAST_DATE.month
            row["month_sin"] = np.sin(2 * np.pi * m / 12)
            row["month_cos"] = np.cos(2 * np.pi * m / 12)
            q = (m - 1) // 3 + 1
            row["quarter_sin"] = np.sin(2 * np.pi * q / 4)
            row["quarter_cos"] = np.cos(2 * np.pi * q / 4)
            row["year"] = FORECAST_DATE.year
            row["is_holiday_season"] = 1 if m in [11, 12] else 0
            row["is_launch_season"]  = 1 if m in [9, 10] else 0
            for col in ["gdp_per_capita", "inflation_rate", "exchange_rate", "internet_usage_pct",
                         "gdp_change", "inflation_change", "exchange_change", "internet_usage_change"]:
                row[col] = hist[col].iloc[-1] if col in hist.columns else 0
            row["store_encoded"]       = hist["store_encoded"].iloc[-1]
            row["country_encoded"]     = hist["country_encoded"].iloc[-1]
            row["store_target_enc"]    = hist["store_target_enc"].iloc[-1]
            row["num_transactions"]    = hist["num_transactions"].iloc[-1]
            row["num_unique_products"] = hist["num_unique_products"].iloc[-1] if "num_unique_products" in hist.columns else 0
            row["num_categories"]      = hist["num_categories"].iloc[-1] if "num_categories" in hist.columns else 0

            X_fut = pd.DataFrame([row])[available]
            pred  = max(model.predict(X_fut)[0], 0)
            sname = hist["store_name"].iloc[-1] if "store_name" in hist.columns else str(store)
            country = hist["country_norm_mapped"].iloc[-1] if "country_norm_mapped" in hist.columns else "N/A"
            forecasts.append({"store_id": store, "store_name": sname, "country": country,
                              "date": FORECAST_DATE, "predicted_sales": pred})

        forecast_df = pd.DataFrame(forecasts)
        hist_agg = df_clean.groupby("date")[TARGET].sum().reset_index()

        return metrics, importances, y_test.values, y_pred, forecast_df, hist_agg, FORECAST_DATE, df_clean.shape[0], len(available)

    # ── Run the pipeline ──────────────────────────────────────────────
    res = train_catboost_model(df)
    metrics, importances, y_test_vals, y_pred_vals, forecast_df, hist_agg, FORECAST_DATE, n_rows, n_feats = res

    # ── Metrics KPIs ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">\U0001F4CA Model Evaluation Metrics</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    _r2_key = "R\u00b2"
    metric_items = [
        ("\U0001F4CF MAE",  f"${metrics['MAE']:,.0f}",  mc1),
        ("\U0001F4D0 RMSE", f"${metrics['RMSE']:,.0f}", mc2),
        ("\U0001F4C9 MAPE", f"{metrics['MAPE']:.2f}%",  mc3),
        ("\U0001F3AF R\u00b2",    f"{metrics[_r2_key]:.4f}",       mc4),
    ]
    for label, value, col in metric_items:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta">Tuned CatBoost (depth=4, iter=800)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance & Scatter in two columns ───────────────────
    col_fi, col_sc = st.columns(2)

    with col_fi:
        st.markdown('<div class="section-header">\U0001F511 Top 20 Feature Importances</div>', unsafe_allow_html=True)
        top20 = importances.nlargest(20).sort_values()
        fig_fi = go.Figure(go.Bar(
            x=top20.values, y=top20.index, orientation="h",
            marker_color=PALETTE[0],
        ))
        fig_fi.update_layout(yaxis=dict(autorange="reversed"))
        style_fig(fig_fi, 480)
        fig_fi.update_layout(title="Feature Importance — Tuned CatBoost")
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_sc:
        st.markdown('<div class="section-header">\U0001F3AF Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=y_test_vals, y=y_pred_vals, mode="markers",
            marker=dict(color=PALETTE[1], size=5, opacity=0.5),
            name="Predictions",
        ))
        mn = min(y_test_vals.min(), y_pred_vals.min())
        mx = max(y_test_vals.max(), y_pred_vals.max())
        fig_sc.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=PALETTE[3], dash="dash"), name="Perfect Fit",
        ))
        style_fig(fig_sc, 480)
        fig_sc.update_layout(
            title=f"Actual vs Predicted (R\u00b2 = {metrics['R\u00b2']:.4f})",
            xaxis_title="Actual Sales ($)", yaxis_title="Predicted Sales ($)",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Future Forecast Chart ─────────────────────────────────────────
    st.markdown(f'<div class="section-header">\U0001F4C8 Sales Forecast — {FORECAST_DATE.strftime("%b %Y")}</div>', unsafe_allow_html=True)
    forecast_total = forecast_df["predicted_sales"].sum()
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=hist_agg["date"], y=hist_agg["sales_amount_realistic"],
        mode="lines+markers", name="Historical (Actual)",
        line=dict(color=PALETTE[5], width=2),
    ))
    fig_fc.add_trace(go.Scatter(
        x=[FORECAST_DATE], y=[forecast_total],
        mode="markers", name=f"CatBoost Forecast ({FORECAST_DATE.strftime('%b %Y')})",
        marker=dict(color=PALETTE[2], size=14, symbol="square"),
    ))
    fig_fc.add_vline(x=FORECAST_DATE, line_dash="dash", line_color="gray", opacity=0.5)
    fig_fc.add_annotation(
        x=FORECAST_DATE, y=forecast_total,
        text=f"${forecast_total:,.0f}", showarrow=True,
        arrowhead=2, arrowcolor=PALETTE[2],
        font=dict(size=13, color=PALETTE[2]),
    )
    style_fig(fig_fc, 420)
    fig_fc.update_layout(
        title="Total Monthly Sales — Historical & Future Forecast (CatBoost)",
        xaxis_title="Date", yaxis_title="Total Sales ($)",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Top / Bottom store forecasts ──────────────────────────────────
    if not forecast_df.empty:
        st.markdown(f'<div class="section-header">\U0001F3E2 Store-Level Forecasts — {FORECAST_DATE.strftime("%b %Y")}</div>', unsafe_allow_html=True)
        col_top, col_bot = st.columns(2)
        with col_top:
            st.markdown("**Top 10 Stores (Highest Predicted Sales)**")
            top10 = forecast_df.nlargest(10, "predicted_sales")[["store_name", "country", "predicted_sales"]].reset_index(drop=True)
            top10["predicted_sales"] = top10["predicted_sales"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(top10, use_container_width=True, hide_index=True)
        with col_bot:
            st.markdown("**Bottom 10 Stores (Lowest Predicted Sales)**")
            bot10 = forecast_df.nsmallest(10, "predicted_sales")[["store_name", "country", "predicted_sales"]].reset_index(drop=True)
            bot10["predicted_sales"] = bot10["predicted_sales"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(bot10, use_container_width=True, hide_index=True)



# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#334155;font-size:0.78rem;padding-bottom:12px;'>
    Apple Sales EDA Dashboard - Built with Streamlit & Plotly - Data: 2021-2025
</div>
""", unsafe_allow_html=True)
