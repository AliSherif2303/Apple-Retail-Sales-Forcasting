# -*- coding: utf-8 -*-
"""Page 4 — Market Clustering: Store Expansion Prediction
Based on: notebooks/market_expansion_stores.ipynb
Predicts how many additional stores each city needs for optimal expansion.
Uses GradientBoosting, RandomForest, and Ridge Regression models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT = APP_DIR.parent
PROC = PROJECT / "data" / "processed"

CHART_BG = "rgba(28,28,30,0.6)"
PAPER_BG = "rgba(0,0,0,0)"
FONT_COLOR = "#E1E1E6"
GRID_COLOR = "rgba(0,240,255,0.10)"
PALETTE = ["#00F0FF","#39FF14","#FF5252","#5E5CE6","#FFFFFF","#00B4FF","#A0A0A5","#FFD60A"]

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

st.markdown("# 🗺️ Market Clustering — Store Expansion Prediction")
st.markdown("*Predicts how many additional stores each city needs using ML models.*")

def _kpi(v, l):
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{v}</div>'
                f'<div class="kpi-label">{l}</div></div>', unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading city sales data...")
def load_data():
    p = PROC / "merged_city_sales_data.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

df = load_data()
if df is None:
    st.error("merged_city_sales_data.csv not found in data/processed/.")
    st.stop()

# ─── Feature Engineering (City x Year Aggregation) ────────────────────
@st.cache_data(show_spinner="Engineering features...")
def build_features(_df):
    agg_dict = {
        'sales_amount_realistic': 'sum',
        'quantity_realistic': ('mean', 'sum'),
        'price_realistic': 'mean',
        'store_id': 'nunique',
        'product_id': 'nunique',
        'gdp_per_capita': 'median',
        'inflation_rate': 'median',
        'internet_usage_pct': 'median',
        'Population': 'median',
        'promo_flag': 'mean',
    }
    # Build city x year aggregation
    city_year = _df.groupby(['city', 'year']).agg(
        total_sales=('sales_amount_realistic', 'sum'),
        total_txn=('sale_id', 'count'),
        avg_quantity=('quantity_realistic', 'mean'),
        avg_price=('price_realistic', 'mean'),
        total_quantity=('quantity_realistic', 'sum'),
        store_count=('store_id', 'nunique'),
        product_count=('product_id', 'nunique'),
        gdp_per_capita=('gdp_per_capita', 'median'),
        inflation_rate=('inflation_rate', 'median'),
        internet_usage_pct=('internet_usage_pct', 'median'),
        population=('Population', 'median'),
        promo_rate=('promo_flag', 'mean'),
    ).reset_index()
    # optional columns
    if 'economic_factor' in _df.columns:
        ef = _df.groupby(['city','year'])['economic_factor'].median().reset_index()
        city_year = city_year.merge(ef, on=['city','year'], how='left')
    else:
        city_year['economic_factor'] = 1.0
    if 'mu_demand' in _df.columns:
        md = _df.groupby(['city','year'])['mu_demand'].mean().reset_index().rename(columns={'mu_demand':'avg_mu_demand'})
        city_year = city_year.merge(md, on=['city','year'], how='left')
    else:
        city_year['avg_mu_demand'] = 0.0
    country_map = _df.groupby('city')['country_norm_mapped'].first().to_dict()
    city_year['country'] = city_year['city'].map(country_map)
    # Derived features
    city_year['revenue_per_store'] = city_year['total_sales'] / city_year['store_count']
    city_year['revenue_per_capita'] = city_year['total_sales'] / city_year['population']
    city_year['txn_per_store'] = city_year['total_txn'] / city_year['store_count']
    city_year['store_density'] = city_year['store_count'] / (city_year['population'] / 1e6)
    city_year['market_size'] = city_year['population'] * city_year['gdp_per_capita']
    city_year['demand_index'] = city_year['avg_mu_demand'] * city_year['population'] / 1e6
    city_year['sales_per_capita_gdp'] = city_year['total_sales'] / city_year['market_size'].replace(0, np.nan)
    city_year = city_year.sort_values(['city','year'])
    city_year['sales_growth'] = city_year.groupby('city')['total_sales'].pct_change().fillna(0)
    city_year['txn_growth'] = city_year.groupby('city')['total_txn'].pct_change().fillna(0)
    # Target variable
    bench_rev = city_year['revenue_per_store'].median()
    bench_den = city_year['store_density'].median()
    city_year['demand_based_stores'] = np.ceil(city_year['total_sales'] / bench_rev)
    city_year['pop_based_stores'] = np.ceil((city_year['population'] / 1e6) * bench_den)
    city_year['optimal_stores'] = np.ceil(0.6 * city_year['demand_based_stores'] + 0.4 * city_year['pop_based_stores'])
    city_year['additional_stores'] = np.maximum(city_year['optimal_stores'] - city_year['store_count'], 0).astype(int)
    city_year['additional_stores'] = city_year['additional_stores'].clip(upper=20)
    return city_year, bench_rev, bench_den

city_year, bench_rev, bench_den = build_features(df)

# ─── Train Models ─────────────────────────────────────────────────────
feature_cols = [
    'total_sales', 'total_txn', 'avg_quantity', 'avg_price',
    'store_count', 'gdp_per_capita', 'inflation_rate', 'internet_usage_pct',
    'population', 'economic_factor', 'promo_rate',
    'revenue_per_store', 'revenue_per_capita', 'txn_per_store',
    'demand_index', 'sales_growth',
]
feature_cols = [c for c in feature_cols if c in city_year.columns]

@st.cache_data(show_spinner="Training models...")
def train_models(_cy, _fcols):
    X = _cy[_fcols].fillna(0)
    y = _cy['additional_stores']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=_fcols, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    models_cfg = {
        'GradientBoosting': GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                                       min_samples_leaf=5, subsample=0.8, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42),
        'Ridge': Ridge(alpha=1.0),
    }
    results = {}
    for name, mdl in models_cfg.items():
        mdl.fit(X_train, y_train)
        cv = cross_val_score(mdl, X_scaled, y, cv=5, scoring='r2')
        train_r2 = r2_score(y_train, mdl.predict(X_train))
        test_r2 = r2_score(y_test, mdl.predict(X_test))
        mae = mean_absolute_error(y_test, mdl.predict(X_test))
        results[name] = {'model': mdl, 'cv_mean': cv.mean(), 'cv_std': cv.std(),
                         'train_r2': train_r2, 'test_r2': test_r2, 'mae': mae}
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    return results, best_name, scaler

model_results, best_model_name, scaler = train_models(city_year, feature_cols)
best_model = model_results[best_model_name]['model']

# ─── Generate Expansion Recommendations ──────────────────────────────
@st.cache_data(show_spinner="Generating recommendations...")
def get_recommendations(_cy, _fcols, _scaler, _best_model):
    latest = _cy[_cy['year'] == _cy['year'].max()].copy()
    latest_X = pd.DataFrame(_scaler.transform(latest[_fcols].fillna(0)), columns=_fcols)
    latest['predicted_additional'] = np.round(_best_model.predict(latest_X)).astype(int).clip(min=0)
    latest['predicted_total'] = latest['store_count'] + latest['predicted_additional']
    results = latest.sort_values('predicted_additional', ascending=False).reset_index(drop=True)
    results.index += 1
    results.index.name = 'Rank'
    return results

recs = get_recommendations(city_year, feature_cols, scaler, best_model)

# ═══════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════
total_new = recs['predicted_additional'].sum()
cities_need = (recs['predicted_additional'] > 0).sum()
cities_ok = len(recs) - cities_need

c1, c2, c3, c4 = st.columns(4)
with c1: _kpi(best_model_name, "Best Model")
with c2: _kpi(str(total_new), "New Stores Recommended")
with c3: _kpi(f"{cities_need}/{len(recs)}", "Cities Need Expansion")
with c4: _kpi(str(cities_ok), "Cities at Optimal")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab_recs, tab_models, tab_fi = st.tabs(["🏗️ Expansion Recommendations", "📊 Model Comparison", "🔬 Feature Importance"])

# ── TAB 1: Recommendations ──
with tab_recs:
    st.markdown('<div class="section-header">🏗️ Store Expansion Recommendations by City</div>', unsafe_allow_html=True)
    expand = recs[recs['predicted_additional'] > 0].copy()
    if len(expand) > 0:
        fig = px.bar(expand.sort_values('predicted_additional'),
                     x='predicted_additional', y='city', orientation='h',
                     title="Additional Stores Recommended per City",
                     color='predicted_additional', color_continuous_scale='RdYlGn_r',
                     text='predicted_additional')
        fig.update_traces(textposition='outside')
        style_fig(fig, max(400, len(expand)*25))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("All cities are at optimal capacity!")

    # Current vs Recommended (Top 20)
    st.markdown('<div class="section-header">📊 Current vs Recommended Store Count (Top 20)</div>', unsafe_allow_html=True)
    top20 = recs.head(20)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name='Current Stores', y=top20['city'], x=top20['store_count'],
                          orientation='h', marker_color='#3498db'))
    fig2.add_trace(go.Bar(name='Additional Needed', y=top20['city'], x=top20['predicted_additional'],
                          orientation='h', marker_color='#2ecc71'))
    fig2.update_layout(barmode='stack', title='Current vs Recommended Store Count (Top 20)')
    style_fig(fig2, 550)
    st.plotly_chart(fig2, use_container_width=True)

    # Full results table
    st.markdown('<div class="section-header">📋 Complete Recommendations</div>', unsafe_allow_html=True)
    display_cols = [c for c in ['city','country','store_count','predicted_additional','predicted_total',
                                'revenue_per_store','population','gdp_per_capita'] if c in recs.columns]
    st.dataframe(recs[display_cols], use_container_width=True, height=400)
    st.download_button("📥 Download Recommendations CSV", recs.to_csv(), "store_expansion_recommendations.csv")

# ── TAB 2: Model Comparison ──
with tab_models:
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
    comp_data = []
    for name, res in model_results.items():
        comp_data.append({
            'Model': name, 'Train R²': f"{res['train_r2']:.4f}",
            'Test R²': f"{res['test_r2']:.4f}", 'MAE (stores)': f"{res['mae']:.2f}",
            'CV R² Mean': f"{res['cv_mean']:.4f}", 'CV R² Std': f"{res['cv_std']:.4f}",
            'Best': '✅' if name == best_model_name else ''
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    # Visual comparison
    names = list(model_results.keys())
    test_r2s = [model_results[n]['test_r2'] for n in names]
    maes = [model_results[n]['mae'] for n in names]

    c1, c2 = st.columns(2)
    with c1:
        fig3 = px.bar(x=names, y=test_r2s, title="Test R² by Model",
                       color=names, color_discrete_sequence=PALETTE, text=[f"{v:.4f}" for v in test_r2s])
        fig3.update_traces(textposition='outside')
        style_fig(fig3, 350)
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        fig4 = px.bar(x=names, y=maes, title="MAE (stores) by Model",
                       color=names, color_discrete_sequence=PALETTE[3:], text=[f"{v:.2f}" for v in maes])
        fig4.update_traces(textposition='outside')
        style_fig(fig4, 350)
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 3: Feature Importance ──
with tab_fi:
    st.markdown(f'<div class="section-header">🔬 Top Feature Importances ({best_model_name})</div>', unsafe_allow_html=True)
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_, index=feature_cols)
    elif hasattr(best_model, 'coef_'):
        fi = pd.Series(np.abs(best_model.coef_), index=feature_cols)
    else:
        fi = pd.Series(dtype=float)

    if len(fi) > 0:
        fi = fi.sort_values(ascending=False).head(10)
        fig5 = px.bar(x=fi.values, y=fi.index, orientation='h',
                       title=f"Top 10 Features ({best_model_name})",
                       color=fi.values, color_continuous_scale="Purples")
        fig5.update_layout(yaxis=dict(autorange="reversed"))
        style_fig(fig5, 400)
        st.plotly_chart(fig5, use_container_width=True)
