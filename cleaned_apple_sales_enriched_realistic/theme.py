# -*- coding: utf-8 -*-
"""
Shared theme for Apple Retail Ultra High-Contrast (Power BI → Streamlit).

Import in every page:
    from theme import CSS, PALETTE, DECISION_COLORS, style_fig, kpi_card
"""
import streamlit as st
import plotly.graph_objects as go

# ── Color palette (matches Power BI JSON dataColors) ──────────────────
PALETTE = [
    "#00F0FF",  # Cyan        – primary accent
    "#39FF14",  # Neon green  – secondary
    "#FF5252",  # Red         – alert / negative
    "#5E5CE6",  # Purple      – tertiary
    "#FFFFFF",  # White       – neutral
    "#00B4FF",  # Blue        – alternate
    "#A0A0A5",  # Gray        – muted
    "#FFD60A",  # Yellow      – warning
]

# ── Named tokens ──────────────────────────────────────────────────────
BG           = "#000000"   # Canvas background
CARD_BG      = "#1C1C1E"   # Visual / card surface
ACCENT       = "#00F0FF"   # Cyan – tableAccent
ACCENT2      = "#39FF14"   # Neon green
ACCENT3      = "#5E5CE6"   # Purple
TEXT_PRIMARY = "#FFFFFF"
TEXT_MUTED   = "#E1E1E6"
TEXT_SUBTLE  = "#A0A0A5"
BORDER       = "#A0A0A5"
BORDER_ACCENT= "#00F0FF"

# ── CSS string (paste into st.markdown at top of each page) ───────────
CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* ── Canvas & sidebar ── */
.stApp {{ background: {BG}; }}
[data-testid="stSidebar"] {{
    background: {CARD_BG};
    border-right: 1px solid rgba(0,240,255,0.2);
}}
[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}

/* ── KPI cards ── */
.kpi-card {{
    background: {CARD_BG};
    border: 1px solid rgba(0,240,255,0.35);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}}
.kpi-value {{ font-size: 1.8rem; font-weight: 800; color: {ACCENT}; }}
.kpi-label {{ font-size: 0.82rem; color: {TEXT_MUTED}; margin-top: 4px; }}

/* ── Section headers ── */
.section-header {{
    font-size: 1.15rem; font-weight: 700; color: {TEXT_PRIMARY};
    margin: 28px 0 12px 0; padding-bottom: 8px;
    border-bottom: 2px solid rgba(0,240,255,0.4);
}}

/* ── Interactive nav cards (home page) ── */
.nav-card {{
    display: block; text-decoration: none !important;
    background: {CARD_BG};
    border: 1px solid rgba(0,240,255,0.25);
    border-radius: 16px; padding: 28px 24px; margin-bottom: 16px;
    min-height: 170px; cursor: pointer;
    transition: transform 0.22s ease, box-shadow 0.22s ease,
                border-color 0.22s ease, background 0.22s ease;
    position: relative; overflow: hidden;
}}
.nav-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 16px 48px rgba(0,240,255,0.25), 0 0 0 1px rgba(0,240,255,0.5);
    border-color: {ACCENT};
    background: rgba(0,240,255,0.07);
    text-decoration: none !important;
}}
.nav-card-icon {{
    font-size: 2.4rem; margin-bottom: 10px; display: block;
    filter: drop-shadow(0 0 10px rgba(0,240,255,0.4));
    transition: filter 0.22s ease, transform 0.22s ease;
}}
.nav-card:hover .nav-card-icon {{
    filter: drop-shadow(0 0 20px rgba(0,240,255,0.9));
    transform: scale(1.12);
}}
.nav-card-title {{
    font-size: 1.08rem; font-weight: 700; color: {TEXT_PRIMARY}; margin-bottom: 8px;
    transition: color 0.22s ease;
}}
.nav-card:hover .nav-card-title {{ color: {ACCENT}; }}
.nav-card-desc {{ font-size: 0.82rem; color: {TEXT_MUTED}; line-height: 1.5; }}
.nav-card-arrow {{
    position: absolute; bottom: 16px; right: 18px; font-size: 1rem;
    color: rgba(0,240,255,0.0);
    transition: color 0.22s ease, transform 0.22s ease;
}}
.nav-card:hover .nav-card-arrow {{
    color: {ACCENT}; transform: translateX(4px);
}}

/* ── Decision banner (target market page) ── */
.decision-banner {{
    border-radius: 20px; padding: 28px 32px; text-align: center; margin: 24px 0;
}}
.decision-title {{ font-size: 2.4rem; font-weight: 900; letter-spacing: 2px; }}
.decision-sub {{ font-size: 0.95rem; margin-top: 8px; opacity: 0.8; }}
.reason-box {{
    background: {CARD_BG}; border-radius: 14px; padding: 20px 24px;
    border: 1px solid rgba(0,240,255,0.2); margin-top: 12px;
}}
.pro {{ color: {ACCENT2}; font-size: 0.9rem; margin: 4px 0; }}
.con {{ color: #FF5252; font-size: 0.9rem; margin: 4px 0; }}
.metric-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }}
.metric-chip {{
    background: rgba(0,240,255,0.08); border: 1px solid rgba(0,240,255,0.25);
    border-radius: 10px; padding: 10px 18px; min-width: 120px;
}}
.chip-val {{ font-size: 1.2rem; font-weight: 700; color: {ACCENT}; }}
.chip-lab {{ font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 2px; }}
</style>"""


def style_fig(fig, h: int = 400):
    """Apply Ultra High-Contrast theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,28,30,0.6)",
        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=12),
        height=h,
        margin=dict(l=16, r=16, t=42, b=16),
        legend=dict(bgcolor="rgba(28,28,30,0.85)", bordercolor=BORDER, borderwidth=1),
        xaxis=dict(gridcolor="rgba(0,240,255,0.10)", zerolinecolor="rgba(0,240,255,0.2)"),
        yaxis=dict(gridcolor="rgba(0,240,255,0.10)", zerolinecolor="rgba(0,240,255,0.2)"),
        colorway=PALETTE,
    )


def kpi_card(col, value: str, label: str, color: str = ACCENT):
    """Render a KPI card inside a given Streamlit column."""
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{color};">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
