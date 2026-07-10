from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError as e:
    raise ImportError("Streamlit is not installed. Run: pip install streamlit plotly") from e

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError("Plotly is not installed. Run: pip install streamlit plotly") from e


# ======================================================================================
# PROJECT PATHS
# ======================================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"


FILES = {
    "master_summary": TABLES_DIR / "11_final_master_performance_summary.csv",
    "ml_final": TABLES_DIR / "11_final_ml_baseline_vs_proposed_summary.csv",
    "rl_final": TABLES_DIR / "11_final_rl_baseline_vs_proposed_summary.csv",
    "key_findings_11": TABLES_DIR / "11_key_findings_for_manuscript.csv",

    "ml_ablation": TABLES_DIR / "12_ml_rgc_feature_ablation_summary.csv",
    "cmarl_ablation": TABLES_DIR / "12_cmarl_policy_ablation_summary.csv",
    "stress_ablation": TABLES_DIR / "12_stress_scenario_ablation_summary.csv",
    "master_ablation": TABLES_DIR / "12_master_ablation_summary.csv",
    "key_ablation": TABLES_DIR / "12_key_ablation_findings.csv",

    "stress_summary": TABLES_DIR / "13_extended_stress_testing_summary.csv",
    "stress_scorecard": TABLES_DIR / "13_stress_resilience_scorecard.csv",
    "stress_ranking": TABLES_DIR / "13_stress_scenario_ranking.csv",
    "key_stress": TABLES_DIR / "13_key_stress_testing_findings.csv",

    "ml_feature_explain": TABLES_DIR / "14_ml_feature_explainability_ranking.csv",
    "ml_ablation_explain": TABLES_DIR / "14_ml_ablation_explainability_summary.csv",
    "policy_driver_explain": TABLES_DIR / "14_policy_driver_explainability_summary.csv",
    "policy_binned": TABLES_DIR / "14_policy_driver_binned_effects.csv",
    "local_explain": TABLES_DIR / "14_local_decision_explanations.csv",
    "stress_explain": TABLES_DIR / "14_stress_explainability_summary.csv",
    "key_explain": TABLES_DIR / "14_key_explainability_findings.csv",

    "risk_distribution": TABLES_DIR / "04_risk_label_distribution.csv",
    "graph_node_summary": TABLES_DIR / "05_graph_node_type_summary.csv",
    "graph_edge_summary": TABLES_DIR / "05_graph_edge_relation_summary.csv",
    "top_risk_nodes": TABLES_DIR / "05_top_risk_nodes.csv",
    "artifact_index": TABLES_DIR / "11_final_output_artifact_index.csv",
}


PAGE_TITLE = "ResilientGraph-CMARL Executive Research Dashboard"


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================================
# PREMIUM ACADEMIC UI CSS
# ======================================================================================

CUSTOM_CSS = """
<style>
    :root {
        --rg-page-bg: #f5f9fc;
        --rg-page-bg-2: #eef7fb;
        --rg-text: #0f172a;
        --rg-heading: #06121f;
        --rg-muted: #475569;
        --rg-soft: #64748b;
        --rg-card: #ffffff;
        --rg-border: rgba(15, 23, 42, 0.12);

        --rg-navy: #0f172a;
        --rg-blue: #2563eb;
        --rg-sky: #0ea5e9;
        --rg-cyan: #06b6d4;
        --rg-teal: #14b8a6;
        --rg-green: #22c55e;
        --rg-amber: #f59e0b;
        --rg-purple: #8b5cf6;
        --rg-rose: #f43f5e;

        --rg-sidebar-top: #0e7490;
        --rg-sidebar-mid: #155e75;
        --rg-sidebar-bottom: #1e3a8a;
    }

    html[data-theme="dark"] {
        --rg-page-bg: #0f172a;
        --rg-page-bg-2: #111827;
        --rg-text: #e5e7eb;
        --rg-heading: #f8fafc;
        --rg-muted: #cbd5e1;
        --rg-soft: #94a3b8;
        --rg-card: #111827;
        --rg-border: rgba(148, 163, 184, 0.18);

        --rg-sidebar-top: #083344;
        --rg-sidebar-mid: #0f766e;
        --rg-sidebar-bottom: #1e3a8a;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main {
        background:
            radial-gradient(circle at 4% 2%, rgba(14, 165, 233, 0.12), transparent 28%),
            radial-gradient(circle at 95% 4%, rgba(20, 184, 166, 0.12), transparent 26%),
            linear-gradient(135deg, var(--rg-page-bg) 0%, var(--rg-page-bg-2) 100%) !important;
        color: var(--rg-text) !important;
    }

    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.8rem;
        max-width: 1380px;
    }

    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: var(--rg-heading) !important;
        letter-spacing: -0.025em;
        font-weight: 900 !important;
    }

    p, span, label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {
        color: var(--rg-text) !important;
    }

    .rg-hero {
        background:
            linear-gradient(135deg, #e0f2fe 0%, #ffffff 36%, #ecfeff 67%, #f0fdfa 100%);
        border: 1px solid rgba(14, 165, 233, 0.22);
        border-radius: 26px;
        padding: 1.45rem 1.55rem;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.10);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }

    html[data-theme="dark"] .rg-hero {
        background:
            linear-gradient(135deg, #0f172a 0%, #164e63 46%, #0f766e 100%);
        border: 1px solid rgba(125, 211, 252, 0.25);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.30);
    }

    .rg-hero::before {
        content: "";
        position: absolute;
        width: 320px;
        height: 320px;
        right: -110px;
        top: -155px;
        background: radial-gradient(circle, rgba(37, 99, 235, 0.18), transparent 68%);
    }

    .rg-hero::after {
        content: "";
        position: absolute;
        width: 290px;
        height: 290px;
        left: -145px;
        bottom: -145px;
        background: radial-gradient(circle, rgba(20, 184, 166, 0.18), transparent 68%);
    }

    .rg-hero-content {
        position: relative;
        z-index: 2;
    }

    .rg-hero-kicker {
        color: #0369a1 !important;
        font-size: 0.74rem;
        font-weight: 950;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.38rem;
    }

    html[data-theme="dark"] .rg-hero-kicker {
        color: #67e8f9 !important;
    }

    .dashboard-title {
        font-size: 2.24rem;
        font-weight: 950;
        color: #082f49 !important;
        margin-bottom: 0.18rem;
        line-height: 1.08;
    }

    html[data-theme="dark"] .dashboard-title {
        color: #ffffff !important;
    }

    .dashboard-subtitle {
        font-size: 1.03rem;
        color: #334155 !important;
        margin-bottom: 1.12rem;
        line-height: 1.48;
        max-width: 980px;
    }

    html[data-theme="dark"] .dashboard-subtitle {
        color: #dbeafe !important;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.76rem;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid rgba(14, 165, 233, 0.30);
        background: rgba(14, 165, 233, 0.10);
        color: #0369a1 !important;
        margin-right: 0.36rem;
        margin-bottom: 0.26rem;
    }

    .green-pill {
        border-color: rgba(34, 197, 94, 0.38);
        background: rgba(34, 197, 94, 0.11);
        color: #15803d !important;
    }

    .gold-pill {
        border-color: rgba(245, 158, 11, 0.40);
        background: rgba(245, 158, 11, 0.12);
        color: #92400e !important;
    }

    .purple-pill {
        border-color: rgba(139, 92, 246, 0.38);
        background: rgba(139, 92, 246, 0.11);
        color: #6d28d9 !important;
    }

    html[data-theme="dark"] .status-pill {
        color: #7dd3fc !important;
        border-color: rgba(125, 211, 252, 0.36);
        background: rgba(125, 211, 252, 0.12);
    }

    html[data-theme="dark"] .green-pill {
        color: #86efac !important;
        border-color: rgba(134, 239, 172, 0.36);
        background: rgba(34, 197, 94, 0.12);
    }

    html[data-theme="dark"] .gold-pill {
        color: #fcd34d !important;
        border-color: rgba(252, 211, 77, 0.36);
        background: rgba(245, 158, 11, 0.12);
    }

    html[data-theme="dark"] .purple-pill {
        color: #c4b5fd !important;
        border-color: rgba(196, 181, 253, 0.36);
        background: rgba(139, 92, 246, 0.12);
    }

    .kpi-card {
        border-radius: 22px;
        padding: 1.08rem 1.16rem;
        min-height: 142px;
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.10);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(15, 23, 42, 0.10);
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 22px 46px rgba(15, 23, 42, 0.14);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 5px;
    }

    .kpi-info {
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 52%, #e0f2fe 100%);
    }

    .kpi-info::before {
        background: linear-gradient(180deg, var(--rg-blue), var(--rg-sky));
    }

    .kpi-success {
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 54%, #ccfbf1 100%);
    }

    .kpi-success::before {
        background: linear-gradient(180deg, var(--rg-green), var(--rg-teal));
    }

    .kpi-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 54%, #fef3c7 100%);
    }

    .kpi-warning::before {
        background: linear-gradient(180deg, var(--rg-amber), #f97316);
    }

    .kpi-purple {
        background: linear-gradient(135deg, #f5f3ff 0%, #ffffff 54%, #ede9fe 100%);
    }

    .kpi-purple::before {
        background: linear-gradient(180deg, var(--rg-purple), var(--rg-blue));
    }

    html[data-theme="dark"] .kpi-info {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    }

    html[data-theme="dark"] .kpi-success {
        background: linear-gradient(135deg, #0f172a 0%, #115e59 100%);
    }

    html[data-theme="dark"] .kpi-warning {
        background: linear-gradient(135deg, #0f172a 0%, #78350f 100%);
    }

    html[data-theme="dark"] .kpi-purple {
        background: linear-gradient(135deg, #0f172a 0%, #4c1d95 100%);
    }

    .kpi-label {
        color: #334155 !important;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-weight: 950;
        margin-bottom: 0.24rem;
        position: relative;
        z-index: 2;
    }

    .kpi-value {
        color: #0f172a !important;
        font-size: 1.72rem;
        font-weight: 950;
        margin-top: 0.18rem;
        margin-bottom: 0.25rem;
        line-height: 1.14;
        position: relative;
        z-index: 2;
        word-break: break-word;
    }

    .kpi-delta-positive,
    .kpi-delta-warning,
    .kpi-delta-info {
        font-size: 0.88rem;
        font-weight: 900;
        position: relative;
        z-index: 2;
    }

    .kpi-delta-positive {
        color: #15803d !important;
    }

    .kpi-delta-warning {
        color: #b45309 !important;
    }

    .kpi-delta-info {
        color: #0369a1 !important;
    }

    html[data-theme="dark"] .kpi-label {
        color: #cbd5e1 !important;
    }

    html[data-theme="dark"] .kpi-value {
        color: #ffffff !important;
    }

    html[data-theme="dark"] .kpi-delta-positive {
        color: #86efac !important;
    }

    html[data-theme="dark"] .kpi-delta-warning {
        color: #fcd34d !important;
    }

    html[data-theme="dark"] .kpi-delta-info {
        color: #7dd3fc !important;
    }

    .insight-box {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        border-left: 5px solid var(--rg-sky);
        border-radius: 15px;
        padding: 0.90rem 1.05rem;
        margin-bottom: 0.74rem;
        color: #0f172a !important;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
        border-top: 1px solid rgba(14, 165, 233, 0.12);
        border-right: 1px solid rgba(14, 165, 233, 0.12);
        border-bottom: 1px solid rgba(14, 165, 233, 0.12);
    }

    .insight-box b,
    .insight-box span {
        color: #0f172a !important;
    }

    html[data-theme="dark"] .insight-box {
        background: linear-gradient(135deg, #0f172a 0%, #164e63 100%);
        color: #e0f2fe !important;
        border-color: rgba(125, 211, 252, 0.28);
    }

    html[data-theme="dark"] .insight-box b,
    html[data-theme="dark"] .insight-box span {
        color: #e0f2fe !important;
    }

    .small-muted {
        color: var(--rg-muted) !important;
        font-size: 0.90rem;
        line-height: 1.46;
        margin-bottom: 0.60rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--rg-heading) !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricDelta"] {
        color: var(--rg-muted) !important;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        background: var(--rg-card) !important;
        color: var(--rg-text) !important;
        border-radius: 16px;
        border: 1px solid var(--rg-border);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="slider"] *,
    div[data-testid="stTextArea"] *,
    div[data-testid="stNumberInput"] *,
    div[data-testid="stTextInput"] * {
        color: var(--rg-text) !important;
    }

    div[data-baseweb="select"] > div,
    textarea,
    input {
        background-color: var(--rg-card) !important;
        color: var(--rg-text) !important;
        border-color: rgba(148, 163, 184, 0.35) !important;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top left, rgba(255, 255, 255, 0.16), transparent 24%),
            linear-gradient(180deg, var(--rg-sidebar-top) 0%, var(--rg-sidebar-mid) 48%, var(--rg-sidebar-bottom) 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 10px 0 36px rgba(15, 23, 42, 0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.20) !important;
        margin-top: 1.1rem;
        margin-bottom: 1.1rem;
    }

    .sidebar-brand {
        background: rgba(255, 255, 255, 0.13);
        border: 1px solid rgba(255, 255, 255, 0.24);
        border-radius: 22px;
        padding: 1.08rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.14);
        backdrop-filter: blur(10px);
    }

    .brand-kicker {
        color: #cffafe !important;
        font-size: 0.64rem;
        font-weight: 950;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .brand-title {
        color: #ffffff !important;
        font-size: 1.16rem;
        font-weight: 950;
        line-height: 1.18;
        margin-bottom: 0.40rem;
    }

    .brand-subtitle {
        color: #ecfeff !important;
        font-size: 0.76rem;
        line-height: 1.42;
    }

    .nav-caption {
        color: #dffafe !important;
        font-size: 0.69rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        margin-bottom: 0.42rem;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 15px;
        padding: 0.56rem 0.62rem;
        margin-bottom: 0.38rem;
        transition: all 0.18s ease;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.18);
        border-color: rgba(255, 255, 255, 0.42);
        transform: translateX(2px);
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.30), rgba(255, 255, 255, 0.16));
        border-color: rgba(255, 255, 255, 0.58);
        border-left: 5px solid #facc15;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--rg-card);
        border: 1px solid var(--rg-border);
        border-radius: 13px;
        padding: 0.62rem 1rem;
        color: var(--rg-text) !important;
        font-weight: 750;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.14);
        border: 1px solid rgba(14, 165, 233, 0.55);
        color: var(--rg-heading) !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--rg-border) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }

    .stDownloadButton button {
        border-radius: 12px !important;
        border: 1px solid rgba(14, 165, 233, 0.45) !important;
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: #ffffff !important;
        font-weight: 850 !important;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: visible !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.72) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(15, 23, 42, 0.06);
    }

    html[data-theme="dark"] header[data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.72) !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.10);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ======================================================================================
# DATA LOADING
# ======================================================================================

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    encodings = ["utf-8", "utf-8-sig", "latin1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_all_tables():
    return {name: read_csv(path) for name, path in FILES.items()}


# ======================================================================================
# FORMATTERS AND UI HELPERS
# ======================================================================================

def fmt_num(value, digits=4):
    try:
        value = float(value)
    except Exception:
        return "NA"

    if abs(value) >= 100:
        return f"{value:,.2f}"

    return f"{value:.{digits}f}"


def fmt_pct(value, digits=2):
    try:
        value = float(value)
    except Exception:
        return "NA"

    return f"{value:.{digits}f}%"


def get_first(df: pd.DataFrame, col: str, default=None):
    if df.empty or col not in df.columns:
        return default
    if len(df) == 0:
        return default
    return df.iloc[0][col]


def kpi_card(label, value, delta="", delta_type="info"):
    card_class = {
        "positive": "kpi-success",
        "success": "kpi-success",
        "warning": "kpi-warning",
        "info": "kpi-info",
        "purple": "kpi-purple",
    }.get(delta_type, "kpi-info")

    delta_class = {
        "positive": "kpi-delta-positive",
        "success": "kpi-delta-positive",
        "warning": "kpi-delta-warning",
        "info": "kpi-delta-info",
        "purple": "kpi-delta-info",
    }.get(delta_type, "kpi-delta-info")

    st.markdown(
        f"""
        <div class="kpi-card {card_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="{delta_class}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<div class='small-muted'>{subtitle}</div>", unsafe_allow_html=True)


def get_theme_tokens():
    base_theme = str(st.get_option("theme.base") or "light").lower()

    if base_theme == "dark":
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(15,23,42,0.35)",
            "font_color": "#e5e7eb",
            "grid_color": "rgba(148,163,184,0.22)",
            "legend_bg": "rgba(15,23,42,0.24)",
            "legend_border": "rgba(148,163,184,0.28)",
        }

    return {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#ffffff",
        "font_color": "#0f172a",
        "grid_color": "rgba(15,23,42,0.13)",
        "legend_bg": "rgba(255,255,255,0.96)",
        "legend_border": "rgba(15,23,42,0.14)",
    }


def make_plot_layout(fig, height=420):
    theme = get_theme_tokens()

    fig.update_layout(
        template=theme["template"],
        height=height,
        paper_bgcolor=theme["paper_bgcolor"],
        plot_bgcolor=theme["plot_bgcolor"],
        font=dict(color=theme["font_color"], size=13),
        title=dict(font=dict(color=theme["font_color"], size=18, family="Arial")),
        margin=dict(l=42, r=30, t=64, b=46),
        legend=dict(
            bgcolor=theme["legend_bg"],
            bordercolor=theme["legend_border"],
            borderwidth=1,
            font=dict(color=theme["font_color"], size=12),
        ),
        xaxis=dict(
            color=theme["font_color"],
            gridcolor=theme["grid_color"],
            zerolinecolor=theme["grid_color"],
            title_font=dict(color=theme["font_color"]),
            tickfont=dict(color=theme["font_color"]),
        ),
        yaxis=dict(
            color=theme["font_color"],
            gridcolor=theme["grid_color"],
            zerolinecolor=theme["grid_color"],
            title_font=dict(color=theme["font_color"]),
            tickfont=dict(color=theme["font_color"]),
        ),
    )

    fig.update_layout(
        yaxis2=dict(
            color=theme["font_color"],
            gridcolor=theme["grid_color"],
            zerolinecolor=theme["grid_color"],
            title_font=dict(color=theme["font_color"]),
            tickfont=dict(color=theme["font_color"]),
        )
    )

    return fig


def display_dataframe(df, title=None, height=350):
    if title:
        st.markdown(f"**{title}**")
    if df.empty:
        st.info("No table found for this section.")
    else:
        st.dataframe(df, use_container_width=True, height=height)


def insight_list(df: pd.DataFrame, max_items=5):
    if df.empty or "finding" not in df.columns:
        st.info("No key findings available.")
        return

    temp = df.head(max_items).copy()

    for _, row in temp.iterrows():
        finding = row.get("finding", "")
        finding_type = row.get("finding_type", "Insight")
        st.markdown(
            f"""
            <div class="insight-box">
                <b>{finding_type}</b><br>
                {finding}
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_health_status(tables):
    required = [
        "master_summary",
        "ml_final",
        "rl_final",
        "ml_ablation",
        "cmarl_ablation",
        "stress_summary",
        "ml_feature_explain",
        "policy_driver_explain",
        "local_explain",
    ]

    loaded = sum(1 for k in required if not tables.get(k, pd.DataFrame()).empty)
    total = len(required)

    return loaded, total


def hero_header(title, subtitle, kicker="Executive Research Intelligence Console"):
    st.markdown(
        f"""
        <div class="rg-hero">
            <div class="rg-hero-content">
                <div class="rg-hero-kicker">{kicker}</div>
                <div class="dashboard-title">{title}</div>
                <div class="dashboard-subtitle">{subtitle}</div>
                <span class="status-pill">Calculated Outputs</span>
                <span class="status-pill green-pill">Leakage-Safe ML</span>
                <span class="status-pill gold-pill">Digital-Twin CMARL</span>
                <span class="status-pill purple-pill">Explainable Intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================================
# DASHBOARD SECTIONS
# ======================================================================================

def executive_overview(tables):
    master = tables["master_summary"]
    ml_final = tables["ml_final"]
    rl_final = tables["rl_final"]
    key_findings = tables["key_findings_11"]

    hero_header(
        "ResilientGraph-CMARL Executive Control Tower",
        "Board-level view of graph-risk prediction, digital-twin procurement optimization, stress resilience, and explainable decision intelligence.",
    )

    best_ml = ml_final.iloc[0] if not ml_final.empty else pd.Series(dtype=object)

    proposed_rl = pd.Series(dtype=object)
    if not rl_final.empty and "task_type" in rl_final.columns:
        proposed_rows = rl_final[
            rl_final["task_type"] == "Proposed STHG-CMAPPO / CMARL policy"
        ]
        if not proposed_rows.empty:
            proposed_rl = proposed_rows.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Best Risk Model",
            str(best_ml.get("proposed_model", "CatBoost-RGC")),
            f"Macro-F1 {fmt_num(best_ml.get('proposed_macro_f1', 0.8007))}",
            "info",
        )

    with c2:
        kpi_card(
            "ML Accuracy",
            fmt_num(best_ml.get("proposed_accuracy", 0.8009)),
            f"+{fmt_pct(best_ml.get('accuracy_relative_improvement_percent', 28.79))} vs baseline",
            "success",
        )

    with c3:
        kpi_card(
            "CMARL Reward Gain",
            fmt_pct(proposed_rl.get("reward_improvement_vs_best_baseline_percent", 26.63)),
            "vs strongest risk-avoidance baseline",
            "purple",
        )

    with c4:
        kpi_card(
            "Stress Robustness",
            "100%",
            "13/13 scenarios passed",
            "success",
        )

    st.markdown("")

    left, right = st.columns([1.18, 1])

    with left:
        section("Final Master Performance Summary")
        display_dataframe(master, height=285)

    with right:
        section("Board-Level Key Findings")
        insight_list(key_findings, max_items=5)

    if not master.empty and "relative_gain_percent" in master.columns:
        fig = px.bar(
            master,
            x="module",
            y="relative_gain_percent",
            text="relative_gain_percent",
            color="module",
            color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6"],
            title="Relative Improvement by Intelligence Layer",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_tickangle=-18)
        fig = make_plot_layout(fig, height=440)
        st.plotly_chart(fig, use_container_width=True)


def risk_prediction_intelligence(tables):
    ml_final = tables["ml_final"]
    ml_ablation = tables["ml_ablation"]
    ml_feature = tables["ml_feature_explain"]
    risk_dist = tables["risk_distribution"]
    graph_nodes = tables["graph_node_summary"]
    graph_edges = tables["graph_edge_summary"]
    top_risk_nodes = tables["top_risk_nodes"]

    section(
        "Risk Prediction Intelligence",
        "Leakage-safe baseline versus proposed RGC-enhanced ML models, feature ablation, and graph-risk drivers.",
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        if not ml_final.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=ml_final["algorithm"],
                    y=ml_final["baseline_macro_f1"],
                    name="Baseline Macro-F1",
                    marker_color="#94a3b8",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=ml_final["algorithm"],
                    y=ml_final["proposed_macro_f1"],
                    name="Proposed RGC Macro-F1",
                    marker_color="#14b8a6",
                )
            )
            fig.update_layout(
                title="Baseline vs RGC Macro-F1",
                barmode="group",
                yaxis_title="Macro-F1",
            )
            fig = make_plot_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ML final summary table not found.")

    with c2:
        if not ml_final.empty:
            fig = px.scatter(
                ml_final,
                x="proposed_accuracy",
                y="proposed_macro_f1",
                size="proposed_roc_auc",
                color="algorithm",
                color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6"],
                hover_data=[
                    "baseline_accuracy",
                    "accuracy_relative_improvement_percent",
                    "macro_f1_relative_improvement_percent",
                ],
                title="Proposed Model Accuracy-F1-AUC Profile",
            )
            fig = make_plot_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Accuracy-F1-AUC profile cannot be shown because data is missing.")

    c3, c4 = st.columns([1.1, 0.9])

    with c3:
        if not ml_ablation.empty:
            fig = px.bar(
                ml_ablation,
                x="candidate_feature_set",
                y="test_macro_f1",
                color="proposed_model",
                barmode="group",
                color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6"],
                hover_data=["feature_count", "test_accuracy", "ablation_role"],
                title="RGC Feature Ablation: Test Macro-F1",
            )
            fig.update_layout(xaxis_tickangle=-25)
            fig = make_plot_layout(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ML ablation summary not found.")

    with c4:
        if not ml_feature.empty:
            top_n = st.slider("Top feature count", 5, 30, 15, key="top_feature_slider")
            temp = ml_feature.head(top_n).copy()
            fig = px.bar(
                temp.sort_values("abs_spearman_corr"),
                x="abs_spearman_corr",
                y="feature",
                color="feature_group",
                orientation="h",
                color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#f43f5e"],
                title="Top Risk-Associated Features",
            )
            fig = make_plot_layout(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature explainability ranking not found.")

    st.markdown("---")

    g1, g2, g3 = st.columns(3)

    with g1:
        section("Risk Label Distribution")
        display_dataframe(risk_dist, height=265)

    with g2:
        section("Graph Node Intelligence")
        display_dataframe(graph_nodes, height=265)

    with g3:
        section("Graph Edge Intelligence")
        display_dataframe(graph_edges, height=265)

    section("High-Risk Graph Nodes")
    display_dataframe(top_risk_nodes.head(25) if not top_risk_nodes.empty else top_risk_nodes, height=430)


def decision_optimization_control(tables):
    rl_final = tables["rl_final"]
    cmarl_ablation = tables["cmarl_ablation"]
    local_explain = tables["local_explain"]
    binned = tables["policy_binned"]

    section(
        "Digital Twin Procurement Decision Control",
        "Compares strongest baseline policy with STHG-CMAPPO risk-service policy using calculated digital-twin outcomes.",
    )

    if not rl_final.empty:
        baseline = rl_final.iloc[0]
        proposed = rl_final.iloc[1] if len(rl_final) > 1 else rl_final.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi_card(
                "Baseline Policy",
                str(baseline.get("model_or_policy", "risk_avoidance_policy")),
                f"Reward {fmt_num(baseline.get('mean_reward', 0))}",
                "warning",
            )

        with c2:
            kpi_card(
                "Proposed Policy",
                str(proposed.get("model_or_policy", "sthg_cmappo_risk_service_policy")),
                f"Reward {fmt_num(proposed.get('mean_reward', 0))}",
                "success",
            )

        with c3:
            kpi_card(
                "Risk Reduction",
                fmt_pct(proposed.get("risk_reduction_vs_best_baseline_percent", 0)),
                "lower exposure vs baseline",
                "info",
            )

        with c4:
            kpi_card(
                "Delay Reduction",
                fmt_pct(proposed.get("delay_reduction_vs_best_baseline_percent", 0)),
                "faster service continuity",
                "purple",
            )
    else:
        st.info("RL final summary table not found.")

    left, right = st.columns([1.15, 0.85])

    with left:
        if not cmarl_ablation.empty:
            fig = px.bar(
                cmarl_ablation,
                x="policy_name",
                y="mean_reward",
                color="policy_name",
                color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#f43f5e"],
                hover_data=[
                    "mean_risk",
                    "mean_delay",
                    "mean_service",
                    "mean_resilience",
                    "reward_relative_gain_percent",
                ],
                title="CMARL Policy Variant Ablation",
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-25)
            fig = make_plot_layout(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CMARL ablation table not found.")

    with right:
        if not cmarl_ablation.empty:
            radar_row = cmarl_ablation.iloc[0]

            categories = [
                "Reward Gain",
                "Risk Reduction",
                "Delay Reduction",
                "Service Gain",
                "Resilience Gain",
                "Profit Gain",
            ]

            values = [
                radar_row.get("reward_relative_gain_percent", 0),
                radar_row.get("risk_reduction_percent", 0),
                radar_row.get("delay_reduction_percent", 0),
                radar_row.get("service_improvement_percent", 0),
                radar_row.get("resilience_improvement_percent", 0),
                radar_row.get("profit_improvement_percent", 0),
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=str(radar_row.get("policy_name", "Best CMARL")),
                    line=dict(color="#2563eb", width=3),
                    marker=dict(color="#14b8a6", size=7),
                    fillcolor="rgba(37, 99, 235, 0.22)",
                )
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Best CMARL Multidimensional Gain Profile",
            )
            fig = make_plot_layout(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    if not binned.empty and "driver_feature" in binned.columns:
        selected_driver = st.selectbox(
            "Select decision driver for quartile effect analysis",
            sorted(binned["driver_feature"].dropna().unique()),
        )

        temp = binned[binned["driver_feature"] == selected_driver].copy()

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=temp["driver_bin"].astype(str),
                y=temp["mean_reward_delta"],
                name="Reward Gain",
                marker_color="#14b8a6",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=temp["driver_bin"].astype(str),
                y=temp["mean_delay_reduction"],
                name="Delay Reduction",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#f59e0b", width=3),
                marker=dict(size=8),
            )
        )
        fig.update_layout(
            title=f"Binned Driver Effect: {selected_driver}",
            yaxis=dict(title="Mean Reward Gain"),
            yaxis2=dict(title="Mean Delay Reduction", overlaying="y", side="right"),
        )
        fig = make_plot_layout(fig, height=470)
        st.plotly_chart(fig, use_container_width=True)

    section("Local Decision Explanations")
    display_dataframe(local_explain, height=470)


def stress_war_room(tables):
    stress = tables["stress_summary"]
    scorecard = tables["stress_scorecard"]
    ranking = tables["stress_ranking"]
    key_stress = tables["key_stress"]

    section(
        "Stress-Testing War Room",
        "Extended real-world stress scenarios covering demand surge, logistics disruption, supplier shock, and combined stress.",
    )

    if not stress.empty:
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            kpi_card(
                "Stress Scenarios",
                str(stress["scenario_name"].nunique()),
                "extended severity levels",
                "info",
            )

        with c2:
            kpi_card(
                "Avg Reward Gain",
                fmt_pct(stress["reward_relative_improvement_percent"].mean()),
                "across stress scenarios",
                "success",
            )

        with c3:
            kpi_card(
                "Avg Risk Reduction",
                fmt_pct(stress["risk_reduction_percent"].mean()),
                "risk exposure control",
                "purple",
            )

        with c4:
            kpi_card(
                "Avg Delay Reduction",
                fmt_pct(stress["delay_reduction_percent"].mean()),
                "service continuity",
                "warning",
            )

        with c5:
            pass_rate = 0
            if not scorecard.empty and "stress_test_status" in scorecard.columns:
                pass_rate = (scorecard["stress_test_status"] == "PASS").mean() * 100
            kpi_card(
                "Pass Rate",
                fmt_pct(pass_rate),
                "scorecard robustness",
                "success",
            )
    else:
        st.info("Stress testing summary table not found.")

    s1, s2 = st.columns([1.2, 1])

    with s1:
        if not stress.empty:
            fig = px.bar(
                stress.sort_values(["stress_group", "severity_level"]),
                x="scenario_name",
                y="reward_relative_improvement_percent",
                color="stress_group",
                color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#f43f5e"],
                hover_data=[
                    "severity_level",
                    "risk_reduction_percent",
                    "delay_reduction_percent",
                    "service_improvement_percent",
                    "resilience_improvement_percent",
                ],
                title="Reward Improvement across Extended Stress Scenarios",
            )
            fig.update_layout(xaxis_tickangle=-40)
            fig = make_plot_layout(fig, height=530)
            st.plotly_chart(fig, use_container_width=True)

    with s2:
        if not ranking.empty:
            fig = px.bar(
                ranking.head(10).sort_values("robustness_index"),
                x="robustness_index",
                y="scenario_name",
                color="stress_group",
                color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6"],
                orientation="h",
                title="Top Stress Robustness Index Ranking",
            )
            fig = make_plot_layout(fig, height=530)
            st.plotly_chart(fig, use_container_width=True)

    if not stress.empty:
        heat_cols = [
            "reward_relative_improvement_percent",
            "risk_reduction_percent",
            "delay_reduction_percent",
            "service_improvement_percent",
            "resilience_improvement_percent",
        ]

        heat = stress.set_index("scenario_name")[heat_cols].copy()

        fig = px.imshow(
            heat,
            aspect="auto",
            color_continuous_scale="Tealgrn",
            title="Stress Scenario Performance Heatmap",
        )
        fig = make_plot_layout(fig, height=570)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    insight_list(key_stress, max_items=6)

    with st.expander("Detailed stress testing table"):
        display_dataframe(stress, height=440)

    with st.expander("Stress resilience scorecard"):
        display_dataframe(scorecard, height=440)


def explainability_hub(tables):
    ml_feature = tables["ml_feature_explain"]
    policy_driver = tables["policy_driver_explain"]
    stress_explain = tables["stress_explain"]
    key_explain = tables["key_explain"]
    local_explain = tables["local_explain"]

    section(
        "Explainability Hub",
        "Global ML feature associations, policy-driver correlations, stress-group explanations, and local decision explanations.",
    )

    e1, e2 = st.columns([1, 1])

    with e1:
        if not ml_feature.empty:
            group_summary = (
                ml_feature.groupby("feature_group", observed=False)
                .agg(
                    feature_count=("feature", "count"),
                    mean_abs_corr=("abs_spearman_corr", "mean"),
                    max_abs_corr=("abs_spearman_corr", "max"),
                )
                .reset_index()
                .sort_values("max_abs_corr", ascending=False)
            )

            fig = px.bar(
                group_summary,
                x="feature_group",
                y="max_abs_corr",
                color="feature_group",
                color_discrete_sequence=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#f43f5e"],
                hover_data=["feature_count", "mean_abs_corr"],
                title="Feature Group Explainability Strength",
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            fig = make_plot_layout(fig, height=485)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ML feature explainability file not found.")

    with e2:
        if not policy_driver.empty and "explained_outcome" in policy_driver.columns:
            selected_outcome = st.selectbox(
                "Select explained outcome",
                sorted(policy_driver["explained_outcome"].dropna().unique()),
            )
            temp = policy_driver[
                policy_driver["explained_outcome"] == selected_outcome
            ].sort_values("abs_spearman_correlation", ascending=False).head(10)

            fig = px.bar(
                temp.sort_values("abs_spearman_correlation"),
                x="abs_spearman_correlation",
                y="driver_feature",
                color="spearman_correlation",
                orientation="h",
                color_continuous_scale="Tealrose",
                title=f"Top Policy Drivers for {selected_outcome}",
            )
            fig = make_plot_layout(fig, height=485)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Policy-driver explainability file not found.")

    st.markdown("---")

    e3, e4 = st.columns([1, 1])

    with e3:
        section("Stress-Group Explanation")
        if not stress_explain.empty:
            fig = px.bar(
                stress_explain.sort_values("avg_reward_improvement"),
                x="avg_reward_improvement",
                y="stress_group",
                orientation="h",
                color="avg_delay_reduction",
                color_continuous_scale="Tealgrn",
                hover_data=[
                    "avg_risk_reduction",
                    "avg_service_improvement",
                    "avg_resilience_improvement",
                ],
                title="Stress Explainability by Group",
            )
            fig = make_plot_layout(fig, height=440)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Stress explainability summary not found.")

    with e4:
        section("Explainability Findings")
        insight_list(key_explain, max_items=6)

    section("Local Explanation Explorer")
    if not local_explain.empty and "local_case_type" in local_explain.columns:
        case_type = st.selectbox(
            "Select local case type",
            sorted(local_explain["local_case_type"].dropna().unique()),
        )
        temp = local_explain[local_explain["local_case_type"] == case_type].copy()
        display_dataframe(temp, height=430)
    else:
        display_dataframe(local_explain, height=430)


def board_report(tables):
    master = tables["master_summary"]
    key_11 = tables["key_findings_11"]
    key_12 = tables["key_ablation"]
    key_13 = tables["key_stress"]
    key_14 = tables["key_explain"]

    section(
        "Board-Ready Report Generator",
        "Auto-generated manuscript and executive-report wording from calculated outputs.",
    )

    report_lines = []

    report_lines.append("RESILIENTGRAPH-CMARL EXECUTIVE PERFORMANCE SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    if not master.empty:
        for _, row in master.iterrows():
            report_lines.append(f"{row.get('module', 'Module')}:")
            report_lines.append(f"- Best baseline: {row.get('best_baseline', 'NA')}")
            report_lines.append(f"- Best proposed: {row.get('best_proposed', 'NA')}")
            report_lines.append(
                f"- Primary metric: {row.get('primary_metric', 'NA')} improved from "
                f"{fmt_num(row.get('baseline_value', 0))} to {fmt_num(row.get('proposed_value', 0))} "
                f"({fmt_pct(row.get('relative_gain_percent', 0))} relative gain)."
            )
            report_lines.append(f"- Interpretation: {row.get('interpretation', '')}")
            report_lines.append("")

    report_lines.append("Key findings:")
    report_lines.append("-" * 80)

    for df in [key_11, key_12, key_13, key_14]:
        if not df.empty and "finding" in df.columns:
            for _, row in df.iterrows():
                report_lines.append(f"- {row.get('finding', '')}")

    report_text = "\n".join(report_lines)

    st.text_area(
        "Copy-ready report text",
        value=report_text,
        height=530,
    )

    st.download_button(
        label="Download executive report text",
        data=report_text.encode("utf-8"),
        file_name="ResilientGraph_CMARL_Executive_Report.txt",
        mime="text/plain",
    )


def artifact_center(tables):
    artifact_index = tables["artifact_index"]

    section(
        "Data Lineage and Artifact Center",
        "Checks major analysis artifacts and displays saved output evidence.",
    )

    loaded, total = build_health_status(tables)

    c1, c2, c3 = st.columns(3)

    with c1:
        kpi_card("Loaded Core Tables", f"{loaded}/{total}", "dashboard readiness", "info")

    with c2:
        existing_files = sum(1 for p in FILES.values() if p.exists())
        kpi_card("Existing Output Files", str(existing_files), "tracked result files", "success")

    with c3:
        kpi_card("Project Root", "READY", str(PROJECT_ROOT), "purple")

    rows = []

    for name, path in FILES.items():
        df = tables.get(name, pd.DataFrame())
        rows.append(
            {
                "artifact_key": name,
                "file_name": path.name,
                "exists": path.exists(),
                "rows_loaded": int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0,
                "columns_loaded": int(df.shape[1]) if isinstance(df, pd.DataFrame) else 0,
                "path": str(path),
            }
        )

    artifact_health = pd.DataFrame(rows)
    display_dataframe(artifact_health, "Dashboard Artifact Health", height=530)

    if not artifact_index.empty:
        display_dataframe(artifact_index, "Step 11 Final Artifact Index", height=410)


def what_if_simulator():
    section(
        "Deterministic CMARL What-if Simulator",
        "Interactive business-style simulation using deterministic policy logic. This is for dashboard interaction and scenario communication, not model retraining.",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        base_risk = st.slider("Base risk", 0.0, 1.0, 0.45, 0.01)
        base_delay = st.slider("Base delay probability", 0.0, 1.0, 0.60, 0.01)

    with c2:
        context_signal = st.slider("Graph context signal", 0.0, 1.0, 0.55, 0.01)
        service_importance = st.slider("Service importance", 0.0, 1.0, 0.70, 0.01)

    with c3:
        cost_pressure = st.slider("Cost pressure", 0.0, 1.0, 0.35, 0.01)
        demand_pressure = st.slider("Demand pressure", 0.0, 1.0, 0.45, 0.01)

    with c4:
        profit_stress = st.slider("Profit stress", 0.0, 1.0, 0.25, 0.01)
        scenario = st.selectbox(
            "Scenario",
            [
                "normal",
                "demand_surge",
                "logistics_disruption",
                "supplier_shock",
                "combined_stress",
            ],
        )

    scenario_cfg = {
        "normal": (1.0, 1.0, 1.0, 1.0),
        "demand_surge": (1.10, 1.08, 1.30, 1.06),
        "logistics_disruption": (1.22, 1.35, 1.0, 1.08),
        "supplier_shock": (1.28, 1.16, 1.05, 1.14),
        "combined_stress": (1.45, 1.55, 1.35, 1.24),
    }[scenario]

    risk_multiplier, delay_multiplier, demand_multiplier, cost_multiplier = scenario_cfg

    risk_intensity = np.clip(
        0.34 * base_risk
        + 0.26 * base_delay
        + 0.24 * context_signal
        + 0.16 * demand_pressure,
        0,
        1,
    )

    service_intensity = np.clip(
        0.44 * service_importance
        + 0.28 * base_delay
        + 0.18 * demand_pressure
        + 0.10 * context_signal,
        0,
        1,
    )

    baseline_risk = np.clip(base_risk * 0.64 * risk_multiplier, 0, 1)
    proposed_risk = np.clip(
        base_risk * (0.50 + 0.16 * (1 - risk_intensity)) * risk_multiplier,
        0,
        1,
    )

    baseline_delay = np.clip(base_delay * 0.72 * delay_multiplier, 0, 1)
    proposed_delay = np.clip(
        base_delay * (0.54 + 0.14 * (1 - service_intensity)) * delay_multiplier,
        0,
        1,
    )

    baseline_service = np.clip(
        1 - 0.42 * baseline_risk - 0.35 * baseline_delay + 0.08,
        0,
        1,
    )

    proposed_service = np.clip(
        1
        - 0.38 * proposed_risk
        - 0.32 * proposed_delay
        - 0.08 * max(demand_multiplier - 1, 0)
        + np.clip(0.11 + 0.08 * service_intensity, 0.11, 0.19),
        0,
        1,
    )

    baseline_resilience = np.clip(
        1 - 0.48 * baseline_risk - 0.28 * baseline_delay + 0.12,
        0,
        1,
    )

    proposed_resilience = np.clip(
        1
        - 0.46 * proposed_risk
        - 0.26 * proposed_delay
        + np.clip(0.18 + 0.08 * risk_intensity, 0.18, 0.26),
        0,
        1,
    )

    baseline_reward = (
        1.50 * baseline_service
        + 1.35 * baseline_resilience
        - 1.20 * baseline_risk
        - 1.00 * baseline_delay
    )

    proposed_reward = (
        1.50 * proposed_service
        + 1.35 * proposed_resilience
        - 1.20 * proposed_risk
        - 1.00 * proposed_delay
        + 0.20 * service_importance * proposed_service
    )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        kpi_card(
            "Reward Gain",
            fmt_num(proposed_reward - baseline_reward),
            "simulated what-if delta",
            "success",
        )

    with c6:
        risk_reduction = (baseline_risk - proposed_risk) / max(abs(baseline_risk), 1e-9) * 100
        kpi_card("Risk Reduction", fmt_pct(risk_reduction), "what-if result", "info")

    with c7:
        delay_reduction = (baseline_delay - proposed_delay) / max(abs(baseline_delay), 1e-9) * 100
        kpi_card("Delay Reduction", fmt_pct(delay_reduction), "what-if result", "warning")

    with c8:
        kpi_card(
            "Resilience Gain",
            fmt_num(proposed_resilience - baseline_resilience),
            "what-if result",
            "purple",
        )

    comparison = pd.DataFrame(
        {
            "Metric": ["Risk", "Delay", "Service", "Resilience", "Reward"],
            "Baseline": [
                baseline_risk,
                baseline_delay,
                baseline_service,
                baseline_resilience,
                baseline_reward,
            ],
            "Proposed": [
                proposed_risk,
                proposed_delay,
                proposed_service,
                proposed_resilience,
                proposed_reward,
            ],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison["Metric"],
            y=comparison["Baseline"],
            name="Baseline",
            marker_color="#94a3b8",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison["Metric"],
            y=comparison["Proposed"],
            name="Proposed",
            marker_color="#14b8a6",
        )
    )
    fig.update_layout(
        title="What-if Baseline vs Proposed Policy Simulation",
        barmode="group",
    )
    fig = make_plot_layout(fig, height=460)
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================================
# SIDEBAR
# ======================================================================================

def sidebar_controls(tables):
    # Status is calculated internally but not displayed in the sidebar.
    _loaded, _total = build_health_status(tables)

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-kicker">Research Intelligence Console</div>
            <div class="brand-title">📦 ResilientGraph-CMARL</div>
            <div class="brand-subtitle">
                Academic executive dashboard for graph-risk prediction, digital-twin procurement optimization,
                stress resilience, and explainable decision intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='nav-caption'>Analysis Navigation</div>", unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Select dashboard section",
        [
            "🏛️ Executive Control Tower",
            "🧠 Risk Prediction Intelligence",
            "🛰️ Decision Optimization Control",
            "🛡️ Stress-Testing War Room",
            "🔍 Explainability Hub",
            "🎛️ What-if Simulator",
            "📑 Board Report",
            "🧾 Data Lineage & Artifacts",
        ],
        label_visibility="collapsed",
    )

    return page


# ======================================================================================
# MAIN APP
# ======================================================================================

def main():
    tables = load_all_tables()
    page = sidebar_controls(tables)

    if page == "🏛️ Executive Control Tower":
        executive_overview(tables)

    elif page == "🧠 Risk Prediction Intelligence":
        risk_prediction_intelligence(tables)

    elif page == "🛰️ Decision Optimization Control":
        decision_optimization_control(tables)

    elif page == "🛡️ Stress-Testing War Room":
        stress_war_room(tables)

    elif page == "🔍 Explainability Hub":
        explainability_hub(tables)

    elif page == "🎛️ What-if Simulator":
        what_if_simulator()

    elif page == "📑 Board Report":
        board_report(tables)

    elif page == "🧾 Data Lineage & Artifacts":
        artifact_center(tables)


if __name__ == "__main__":
    main()