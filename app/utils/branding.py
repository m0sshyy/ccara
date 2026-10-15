"""CCaRA brand tokens + shared CSS injection.

Colours are taken directly from CCARA BRANDING.pdf:
  base  (navy)  oklch(0.28 0.09 258)  -> #072754
  accent (pink) oklch(0.82 0.07 350)  -> #e8b2ca
  accent (green) oklch(0.72 0.1 150)  -> #75b683
Typeface: Archivo / Archivo Black. Brand spec calls for "no corner radius".
"""

from pathlib import Path
import base64
import textwrap
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

# ---------------------------------------------------------------------
# Streamlit renders indented multi-line strings passed to st.markdown()
# as an escaped code block instead of raw HTML (a well-known Streamlit
# quirk - see github.com/streamlit/streamlit issues #859 / #586).
# Since every HTML snippet in this app is written as an indented
# triple-quoted string (for readable Python source), markdown() is
# patched once, here, to dedent its input before rendering. This makes
# every st.markdown(..., unsafe_allow_html=True) call in the app render
# correctly regardless of the surrounding Python indentation.
# ---------------------------------------------------------------------
_original_markdown = DeltaGenerator.markdown


def _dedented_markdown(self, body="", *args, **kwargs):
    if isinstance(body, str):
        body = textwrap.dedent(body)
    return _original_markdown(self, body, *args, **kwargs)


DeltaGenerator.markdown = _dedented_markdown

# --- Brand palette -----------------------------------------------------
NAVY = "#072754"
NAVY_LIGHT = "#0f3a73"
NAVY_SOFT = "#eef1f7"      # tinted background derived from navy
PINK = "#e8b2ca"
PINK_STRONG = "#d9749f"
GREEN = "#75b683"
GREEN_STRONG = "#4c9660"
BG = "#f4f6fb"
TEXT = "#0d1b33"
MUTED = "#5b6478"

# Risk-severity colours (kept distinct from decorative brand accents so
# risk level is always legible at a glance)
RISK_COLORS = {
    "Low": GREEN_STRONG,
    "Moderate": "#c98a2e",
    "High": "#b3263a",
}
RISK_BG = {
    "Low": "#e9f5ec",
    "Moderate": "#fbf1e2",
    "High": "#fbe9ea",
}
URGENCY_COLORS = {
    "High-Low": "#d97a4f",
    "High-Med": "#c1443a",
    "High-Critical": "#7a1620",
}

FONT_IMPORT = "https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Archivo+Black&display=swap"

PAGE_ICON = str(ASSETS_DIR / "ccara_icon_navy.png")


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def logo_data_uri(name: str) -> str:
    return f"data:image/png;base64,{_b64(ASSETS_DIR / name)}"


def inject_css():
    # NOTE: this is deliberately written with (a) the font <link> tags in
    # their own st.markdown call and (b) zero blank lines inside the
    # <style> block. Streamlit's markdown renderer can break out of an
    # unsafe-HTML block into escaped/paragraph text at a blank line
    # (streamlit/streamlit issues #586, #859) — mixing tag types and
    # leaving blank lines for readability is exactly what triggers it.
    st.markdown(
        f"""<link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="{FONT_IMPORT}" rel="stylesheet">""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<style>
        :root {{
            --navy: {NAVY};
            --pink: {PINK};
            --green: {GREEN};
            --bg: {BG};
            --text: {TEXT};
        }}
        html, body, [class*="css"] {{
            font-family: 'Archivo', sans-serif;
        }}
        h1, h2, h3, .ccara-wordmark {{
            font-family: 'Archivo Black', 'Archivo', sans-serif;
            letter-spacing: -0.01em;
        }}
        /* App background */
        .stApp {{
            background-color: {BG};
        }}
        /* Sharp corners everywhere, per brand spec */
        button, .stButton>button, .stDownloadButton>button, .stTextInput input,
        .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea,
        div[data-testid="stMetric"], .ccara-card, .ccara-banner, .ccara-badge, .ccara-pill {{
            border-radius: 0 !important;
        }}
        .stButton>button, .stDownloadButton>button {{
            background-color: {NAVY};
            color: white;
            border: none;
            font-weight: 600;
            padding: 0.55em 1.4em;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.82rem;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            background-color: {NAVY_LIGHT};
            color: white;
        }}
        /* Top banner */
        .ccara-banner {{
            background-color: {NAVY};
            padding: 1.4rem 1.8rem;
            display: flex;
            align-items: center;
            gap: 1.1rem;
            margin-bottom: 1.8rem;
        }}
        .ccara-banner img {{ height: 64px; }}
        .ccara-banner .ccara-wordmark {{
            color: white;
            font-size: 2.1rem;
            line-height: 1;
        }}
        .ccara-banner .ccara-tagline {{
            color: #c7d0e6;
            font-size: 0.88rem;
            letter-spacing: 0.03em;
            margin-top: 4px;
        }}
        /* Cards */
        .ccara-card {{
            background: white;
            border: 1px solid #dfe4ee;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }}
        .ccara-card.navy {{ background: {NAVY}; color: white; border: none; }}
        .ccara-card.soft {{ background: {NAVY_SOFT}; border: none; }}
        /* Badge / pill for risk level */
        .ccara-badge {{
            display: inline-block;
            padding: 0.35em 0.9em;
            font-weight: 700;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .ccara-dot {{
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .ccara-divider {{
            height: 3px;
            background: linear-gradient(90deg, {GREEN} 0%, {NAVY} 45%, {PINK} 100%);
            margin: 0.4rem 0 1.6rem 0;
        }}
        section[data-testid="stSidebar"] {{
            background-color: white;
            border-right: 1px solid #e3e7f0;
        }}
        .ccara-footer {{
            color: {MUTED};
            font-size: 0.78rem;
            text-align: center;
            padding: 1.5rem 0 0.5rem 0;
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def top_banner(tagline: str = "Colorectal Cancer Risk Awareness"):
    st.markdown(
        f"""
        <div class="ccara-banner">
            <img src="{logo_data_uri('ccara_icon_reversed.png')}" />
            <div>
                <div class="ccara-wordmark">CCaRA</div>
                <div class="ccara-tagline">{tagline}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_logo():
    st.sidebar.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.7rem;padding:0.5rem 0 1.2rem 0;">
            <img src="{logo_data_uri('ccara_icon_navy.png')}" style="height:48px;" />
            <div style="font-family:'Archivo Black';color:{NAVY};font-size:1.5rem;">CCaRA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(risk_class: str) -> str:
    color = RISK_COLORS.get(risk_class, MUTED)
    return (
        f'<span class="ccara-badge" style="background:{color};color:white;">{risk_class} Risk</span>'
    )


def footer():
    st.markdown(
        """
        <div class="ccara-footer">
            CCaRA is an awareness prototype, not a diagnostic device. It does not replace
            professional medical advice, screening, or diagnosis.<br>
            Built for a Final Year Project · XGBoost + SHAP colorectal cancer risk model
        </div>
        """,
        unsafe_allow_html=True,
    )
