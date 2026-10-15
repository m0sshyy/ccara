"""CCaRA — Colorectal Cancer Risk Awareness.

Home / landing page. Run with:  streamlit run app.py
"""

import streamlit as st

from utils import branding

st.set_page_config(
    page_title="CCaRA — Colorectal Cancer Risk Awareness",
    page_icon=branding.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

branding.inject_css()
branding.sidebar_logo()
branding.top_banner()

# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
left, right = st.columns([1.3, 1], gap="large")
with left:
    st.markdown(
        """
        <div style="font-size:2.1rem;font-family:'Archivo Black';color:#072754;line-height:1.15;">
            Know your colorectal cancer risk<br>before it becomes a diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style="color:#5b6478;font-size:1.02rem;margin-top:0.8rem;max-width:44rem;">
        CCaRA is a low-cost, non-invasive awareness tool built on an explainable
        XGBoost model. Answer 11 questions about demographics, lifestyle, and medical
        history — no lab tests, imaging, or clinic visit required — and get a risk
        level, a plain-language explanation of <em>why</em>, and a screening
        recommendation you can bring to a healthcare provider.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Risk_Assessment.py", label="Start My Risk Assessment", icon="🩺")
    st.write("")
    st.page_link("pages/2_History.py", label="View Assessment History", icon="📜")
    st.page_link("pages/3_About.py", label="About the Model & Methodology", icon="ℹ️")

with right:
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:1.2rem;">
            <img src="{branding.logo_data_uri('ccara_icon.png')}" style="height:170px;" />
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="ccara-card navy">
            <div style="font-family:'Archivo Black';font-size:1.05rem;margin-bottom:0.6rem;">
                Model snapshot
            </div>
            <div style="font-size:0.92rem;line-height:2.0;">
                <b>85.49%</b> test accuracy (XGBoost, tuned)<br>
                <b>0.8437</b> macro F1-score<br>
                <b>0.9050</b> ROC-AUC (one-vs-rest)<br>
                <b>167,497</b> patient records used to train and validate<br>
                <b>11</b> non-invasive risk factors, no lab tests needed<br>
                <b>3-tier</b> risk framework (Low / Moderate / High), independently
                confirmed with unsupervised clustering
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="ccara-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Why this matters
# ---------------------------------------------------------------------
st.markdown("### Why early risk awareness matters")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="ccara-card">
        <div class="ccara-dot" style="background:#b3263a;"></div>
        <b>CRC is often caught late.</b><br><br>
        Global incidence more than doubled between 1990 and 2019, and many
        cases are still detected after symptoms appear rather than through
        proactive screening.
        </div>
        """, unsafe_allow_html=True)
with c2:
    st.markdown(
        """
        <div class="ccara-card">
        <div class="ccara-dot" style="background:#c98a2e;"></div>
        <b>Screening has real barriers.</b><br><br>
        Colonoscopy and lab-based tests are effective but can be costly,
        invasive, and hard to access — many people simply don't know their
        risk level well enough to prioritise them.
        </div>
        """, unsafe_allow_html=True)
with c3:
    st.markdown(
        """
        <div class="ccara-card">
        <div class="ccara-dot" style="background:#4c9660;"></div>
        <b>Awareness is the first step.</b><br><br>
        A short, explainable risk check can help people understand their
        personal risk factors and decide, with a clinician, whether earlier
        screening makes sense.
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Who it's for
# ---------------------------------------------------------------------
with st.container():
    st.markdown(
        """
        <div class="ccara-card soft">
        <b>Who CCaRA is built for</b><br><br>
        <b>End users</b> — adults who want to check their colorectal cancer risk early.<br>
        <b>Clinics & screening centres</b> — a low-cost intake tool that flags who may
        need to prioritise screening.<br>
        <b>Public health & awareness campaigns</b> — a shareable, explainable entry
        point into CRC risk conversations.
        </div>
        """,
        unsafe_allow_html=True,
    )

branding.footer()
