"""CCaRA — About page: methodology, model card, disclaimer, brand meaning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from utils import branding

st.set_page_config(
    page_title="About — CCaRA",
    page_icon=branding.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

branding.inject_css()
branding.sidebar_logo()
branding.top_banner("About CCaRA")

tab1, tab2, tab3, tab4 = st.tabs(["The model", "Risk factors used", "Brand & name", "Disclaimer"])

with tab1:
    st.markdown("### Methodology summary")
    st.markdown(
        """
        <div class="ccara-card">
        CCaRA is built on an <b>XGBoost</b> classifier trained on the Colorectal Cancer
        Global Dataset and Predictions (Kaggle, 167,497 records / 28 variables). None
        of the dataset's original outcome fields were statistically associated with
        known CRC risk factors (chi-square p &gt; 0.05 for all), so a new target —
        <b>CRC_Risk_Level</b> (Low / Moderate / High) — was engineered from 11 core
        risk factors using a log-odds-inspired weighting scheme, then split into
        tertiles.
        <br><br>
        Logistic Regression, Random Forest, and XGBoost were compared; tuned XGBoost
        (via 5-fold GridSearchCV, 1,215 model fits) was selected as the primary model.
        The three-tier structure was independently validated with K-Means and
        Agglomerative clustering, both of which converged on <b>k = 3</b> as the
        natural number of clusters.
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in [
        (m1, "Accuracy", "85.49%"), (m2, "F1-Macro", "0.8437"),
        (m3, "ROC-AUC (OvR)", "0.9050"), (m4, "10-fold CV F1-Macro", "0.8436 ± 0.0052"),
    ]:
        with col:
            st.markdown(
                f"""<div class="ccara-card" style="text-align:center;">
                <div style="font-size:1.4rem;font-family:'Archivo Black';color:#072754;">{value}</div>
                <div style="color:#5b6478;font-size:0.82rem;">{label}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### Explainability")
    st.markdown(
        """
        <div class="ccara-card">
        Every prediction is paired with an exact <b>Tree SHAP</b> explanation
        (Lundberg &amp; Lee, 2017) computed directly from the trained XGBoost booster,
        showing which risk factors pushed the prediction toward — or away from —
        the predicted class. This mirrors the SHAP-based interpretability approach
        used throughout the underlying research.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Within-High-Risk urgency profiling")
    st.markdown(
        """
        <div class="ccara-card">
        High-risk results are further split into three urgency sub-tiers —
        <b>High-Low</b>, <b>High-Med</b>, and <b>High-Critical</b> — based on a
        combined lifestyle and non-modifiable risk burden score. This extends the
        standard 3-class framework so High-risk users get a proportionate
        recommendation (routine screening → lifestyle counselling → immediate
        referral) instead of one blanket message.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("### The 11 inputs")
    st.markdown(
        """
        <div class="ccara-card">
        All inputs are self-reportable — no lab tests, imaging, or prior screening
        history required.
        </div>
        """,
        unsafe_allow_html=True,
    )
    rows = [
        ("Age", "Demographic", "CRC risk rises markedly after age 50, and further after 70."),
        ("Gender", "Demographic", "Included as a modest, well-documented epidemiological factor."),
        ("Family history", "Non-modifiable", "A first-degree relative diagnosed with CRC — one of the strongest predictors in the model."),
        ("Genetic mutation", "Non-modifiable", "e.g. Lynch syndrome, FAP — a strong hereditary risk signal."),
        ("Inflammatory bowel disease", "Non-modifiable", "Crohn's disease or ulcerative colitis."),
        ("Smoking history", "Modifiable / lifestyle", "One of the most consistently reported modifiable risk factors."),
        ("Alcohol consumption", "Modifiable / lifestyle", "Regular consumption is associated with elevated risk."),
        ("BMI category", "Modifiable / lifestyle", "Obesity is linked to CRC risk and metabolic risk."),
        ("Diet risk", "Modifiable / lifestyle", "High-risk diets are low in fibre and high in red/processed meat."),
        ("Physical activity", "Modifiable / lifestyle", "Higher activity is protective against CRC."),
        ("Diabetes", "Metabolic", "Associated with elevated CRC risk via metabolic pathways."),
    ]
    st.dataframe(
        {"Factor": [r[0] for r in rows], "Category": [r[1] for r in rows], "Why it matters": [r[2] for r in rows]},
        use_container_width=True, hide_index=True,
    )
    st.markdown(
        """
        <div class="ccara-card soft" style="margin-top:0.5rem;">
        Internally, the model also computes 5 <b>engineered</b> features from these
        11 inputs: <i>Lifestyle Burden</i> (sum of 5 modifiable risk flags),
        <i>Non-Modifiable Load</i> (family history + genetic mutation + IBD),
        <i>Metabolic Risk</i> (diabetes + obesity), <i>Age Risk Tier</i>, and
        <i>Compound Risk</i> (lifestyle + non-modifiable combined) — the two
        strongest predictors in the whole model.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab3:
    lcol, rcol = st.columns([1, 1.6], gap="large")
    with lcol:
        st.image(str(branding.ASSETS_DIR / "ccara_lockup.png"))
    with rcol:
        st.markdown(
            """
            <div class="ccara-card">
            <b>CCaRA</b> stands for <b>C</b>olorectal <b>C</b>ancer <b>R</b>isk
            <b>A</b>wareness. In Kelantanese, <i>"ambil kisah"</i> means to take
            heed, to care — the quiet second meaning behind the name.
            <br><br>
            The mark is two "C"s (for Colorectal Cancer) drawn as mirrored arcs,
            interlocked at their open ends rather than sitting side by side —
            symbolising connection and continuity of care, and echoing the idea of
            early detection catching something before it closes into a full circle.
            <br><br>
            The two accent dots sit exactly where the C's meet: green at the top
            link, pink at the bottom. They represent the people the awareness
            reaches beyond a single clinical audience — small, deliberate points of
            colour against a serious navy field, warmth and humanity without
            diluting the clinical trust the navy carries.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("##### Who CCaRA is for")
    st.markdown(
        """
        <div class="ccara-card">
        <b>End users</b> — adults who want to check their colorectal cancer risk early.<br>
        <b>Paying customers</b> — private clinics, health screening centres, university
        health facilities, healthcare organisations.<br>
        <b>Supporting organisations</b> — NGOs and public health agencies running
        cancer awareness campaigns.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab4:
    st.markdown(
        """
        <div class="ccara-card" style="border-left:6px solid #b3263a;">
        <b>CCaRA is an awareness prototype, not a diagnostic device.</b><br><br>
        It is intended to raise early awareness and support conversations with a
        healthcare provider — not to diagnose colorectal cancer or replace
        professional medical examination, colonoscopy, or laboratory testing.
        <br><br>
        The underlying model was trained on a public dataset with an engineered
        risk target and has not been validated against real-world clinical
        diagnoses. Predictions should not be used to make medical decisions
        without consulting a qualified healthcare provider. Further research with
        real clinical data and external validation would be required before any
        clinical use.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Final Year Project — Alya Fariha Binti Azairry · Supervised by Prof. Dr. Marina binti Yusoff")

branding.footer()
