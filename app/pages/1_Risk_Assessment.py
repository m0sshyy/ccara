"""CCaRA — Risk Assessment page.

Collects the 11 non-invasive inputs, runs the trained XGBoost model,
shows the predicted risk level with a Tree SHAP explanation, applies the
Within-High-Risk Urgency sub-tiering for High-risk results, offers a
downloadable branded PDF report, and saves the assessment to local
history.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from utils import branding, model_utils as mu, db_utils as du, pdf_report as pr, charts

st.set_page_config(
    page_title="Risk Assessment — CCaRA",
    page_icon=branding.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

branding.inject_css()
branding.sidebar_logo()
branding.top_banner("Risk Assessment")


@st.cache_resource(show_spinner="Loading model artefacts...")
def _load():
    return mu.load_artifacts()


model, label_encoder, feature_names, base_features = _load()

st.markdown(
    """
    <p style="color:#5b6478;max-width:50rem;">
    Answer the questions below as accurately as you can. This tool uses only
    demographic, lifestyle, and medical-history information — <b>no lab
    tests, imaging, or screening history</b> are required. It takes about a
    minute.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.form("assessment_form"):
    label = st.text_input(
        "Name or ID for this assessment (optional — use a nickname, not required)",
        placeholder="e.g. Self-check 20 Jul, or leave blank",
    )

    st.markdown("#### Demographics")
    d1, d2 = st.columns(2)
    with d1:
        age = st.number_input(
            mu.FEATURE_META["Age"]["label"], min_value=mu.FEATURE_META["Age"]["min"],
            max_value=mu.FEATURE_META["Age"]["max"], value=mu.FEATURE_META["Age"]["default"],
            help=mu.FEATURE_META["Age"]["help"],
        )
    with d2:
        gender = st.selectbox(mu.FEATURE_META["Gender"]["label"], mu.FEATURE_META["Gender"]["options"])

    st.markdown("#### Lifestyle")
    l1, l2, l3 = st.columns(3)
    with l1:
        smoking = st.selectbox(mu.FEATURE_META["Smoking_History"]["label"], mu.FEATURE_META["Smoking_History"]["options"])
        diet_risk = st.selectbox(mu.FEATURE_META["Diet_Risk"]["label"], mu.FEATURE_META["Diet_Risk"]["options"],
                                  index=1, help=mu.FEATURE_META["Diet_Risk"]["help"])
    with l2:
        alcohol = st.selectbox(mu.FEATURE_META["Alcohol_Consumption"]["label"], mu.FEATURE_META["Alcohol_Consumption"]["options"])
        activity = st.selectbox(mu.FEATURE_META["Physical_Activity"]["label"], mu.FEATURE_META["Physical_Activity"]["options"], index=1)
    with l3:
        bmi = st.selectbox(mu.FEATURE_META["Obesity_BMI"]["label"], mu.FEATURE_META["Obesity_BMI"]["options"])

    st.markdown("#### Medical history")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        family_history = st.selectbox(mu.FEATURE_META["Family_History"]["label"], mu.FEATURE_META["Family_History"]["options"],
                                       help=mu.FEATURE_META["Family_History"]["help"])
    with m2:
        diabetes = st.selectbox(mu.FEATURE_META["Diabetes"]["label"], mu.FEATURE_META["Diabetes"]["options"])
    with m3:
        ibd = st.selectbox(mu.FEATURE_META["Inflammatory_Bowel_Disease"]["label"], mu.FEATURE_META["Inflammatory_Bowel_Disease"]["options"])
    with m4:
        genetic = st.selectbox(mu.FEATURE_META["Genetic_Mutation"]["label"], mu.FEATURE_META["Genetic_Mutation"]["options"],
                                help=mu.FEATURE_META["Genetic_Mutation"]["help"])

    submitted = st.form_submit_button("Predict My Risk")

if submitted:
    form_values = {
        "Age": int(age), "Gender": gender, "Family_History": family_history,
        "Smoking_History": smoking, "Alcohol_Consumption": alcohol, "Obesity_BMI": bmi,
        "Diet_Risk": diet_risk, "Physical_Activity": activity, "Diabetes": diabetes,
        "Inflammatory_Bowel_Disease": ibd, "Genetic_Mutation": genetic,
    }
    raw = mu.encode_form_inputs(form_values)
    result = mu.predict(model, label_encoder, feature_names, raw)
    top_factors = mu.top_contributors(result.contributions, result.engineered, n=6,
                                       predicted_class=result.predicted_class)
    recommendation = mu.get_recommendation(result.predicted_class, result.urgency_tier)

    try:
        du.save_assessment(label=label, form_values=form_values, prediction=result, top_factors=top_factors)
        saved_ok = True
    except Exception:
        saved_ok = False

    st.markdown('<div class="ccara-divider"></div>', unsafe_allow_html=True)

    # --- Result banner ---------------------------------------------------
    badge = branding.risk_badge_html(result.predicted_class)
    r_bg = branding.RISK_BG[result.predicted_class]
    r_color = branding.RISK_COLORS[result.predicted_class]
    st.markdown(
        f"""
        <div style="background:{r_bg};border-left:6px solid {r_color};padding:1.2rem 1.4rem;margin-bottom:1rem;">
            {badge}
            <div style="font-size:1.6rem;font-family:'Archivo Black';color:{r_color};margin-top:0.5rem;">
                {result.confidence*100:.1f}% model confidence
            </div>
            <div style="color:#0d1b33;margin-top:0.4rem;">{mu.RISK_BLURB[result.predicted_class]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.urgency_tier:
        u_color = branding.URGENCY_COLORS[result.urgency_tier]
        st.markdown(
            f"""
            <div class="ccara-card" style="border-left:6px solid {u_color};">
                <b>Within-High-Risk urgency sub-tier: <span style="color:{u_color};">{result.urgency_tier}</span></b>
                <span style="color:#5b6478;"> (urgency score {result.urgency_score:.0f})</span><br>
                <span style="color:#5b6478;font-size:0.9rem;">
                Based on the study's Within-High-Risk Urgency Profiling (Section 4.4.3): High-risk
                cases are further split into High-Low, High-Med, and High-Critical using a
                combined lifestyle + non-modifiable risk burden score, enabling a more targeted
                recommendation than a single "High risk" label.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.markdown("##### Recommendation")
        st.markdown(
            f'<div class="ccara-card" style="font-size:1rem;">{recommendation}</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown("##### Class probabilities")
        st.pyplot(charts.probability_bar_chart(result.probabilities), use_container_width=True)

    st.markdown("##### What's driving this prediction (SHAP waterfall)")
    st.caption(
        "Starting from the model's baseline expectation E[f(x)], each factor bridges the "
        "score to f(x) — this patient's final score. Red = increases CRC risk, "
        "green = protective, regardless of which class was predicted. Computed with exact "
        "Tree SHAP values from the trained XGBoost model."
    )
    display_values = {feat: mu.display_value(feat, val) for feat, val in result.engineered.items()}
    st.pyplot(
        charts.shap_waterfall_chart(
            result.contributions, result.base_value, result.predicted_class,
            result.engineered, mu.FRIENDLY_NAMES, top_n=6, display_values=display_values,
        ),
        use_container_width=True,
    )

    # --- Download report ---------------------------------------------------
    pdf_bytes = pr.build_report(
        label=label, form_values=form_values, prediction=result,
        top_factors=top_factors, recommendation=recommendation,
    )
    dcol1, dcol2 = st.columns([1, 2])
    with dcol1:
        st.download_button(
            "Download Risk Report (PDF)", data=pdf_bytes,
            file_name="CCaRA_risk_report.pdf", mime="application/pdf",
        )
    with dcol2:
        if saved_ok:
            st.caption("Saved to your local assessment history.")
        else:
            st.caption("Result generated — could not save to history this time.")

    st.info(
        "CCaRA is an awareness prototype, not a diagnostic device. It has not been "
        "validated on real clinical data and should not be used to make medical "
        "decisions on its own. Please discuss these results with a qualified "
        "healthcare provider.",
        icon="⚠️",
    )

branding.footer()
