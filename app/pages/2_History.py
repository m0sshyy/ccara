"""CCaRA — Assessment History.

Local SQLite log of completed assessments. Demonstrates the Lean Model
Canvas key metrics (number of completed assessments, number of high-risk
users referred for screening) and lets a user or clinic reviewer look back
at past results. Everything is stored locally — nothing leaves the
machine running the app.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from utils import branding, db_utils as du

st.set_page_config(
    page_title="History — CCaRA",
    page_icon=branding.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

branding.inject_css()
branding.sidebar_logo()
branding.top_banner("Assessment History")

stats = du.summary_stats()

k1, k2, k3, k4 = st.columns(4)
for col, label, value in [
    (k1, "Completed assessments", stats["total"]),
    (k2, "Avg. model confidence", f"{stats['avg_confidence']*100:.1f}%" if stats["total"] else "—"),
    (k3, "High-risk results", stats["by_class"].get("High", 0)),
    (k4, "High-Critical referrals", stats["high_urgent"]),
]:
    with col:
        st.markdown(
            f"""
            <div class="ccara-card" style="text-align:center;">
                <div style="font-size:1.7rem;font-family:'Archivo Black';color:#072754;">{value}</div>
                <div style="color:#5b6478;font-size:0.85rem;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="ccara-divider"></div>', unsafe_allow_html=True)

df = du.fetch_history()

if df.empty:
    st.markdown(
        """
        <div class="ccara-card soft">
        No assessments yet. Head to <b>Risk Assessment</b> in the sidebar to run your first check —
        it will show up here automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    with st.expander("Filter results", expanded=False):
        classes = st.multiselect("Risk class", ["Low", "Moderate", "High"], default=["Low", "Moderate", "High"])
        search = st.text_input("Search by name / ID", "")

    view = df[df["predicted_class"].isin(classes)] if classes else df
    if search:
        view = view[view["label"].fillna("").str.contains(search, case=False)]

    st.markdown("##### Risk class distribution over time")
    dist = df.groupby(["predicted_class"]).size().reindex(["Low", "Moderate", "High"]).fillna(0)
    st.bar_chart(dist, color="#072754")

    st.markdown("##### Assessment log")
    display_cols = {
        "timestamp": "Date/time", "label": "Name/ID", "age": "Age", "gender": "Gender",
        "predicted_class": "Risk", "confidence": "Confidence", "urgency_tier": "Urgency tier",
    }
    show = view[list(display_cols.keys())].rename(columns=display_cols)
    show["Confidence"] = (show["Confidence"] * 100).round(1).astype(str) + "%"
    st.dataframe(show, use_container_width=True, hide_index=True)

    selected_id = st.selectbox(
        "Inspect a specific assessment (by row id)",
        options=view["id"].tolist(),
        format_func=lambda i: f"#{i} — {view.loc[view['id']==i,'label'].values[0] or 'Anonymous'} "
                               f"({view.loc[view['id']==i,'predicted_class'].values[0]})",
    ) if not view.empty else None

    if selected_id is not None:
        row = view[view["id"] == selected_id].iloc[0]
        factors = json.loads(row["top_factors"]) if row["top_factors"] else []
        st.markdown(
            f"""
            <div class="ccara-card">
            <b>#{row['id']} — {row['label'] or 'Anonymous'}</b> · {row['timestamp']}<br>
            Risk: <b>{row['predicted_class']}</b> ({row['confidence']*100:.1f}% confidence)
            {f" · Urgency tier: <b>{row['urgency_tier']}</b>" if row['urgency_tier'] else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if factors:
            st.markdown("Top contributing factors at the time of this assessment:")
            for f in factors:
                direction = f.get("direction", "increases" if f["shap"] > 0 else "decreases")
                arrow = "↑" if direction == "increases" else "↓"
                shown_val = f.get("display_value", f["value"])
                st.write(f"{arrow} {f['feature']} = {shown_val}  (SHAP {f['shap']:+.3f})")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Danger zone"):
        st.caption("Clears all locally stored assessment history. This cannot be undone.")
        if st.button("Clear all history", type="secondary"):
            du.clear_history()
            st.success("History cleared.")
            st.rerun()

branding.footer()
