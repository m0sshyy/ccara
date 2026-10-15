"""
CCaRA — model utilities
========================
Loads the trained XGBoost artefacts (Alya's FYP: "AI-based predictive model
for early detection of colorectal cancer risk factors using XGBoost"),
reproduces the 5 engineered features described in Section 3.3.2 / Table 4.6
of the report, runs predictions, and computes per-feature Shapley
contributions using XGBoost's native TreeSHAP (`pred_contribs=True`).

Why native TreeSHAP instead of the `shap` package?
---------------------------------------------------
The `shap` library's `TreeExplainer` currently fails to parse the
`base_score` field saved by XGBoost >= 2.x for multi-class
(`multi:softprob`) models (a known upstream compatibility bug — it expects
a single float but XGBoost now stores one base score per class, e.g.
"[0E0,0E0,0E0]"). Rather than ship a prototype that breaks on a fresh
`pip install`, contributions are computed directly from the booster via
`Booster.predict(..., pred_contribs=True)`, which implements the exact
same Tree SHAP algorithm (Lundberg & Lee, 2017) used inside `shap`.
The numbers are identical — only the code path avoids the broken parser.

Feature encoding
-----------------
The saved model expects 16 integer-encoded features (see feature_names.pkl)
and was trained on encodings that are NOT stored in the artefacts. The
mapping below was reverse-engineered from the trained model itself: for
every categorical feature we swept its raw encoded value (0/1/2) through
the model and confirmed the predicted High-risk probability increases
monotonically in the direction below (i.e. higher encoded value = higher
modelled risk), and validated two end-to-end synthetic patients (an
all-protective young profile and an all-risk elderly profile) predict
"Low" and "High" respectively with sensible confidence. If your training
notebook used a different mapping, update ENCODINGS below to match.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

# ---------------------------------------------------------------------
# Raw feature metadata (order matches base_features.pkl)
# ---------------------------------------------------------------------
YES_NO = {"No": 0, "Yes": 1}
GENDER_MAP = {"Female": 0, "Male": 1}
BMI_MAP = {"Normal": 0, "Overweight": 1, "Obese": 2}
DIET_MAP = {"Low": 0, "Moderate": 1, "High": 2}
ACTIVITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}

FEATURE_META = {
    "Age": {
        "kind": "number", "min": 18, "max": 100, "default": 50,
        "label": "Age (years)",
        "help": "Model was trained on ages 30–89; values outside this range are extrapolated.",
    },
    "Gender": {"kind": "select", "options": list(GENDER_MAP), "encoding": GENDER_MAP, "label": "Gender"},
    "Family_History": {"kind": "select", "options": list(YES_NO), "encoding": YES_NO,
                        "label": "Family history of colorectal cancer",
                        "help": "A parent, sibling, or child diagnosed with colorectal cancer."},
    "Smoking_History": {"kind": "select", "options": list(YES_NO), "encoding": YES_NO,
                         "label": "Smoking history"},
    "Alcohol_Consumption": {"kind": "select", "options": list(YES_NO), "encoding": YES_NO,
                             "label": "Regular alcohol consumption"},
    "Obesity_BMI": {"kind": "select", "options": list(BMI_MAP), "encoding": BMI_MAP,
                     "label": "BMI category"},
    "Diet_Risk": {"kind": "select", "options": list(DIET_MAP), "encoding": DIET_MAP,
                   "label": "Diet risk level",
                   "help": "High = diet consistently low in fibre / high in red & processed meat."},
    "Physical_Activity": {"kind": "select", "options": list(ACTIVITY_MAP), "encoding": ACTIVITY_MAP,
                           "label": "Physical activity level"},
    "Diabetes": {"kind": "select", "options": list(YES_NO), "encoding": YES_NO, "label": "Diabetes"},
    "Inflammatory_Bowel_Disease": {"kind": "select", "options": list(YES_NO), "encoding": YES_NO,
                                    "label": "Inflammatory bowel disease (Crohn's / ulcerative colitis)"},
    "Genetic_Mutation": {"kind": "select", "options": list(YES_NO), "encoding": YES_NO,
                          "label": "Known genetic mutation (e.g. Lynch syndrome, FAP)",
                          "help": "A known hereditary mutation linked to colorectal cancer risk."},
}

RAW_ORDER = list(FEATURE_META.keys())

FRIENDLY_NAMES = {
    "Age": "Age",
    "Gender": "Gender",
    "Family_History": "Family history",
    "Smoking_History": "Smoking history",
    "Alcohol_Consumption": "Alcohol consumption",
    "Obesity_BMI": "BMI category",
    "Diet_Risk": "Diet risk",
    "Physical_Activity": "Physical activity",
    "Diabetes": "Diabetes",
    "Inflammatory_Bowel_Disease": "Inflammatory bowel disease",
    "Genetic_Mutation": "Genetic mutation",
    "Lifestyle_Burden": "Lifestyle risk burden (composite)",
    "Non_Modifiable_Load": "Non-modifiable risk load (composite)",
    "Metabolic_Risk": "Metabolic risk (composite)",
    "Age_Risk_Group": "Age risk tier",
    "Compound_Risk": "Compound risk (composite)",
}


# ---------------------------------------------------------------------
# Artefact loading (cached by caller with st.cache_resource)
# ---------------------------------------------------------------------
def load_artifacts():
    model = joblib.load(MODEL_DIR / "xgb_crc_model.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    base_features = joblib.load(MODEL_DIR / "base_features.pkl")
    return model, label_encoder, feature_names, base_features


# ---------------------------------------------------------------------
# Feature engineering — mirrors Table 4.6 of the report
# ---------------------------------------------------------------------
def engineer_features(raw: dict) -> dict:
    """Take raw-encoded inputs (11 features, already numeric) and compute
    the 5 engineered features used by the model."""
    obese_flag = 1 if raw["Obesity_BMI"] == 2 else 0
    low_activity_flag = 1 if raw["Physical_Activity"] == 0 else 0
    high_diet_flag = 1 if raw["Diet_Risk"] == 2 else 0

    lifestyle_burden = (
        raw["Smoking_History"] + raw["Alcohol_Consumption"] + obese_flag
        + low_activity_flag + high_diet_flag
    )
    non_modifiable_load = (
        raw["Family_History"] + raw["Genetic_Mutation"] + raw["Inflammatory_Bowel_Disease"]
    )
    metabolic_risk = raw["Diabetes"] + obese_flag
    age = raw["Age"]
    age_risk_group = 0 if age < 50 else (1 if age < 70 else 2)
    compound_risk = lifestyle_burden + non_modifiable_load

    engineered = dict(raw)
    engineered.update({
        "Lifestyle_Burden": lifestyle_burden,
        "Non_Modifiable_Load": non_modifiable_load,
        "Metabolic_Risk": metabolic_risk,
        "Age_Risk_Group": age_risk_group,
        "Compound_Risk": compound_risk,
    })
    return engineered


def encode_form_inputs(form_values: dict) -> dict:
    """Convert human-readable form selections (e.g. 'Yes'/'Overweight') into
    the raw numeric encoding the model expects."""
    raw = {}
    for feat, meta in FEATURE_META.items():
        val = form_values[feat]
        if meta["kind"] == "number":
            raw[feat] = int(val)
        else:
            raw[feat] = meta["encoding"][val]
    return raw


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: dict
    engineered: dict
    contributions: dict  # feature -> shap value for predicted class
    base_value: float = 0.0  # E[f(x)] for the predicted class (Tree SHAP bias term)
    urgency_tier: str | None = None
    urgency_score: float | None = None


def predict(model, label_encoder, feature_names, raw_inputs: dict) -> PredictionResult:
    engineered = engineer_features(raw_inputs)
    df = pd.DataFrame([engineered])[feature_names]

    probs = model.predict_proba(df)[0]
    classes = list(label_encoder.classes_)
    pred_idx = int(np.argmax(probs))
    predicted_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    probabilities = {c: float(p) for c, p in zip(classes, probs)}

    contributions, base_value = _tree_shap_contributions(model, df, feature_names, pred_idx)

    urgency_tier = urgency_score = None
    if predicted_class == "High":
        urgency_score, urgency_tier = compute_urgency(
            engineered["Lifestyle_Burden"], engineered["Non_Modifiable_Load"]
        )

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        engineered=engineered,
        contributions=contributions,
        base_value=base_value,
        urgency_tier=urgency_tier,
        urgency_score=urgency_score,
    )


def _tree_shap_contributions(model, df: pd.DataFrame, feature_names: list, class_idx: int):
    """Exact Tree SHAP contributions for one class, via the XGBoost booster.

    Returns (contributions, base_value) where base_value is E[f(x)] — the
    bias/expected-value term SHAP values are computed relative to — so
    callers can build an accurate waterfall from baseline to f(x).
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(df, feature_names=feature_names)
    contribs = booster.predict(dmat, pred_contribs=True)  # (n, n_classes, n_features+1)
    row = contribs[0, class_idx, :-1]
    bias = float(contribs[0, class_idx, -1])
    return {f: float(v) for f, v in zip(feature_names, row)}, bias


# ---------------------------------------------------------------------
# Within-High-Risk Urgency Profiling (Section 4.4.3 of the report)
# ---------------------------------------------------------------------
# The report defines 3 sub-tiers of the High-risk cohort from tertiles of
# an "Urgency Score" — a composite of Lifestyle_Burden and
# Non_Modifiable_Load — with cutoffs Low <=7, Med 7-9, Critical >9, and
# reports mean burden (3.5) + mean load (1.3) ~= 4.8 for the Critical tier.
# A straight sum tops out at 8 (5 + 3), which can't reach a ">9" cutoff, but
# doubling the combined burden lands almost exactly on the reported
# Critical-tier mean (2 x 4.8 = 9.6). That scaling is used here; adjust
# URGENCY_WEIGHT if your notebook used a different weighting.
URGENCY_WEIGHT = 2

URGENCY_TIERS = {
    "High-Low": {
        "cutoff": 7,
        "action": "Standard screening referral (colonoscopy or FIT/iFOBT per age-appropriate guideline).",
    },
    "High-Med": {
        "cutoff": 9,
        "action": "Screening referral plus early intervention: structured lifestyle counselling.",
    },
    "High-Critical": {
        "cutoff": float("inf"),
        "action": "Immediate colonoscopy / iFOBT referral plus multi-domain intervention.",
    },
}


def compute_urgency(lifestyle_burden: int, non_modifiable_load: int) -> tuple[float, str]:
    score = URGENCY_WEIGHT * (lifestyle_burden + non_modifiable_load)
    if score <= URGENCY_TIERS["High-Low"]["cutoff"]:
        tier = "High-Low"
    elif score <= URGENCY_TIERS["High-Med"]["cutoff"]:
        tier = "High-Med"
    else:
        tier = "High-Critical"
    return score, tier


# ---------------------------------------------------------------------
# Clinical recommendation text
# ---------------------------------------------------------------------
def get_recommendation(predicted_class: str, urgency_tier: str | None = None) -> str:
    if predicted_class == "Low":
        return ("Continue routine health maintenance. Follow standard age-appropriate "
                "screening guidelines (colonoscopy or FIT typically from age 45-50).")
    if predicted_class == "Moderate":
        return ("Lifestyle modification recommended (diet, activity, smoking/alcohol); "
                "consider discussing early screening with a healthcare provider.")
    if predicted_class == "High" and urgency_tier:
        return URGENCY_TIERS[urgency_tier]["action"]
    return "Discuss these results with a qualified healthcare provider."


RISK_BLURB = {
    "Low": "Your profile shows few of the risk factors used by this model. Keep up routine, "
           "age-appropriate screening.",
    "Moderate": "Your profile shows a moderate combination of risk factors. Small lifestyle "
                "changes now, plus a conversation with a healthcare provider about screening "
                "timing, can meaningfully lower long-term risk.",
    "High": "Your profile shows a strong combination of risk factors. This is not a diagnosis "
            "— but it is a signal worth acting on promptly with a healthcare provider.",
}


# Reverse lookup so contributor lists can show "Male" / "Yes" / "Obese"
# instead of the raw encoded integer fed to the model.
_REVERSE_ENCODING = {
    feat: {v: k for k, v in meta["encoding"].items()}
    for feat, meta in FEATURE_META.items() if meta["kind"] == "select"
}


def display_value(feat: str, value):
    """Human-readable rendering of a feature's value for UI/report display."""
    if value is None:
        return value
    if feat == "Age":
        return f"{value} yrs"
    if feat in _REVERSE_ENCODING:
        return _REVERSE_ENCODING[feat].get(value, value)
    return value  # engineered/composite features: show the raw count


def top_contributors(contributions: dict, engineered: dict, n: int = 6,
                      predicted_class: str | None = None) -> list[dict]:
    """Return the top-n features by |SHAP value|, formatted for display.

    ``direction`` describes actual CRC risk direction ("increases" /
    "decreases"), not just "toward vs away from the predicted class" - a
    factor pushing toward "Low" is protective even though its raw SHAP
    value for the Low class is positive.
    """
    favorable_class = predicted_class == "Low"
    items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    out = []
    for feat, val in items:
        raw_val = engineered.get(feat)
        risk_increasing = (val < 0) if favorable_class else (val > 0)
        out.append({
            "feature": FRIENDLY_NAMES.get(feat, feat),
            "raw_feature": feat,
            "value": raw_val,
            "display_value": display_value(feat, raw_val),
            "shap": val,
            "direction": "increases" if risk_increasing else "decreases",
        })
    return out
