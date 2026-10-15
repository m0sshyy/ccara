"""CCaRA — downloadable risk report (PDF).

Per the Lean Model Canvas ("Generate a risk report that can support
further screening decisions"), each completed assessment can be exported
as a one-page branded PDF summary. Built with fpdf2 (pure-Python, no
system dependencies) so it installs cleanly in the same conda environment
as the rest of the prototype.
"""

from __future__ import annotations

from datetime import datetime

from fpdf import FPDF

# fpdf2's core "Helvetica" font only supports strict Latin-1 (not even the
# full Windows-1252 set), so "smart" punctuation like en/em dashes, curly
# quotes, or arrows raises FPDFUnicodeEncodingException. Every piece of
# dynamic text is routed through sanitize() before being written so any
# future copy changes can't crash report generation again.
_UNICODE_REPLACEMENTS = {
    "–": "-", "—": "-",          # en dash, em dash
    "‘": "'", "’": "'",          # curly single quotes
    "“": '"', "”": '"',          # curly double quotes
    "…": "...",                        # ellipsis
    "•": "-",                          # bullet
    "→": "->", "↑": "^", "↓": "v",
    " ": " ",                          # non-breaking space
}


def sanitize(text) -> str:
    if text is None:
        return ""
    text = str(text)
    for uni, ascii_eq in _UNICODE_REPLACEMENTS.items():
        text = text.replace(uni, ascii_eq)
    # Final safety net: anything still outside Latin-1 gets dropped rather
    # than crashing the whole report.
    return text.encode("latin-1", "replace").decode("latin-1")


NAVY = (7, 39, 84)
PINK = (232, 178, 202)
GREEN = (117, 182, 131)
MUTED = (91, 100, 120)
RISK_RGB = {
    "Low": (76, 150, 96),
    "Moderate": (201, 138, 46),
    "High": (179, 38, 58),
}


class CCaRAReport(FPDF):
    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 24, "F")
        self.set_xy(12, 6)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "CCaRA", ln=1)
        self.set_x(12)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(210, 216, 232)
        self.cell(0, 5, "Colorectal Cancer Risk Awareness", ln=1)
        self.ln(10)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*MUTED)
        self.multi_cell(
            0, 4,
            "CCaRA is an awareness prototype, not a diagnostic device. It does not replace "
            "professional medical advice, screening, or diagnosis. Generated for demonstration "
            f"purposes on {datetime.now().strftime('%d %b %Y, %H:%M')}.",
            align="C",
        )


def _section_title(pdf: CCaRAReport, text: str):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, text, ln=1)
    pdf.set_draw_color(*PINK)
    pdf.set_line_width(0.8)
    y = pdf.get_y()
    pdf.line(12, y, 198, y)
    pdf.ln(3)


def build_report(*, label: str, form_values: dict, prediction, top_factors: list[dict],
                  recommendation: str) -> bytes:
    pdf = CCaRAReport()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    who = sanitize(label).strip() if label and label.strip() else "Anonymous user"
    pdf.cell(0, 6, f"Prepared for: {who}", ln=1)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%d %B %Y, %H:%M')}", ln=1)
    pdf.ln(4)

    # --- Result banner -------------------------------------------------
    risk_class = prediction.predicted_class
    color = RISK_RGB.get(risk_class, NAVY)
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, f"  Predicted Risk Level: {risk_class}", ln=1, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"  Model confidence: {prediction.confidence*100:.1f}%", ln=1, fill=True)
    if prediction.urgency_tier:
        pdf.cell(0, 8, f"  Urgency sub-tier: {prediction.urgency_tier}", ln=1, fill=True)
    pdf.ln(6)

    # --- Recommendation --------------------------------------------------
    _section_title(pdf, "Recommendation")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 6, sanitize(recommendation))
    pdf.ln(3)

    # --- Class probabilities ---------------------------------------------
    _section_title(pdf, "Class Probabilities")
    pdf.set_font("Helvetica", "", 10)
    for cls in ["Low", "Moderate", "High"]:
        p = prediction.probabilities.get(cls, 0.0)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(30, 6, cls)
        bar_w = 100 * p
        pdf.set_fill_color(*RISK_RGB.get(cls, NAVY))
        pdf.cell(bar_w, 6, "", fill=True)
        pdf.cell(0, 6, f"  {p*100:.1f}%", ln=1)
    pdf.ln(3)

    # --- Top contributing factors -----------------------------------------
    # item["direction"] already reflects actual CRC risk direction (red =
    # increases risk, green = protective), computed in top_contributors().
    _section_title(pdf, "Top Contributing Factors (SHAP / Tree SHAP)")
    pdf.set_font("Helvetica", "", 10)
    for item in top_factors:
        risk_increasing = item["direction"] == "increases"
        arrow = "+" if risk_increasing else "-"
        colr = (179, 38, 58) if risk_increasing else (76, 150, 96)
        pdf.set_text_color(*colr)
        pdf.cell(8, 6, arrow)
        pdf.set_text_color(20, 20, 20)
        shown_val = item.get("display_value", item["value"])
        line = f"{item['feature']}: {shown_val}  (SHAP {item['shap']:+.3f}, {item['direction']} risk)"
        pdf.cell(0, 6, sanitize(line), ln=1)
    pdf.ln(3)

    # --- Inputs summary -----------------------------------------------
    _section_title(pdf, "Patient Inputs")
    pdf.set_font("Helvetica", "", 9.5)
    labels = {
        "Age": "Age", "Gender": "Gender", "Family_History": "Family history",
        "Smoking_History": "Smoking history", "Alcohol_Consumption": "Alcohol consumption",
        "Obesity_BMI": "BMI category", "Diet_Risk": "Diet risk",
        "Physical_Activity": "Physical activity", "Diabetes": "Diabetes",
        "Inflammatory_Bowel_Disease": "IBD", "Genetic_Mutation": "Genetic mutation",
    }
    col_w = 95
    items = list(labels.items())
    for i in range(0, len(items), 2):
        pdf.set_text_color(20, 20, 20)
        left_key, left_label = items[i]
        text = f"{left_label}: {form_values.get(left_key)}"
        pdf.cell(col_w, 6, sanitize(text))
        if i + 1 < len(items):
            right_key, right_label = items[i + 1]
            text2 = f"{right_label}: {form_values.get(right_key)}"
            pdf.cell(col_w, 6, sanitize(text2))
        pdf.ln(6)

    return bytes(pdf.output())
