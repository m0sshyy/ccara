"""Matplotlib charts styled with CCaRA brand colours.

Increasing-risk contributions are drawn in the High-risk red so the eye
reads them as "pushing toward danger"; protective / risk-lowering
contributions are drawn in brand green, echoing the green accent dot in
the CCaRA mark (the wordmark for the people the awareness reaches) while
staying visually distinct from the risk-severity red.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

INCREASE = "#b3263a"
DECREASE = "#4c9660"
NAVY = "#072754"
MUTED = "#5b6478"


def shap_waterfall_chart(contributions: dict, base_value: float, predicted_class: str,
                          engineered: dict, friendly_names: dict, top_n: int = 6,
                          display_values: dict | None = None):
    """A SHAP waterfall: baseline E[f(x)] -> each top feature's push -> f(x)
    for this patient, in the classic 'bridge chart' style (as used in the
    report's Figure 4.14), styled in CCaRA brand colours."""
    display_values = display_values or {}
    items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = items[:top_n]
    rest = items[top_n:]
    rest_sum = sum(v for _, v in rest)

    steps = []
    for feat, val in top:
        label = friendly_names.get(feat, feat)
        fv = display_values.get(feat, engineered.get(feat))
        steps.append((f"{label} = {fv}", val))
    if rest:
        steps.append((f"{len(rest)} other factor{'s' if len(rest) != 1 else ''} (combined)", rest_sum))

    n_rows = len(steps)
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * n_rows + 1.9))

    # Colour meaning is tied to *actual CRC risk direction*, not just "toward
    # vs away from the predicted class" - otherwise a Low-risk explanation
    # would paint reassuring factors red simply because they push toward
    # "Low". A factor pushing toward Low (or away from Moderate/High) is
    # always protective (green); pushing toward Moderate/High (or away from
    # Low) is always risk-increasing (red).
    favorable_class = predicted_class == "Low"

    cum = base_value
    lefts, widths, colors, mids, xs_conn = [], [], [], [], [base_value]
    for _, v in steps:
        if v >= 0:
            lefts.append(cum); widths.append(v)
        else:
            lefts.append(cum + v); widths.append(-v)
        risk_increasing = (v < 0) if favorable_class else (v > 0)
        colors.append(INCREASE if risk_increasing else DECREASE)
        mids.append(cum + v / 2)
        cum += v
        xs_conn.append(cum)
    final_value = cum

    y_positions = list(range(n_rows, 0, -1))
    bar_h = 0.52

    for y, left, width, color in zip(y_positions, lefts, widths, colors):
        ax.barh(y, width, left=left, height=bar_h, color=color, zorder=3,
                edgecolor="white", linewidth=0.6)

    # step connectors between consecutive bars
    for i in range(n_rows - 1):
        ax.plot([xs_conn[i + 1], xs_conn[i + 1]],
                 [y_positions[i] - bar_h / 2, y_positions[i + 1] + bar_h / 2],
                 color="#b7bfd1", linewidth=1, linestyle=(0, (2, 2)), zorder=1)

    ax.axvline(base_value, color=MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
    ax.axvline(final_value, color=NAVY, linewidth=1.6, zorder=2)

    for y, (_, v), mid in zip(y_positions, steps, mids):
        ax.text(mid, y, f"{v:+.2f}", ha="center", va="center", fontsize=8.3,
                 color="white", fontweight="bold", zorder=4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in steps], fontsize=9.6, color=NAVY)
    ax.set_ylim(0.3, n_rows + 0.95)

    ax.text(base_value, n_rows + 0.68, f"Baseline\nE[f(x)] = {base_value:.2f}",
             ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(final_value, n_rows + 0.68, f"This patient\nf(x) = {final_value:.2f}",
             ha="center", va="bottom", fontsize=8, color=NAVY, fontweight="bold")

    ax.set_xlabel(f"Contribution to '{predicted_class}' score (log-odds)", color=MUTED, fontsize=9)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#dfe4ee")
    ax.grid(axis="x", color="#e7eaf2", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    return fig


def probability_bar_chart(probabilities: dict):
    from .branding import RISK_COLORS
    classes = ["Low", "Moderate", "High"]
    values = [probabilities.get(c, 0.0) * 100 for c in classes]
    colors = [RISK_COLORS[c] for c in classes]

    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    bars = ax.bar(classes, values, color=colors, width=0.55, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)", color=MUTED, fontsize=9)
    ax.tick_params(axis="x", labelsize=10, colors=NAVY)
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#e7eaf2", zorder=0)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.1f}%",
                 ha="center", fontsize=9, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig
