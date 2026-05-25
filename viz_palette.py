"""
Shared visualization palette and design tokens for the VideoHandTracker report.

Based on research-backed best practices:
- Wong palette (Nature Methods, 2011) for colorblind-safe categorical data
- Viridis for sequential ordinal data (perceptually uniform)
- RdBu for diverging data (sentiment, above/below baseline)

Reference:
- Wong, B. (2011). Color blindness. Nature Methods, 8(6), 441.
- Tufte, E. R. (2001). The Visual Display of Quantitative Information.
- Nielsen Norman Group: Contrast: One of the 3Cs for Better Charts.
"""

# ---------- Categorical palettes (colorblind-safe) ----------
# Wong palette — distinguishable for protanopia, deuteranopia, and tritanopia.
WONG = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
    "#009E73",  # Bluish green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish purple
]

# Compact 5-colour subset for use cases where 4-6 categories is the sweet spot.
WONG_COMPACT = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]

# ---------- Sequential palettes (perceptually uniform) ----------
VIRIDIS = "Viridis"   # purple → blue → green → yellow
INFERNO = "Inferno"   # black → red → yellow
CIVIDIS = "Cividis"   # blue → yellow, optimised for deuteranopia

# ---------- Diverging palettes (data with a meaningful midpoint) ----------
RDBU = "RdBu"         # red ↔ blue, classic for sentiment / signed data
BRBG = "BrBG"         # brown ↔ blue-green, for diverging earth-tone data

# ---------- Semantic colour mapping for this project ----------
# Use these throughout so the same concept always has the same colour.
SEMANTIC = {
    "left_hand":   "#0072B2",   # Wong blue
    "right_hand":  "#D55E00",   # Wong vermillion
    "positive":    "#009E73",   # Wong bluish green
    "negative":    "#CC79A7",   # Wong reddish purple
    "neutral":     "#999999",   # Mid grey
    "highlight":   "#E69F00",   # Wong orange — for "look here"
    "muted":       "#CCCCCC",   # Light grey — for de-emphasised context
}

# ---------- Typography hierarchy ----------
# Use as kwargs to plotly layout updates.
TYPOGRAPHY = {
    "title":    dict(size=22, family="Inter, Segoe UI, sans-serif", color="#1a1a1a"),
    "subtitle": dict(size=14, family="Inter, Segoe UI, sans-serif", color="#666666"),
    "axis":     dict(size=12, family="Inter, Segoe UI, sans-serif", color="#333333"),
    "legend":   dict(size=11, family="Inter, Segoe UI, sans-serif", color="#333333"),
    "annotation": dict(size=10, family="Inter, Segoe UI, sans-serif", color="#555555"),
}

# ---------- Spacing tokens (px) — follow a 4/8 scale ----------
SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 32, "xl": 48}


def apply_clean_layout(fig, title=None, insight=None, height=None):
    """
    Apply consistent layout to a Plotly figure following Tufte principles.

    Args:
        fig: plotly.graph_objects.Figure
        title: Main title text (states what the chart IS)
        insight: Optional subtitle stating the KEY INSIGHT in plain language
        height: Optional explicit height in pixels
    """
    layout_updates = dict(
        font=TYPOGRAPHY["axis"],
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=30, t=80 if title else 30, b=60),
        legend=dict(
            font=TYPOGRAPHY["legend"],
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e5e5e5",
            borderwidth=1,
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#f0f0f0", zerolinecolor="#cccccc",
            title=dict(font=TYPOGRAPHY["axis"]),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#f0f0f0", zerolinecolor="#cccccc",
            title=dict(font=TYPOGRAPHY["axis"]),
        ),
    )
    if title:
        title_text = f"<b>{title}</b>"
        if insight:
            title_text += f"<br><span style='font-size:13px;color:#666;font-weight:normal'>{insight}</span>"
        layout_updates["title"] = dict(text=title_text, font=TYPOGRAPHY["title"], x=0.02, xanchor="left")
    if height:
        layout_updates["height"] = height
    fig.update_layout(**layout_updates)
    return fig
