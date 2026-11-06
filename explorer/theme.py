"""3Blue1Brown-inspired palette, Plotly template and page CSS."""

import plotly.graph_objects as go
import plotly.io as pio

BG = "#0B0D12"
PANEL = "#141821"
INK = "#ECECEC"
MUTED = "#8A8F98"
LINE = "#2A2F3A"

BLUE = "#58C4DD"
TEAL = "#5CD0B3"
GREEN = "#83C167"
YELLOW = "#FFFF00"
GOLD = "#F0AC5F"
RED = "#FC6255"
PURPLE = "#9A72AC"
GREY = "#888888"

SERIF = "'CMU Serif', 'Latin Modern Roman', Georgia, 'Times New Roman', serif"
SANS = "Inter, 'Helvetica Neue', Arial, sans-serif"

# Continuous scales
SCALE_BLUE_TEAL = [[0, "#1C3F6E"], [0.5, BLUE], [1, TEAL]]
SCALE_HEAT = [[0, BG], [0.35, "#3B2F6B"], [0.7, RED], [1, YELLOW]]
SCALE_DIVERGING = [[0, BLUE], [0.5, PANEL], [1, RED]]
SCALE_INK = [[0, "#000000"], [1, "#FFFFFF"]]


def register_template():
    pio.templates["3b1b"] = go.layout.Template(layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, color=INK, size=13),
        title=dict(font=dict(family=SERIF, size=18, color=INK), x=0.0, xanchor="left"),
        colorway=[BLUE, GOLD, TEAL, RED, PURPLE, GREEN, YELLOW, GREY],
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.18)",
                   linecolor=LINE, tickcolor=LINE, title_font=dict(family=SERIF)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.18)",
                   linecolor=LINE, tickcolor=LINE, title_font=dict(family=SERIF)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(family=SANS, color=INK)),
        margin=dict(l=40, r=20, t=50, b=40),
    ))
    pio.templates.default = "3b1b"


PAGE_CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/npm/computer-modern@0.1.2/cmu-serif.css');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: {BG}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem !important; padding-bottom: 3rem; max-width: 1500px; }}
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}

h1, h2, h3, h4 {{ font-family: {SERIF} !important; font-weight: 400 !important; letter-spacing: 0; color: {INK}; }}
p, li, label, .stMarkdown {{ font-family: {SANS}; }}

.hero-title {{ font-family: {SERIF}; font-size: 2.6rem; line-height: 1.1; margin: 0; }}
.hero-title em {{ color: {BLUE}; font-style: italic; }}
.hero-sub {{ color: {MUTED}; font-size: 1.02rem; margin: .35rem 0 0 0; max-width: 60ch; }}

.chip {{ display: inline-flex; align-items: center; gap: .45rem; padding: .32rem .75rem; border-radius: 999px;
        border: 1px solid {LINE}; background: {PANEL}; color: {INK}; font-size: .9rem; margin-right: .4rem; }}
.chip b {{ font-family: {SERIF}; font-size: 1.15rem; font-weight: 400; }}
.chip.gold b {{ color: {GOLD}; }}
.chip.blue b {{ color: {BLUE}; }}
.chip.ok {{ border-color: rgba(131,193,103,.5); color: {GREEN}; }}
.chip.bad {{ border-color: rgba(252,98,85,.5); color: {RED}; }}

.narrative {{ font-family: {SERIF}; font-size: 1.12rem; line-height: 1.65; color: #D6D6D6; max-width: 72ch; }}
.narrative .b {{ color: {BLUE}; }} .narrative .g {{ color: {GOLD}; }} .narrative .t {{ color: {TEAL}; }}
.narrative .r {{ color: {RED}; }} .narrative .y {{ color: {YELLOW}; }}
.caption {{ color: {MUTED}; font-size: .86rem; }}

.card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; padding: .9rem 1.1rem; }}
.card .k {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
.card .v {{ font-family: {SERIF}; font-size: 1.6rem; color: {INK}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {SERIF}; font-size: 1.15rem; color: {MUTED}; padding: .6rem 0; background: transparent; }}
.stTabs [aria-selected="true"] {{ color: {INK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {BLUE}; }}

.stButton > button {{ background: transparent; color: {INK}; border: 1px solid {LINE}; border-radius: 999px;
        font-family: {SANS}; transition: border-color .2s, color .2s; }}
.stButton > button:hover {{ border-color: {BLUE}; color: {BLUE}; }}
[data-testid="stMetricValue"] {{ font-family: {SERIF}; font-weight: 400; }}
.katex {{ font-size: 1.15em; }}
iframe {{ border-radius: 12px; }}
</style>
"""
