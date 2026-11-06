"""
The 3D routing cube, rendered as a client-side Plotly.js page so clicks and
iteration playback never round-trip to Streamlit.

Geometry: a G x G floor grid with the T capsule types stacked above each cell,
class capsules in a row above the stacks, routes drawn as arcs, and the input
image and the capsule attribution projection on planes below the grid. Grid row gy = 0 sits at
the top and columns are mirrored so that each capsule column lies over its
receptive field when seen from the default camera.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from . import analysis as A
from . import theme as T
from .protocol import ROUTE_DRAW_MIN, TOPK_FRACS

SPACING = 1.5
STACK_HEIGHT = 4.8          # total height of a capsule-type stack; types sit STACK_HEIGHT / T apart
REF_TYPES = 32              # layout reference: marker size is shrunk for denser stacks
ELEVATION = 8.0
DIGIT_STEP = 0.8
IMAGE_Y = -4.0
INFLUENCE_Y = -1.0
PX = 0.5                    # scene units per input pixel
N_BINS = 8

# Trace order is shared with hero.js
TR_PRIMARY, TR_DIGITS, TR_IMAGE, TR_INFLUENCE, TR_GRID, TR_LABELS, TR_FOCUS = range(7)
TR_BIN0 = 7

METHODS = [  # (key, label) shown in the attribution switcher
    ("exact", "exact ablation Δ"),
    ("taylor", "Taylor ⟨∂φ/∂u, u⟩"),
    ("taylor_frozen", "Taylor, routing frozen"),
    ("g_A", "‖∂φ/∂u‖"),
    ("g_C", "|∂φ/∂c|"),
    ("coupling", "coupling c (baseline)"),
    ("activation", "activation ‖u‖ (baseline)"),
]


class Layout:
    def __init__(self, geo):
        self.geo = geo
        self.G, self.T = geo.G, geo.T
        self.type_step = STACK_HEIGHT / geo.T
        self.extent = geo.G * SPACING
        self.center = (geo.G - 1) / 2 * SPACING
        self.img_w = geo.img_size * PX
        self.img_x0 = self.extent / 2 - self.img_w / 2

    def x(self, gx):
        return (self.G - 1 - gx + 0.5) * SPACING

    def z(self, gy):
        return (self.G - 1 - gy + 0.5) * SPACING

    def primary_positions(self):
        pos = np.zeros((self.geo.N, 3))
        for i in range(self.geo.N):
            gx, gy, t = self.geo.coords(i)
            pos[i] = (self.x(gx), t * self.type_step, self.z(gy))
        return pos

    def digit_positions(self, K):
        return np.array([(self.center, ELEVATION, self.center + (i - (K - 1) / 2) * DIGIT_STEP)
                         for i in range(K)])


def _bin_style(k):
    """Width/colour of strength bin k. Width is the original max(1, 8·strength);
    colour runs deep blue -> purple -> red and alpha falls off faster than the
    original min(0.8, strength) so hundreds of weak lines don't wash to white."""
    m = (k + 0.5) / N_BINS
    stops = [np.array([47, 91, 211]), np.array([154, 114, 172]), np.array([252, 98, 85])]
    lo, f = (0, m * 2) if m < 0.5 else (1, (m - 0.5) * 2)
    r, g, b = (stops[lo] + (stops[lo + 1] - stops[lo]) * f).astype(int)
    # Plotly 3D lines ignore rgba alpha (they wash to white), so opacity is per trace.
    return max(1.0, m * 8), f"rgb({r},{g},{b})", round(max(0.04, min(0.85, m ** 1.6)), 3)


def _norm01(grid):
    lo, hi = float(np.min(grid)), float(np.max(grid))
    return ((grid - lo) / (hi - lo) if hi > lo else np.zeros_like(grid)), lo, hi


def build_payload(trace, attr, classes):
    geo = trace["geo"]
    lay = Layout(geo)
    iters = trace["iterations"]
    K = iters[0]["c"].shape[0]
    methods = [(k, lab) for k, lab in METHODS if k in attr]
    maps = {}
    for k, _ in methods:
        g = A.grid_sum(geo, attr[k])
        norm, lo, hi = _norm01(g)
        maps[k] = {"norm": np.round(norm, 4).tolist(), "raw": g.tolist(), "lo": lo, "hi": hi,
                   **({} if k == "exact" else A.agreement(attr[k], attr["exact"]))}
    return {
        "nIter": len(iters), "predicted": trace["predicted"], "classes": list(classes),
        "G": geo.G, "T": geo.T, "N": geo.N,
        "primaryPos": np.round(lay.primary_positions(), 3).tolist(),
        "digitPos": lay.digit_positions(K).tolist(),
        "coupling": [np.round(it["c"], 5).tolist() for it in iters],
        "entropy": [np.round(A.normalized_entropy(it["c"]), 4).tolist() for it in iters],
        "digitNorm": np.round(trace["digit_norms"], 4).tolist(),
        # Lines are normalized by the largest coupling of ANY class in ANY iteration,
        # so the uniform first iteration and non-predicted classes read as weak.
        "digitMax": [float(max(it["c"].max() for it in iters))] * K,
        "activation": np.round(trace["u_norm"], 4).tolist(),
        "exact": np.round(attr["exact"], 6).tolist(),
        "taylor": np.round(attr["taylor"], 6).tolist(),
        "methods": methods, "maps": maps,
        "bins": [_bin_style(k) for k in range(N_BINS)],
        "topLabel": f"{100 * TOPK_FRACS[0]:g}%",
        "routeMin": ROUTE_DRAW_MIN,
        "arc": {"points": 20, "height": 3.0},
        "colors": {"blue": T.BLUE, "red": T.RED, "gold": T.GOLD, "teal": T.TEAL,
                   "ink": T.INK, "muted": T.MUTED, "yellow": T.YELLOW},
    }


def _digit_style(norms, pred):
    peak = max(norms.max(), 1e-8)
    sizes, colors = [], []
    gold = np.array([240, 172, 95])
    for i, n in enumerate(norms):
        sizes.append(max(8, n * 20))
        k = 0.25 + 0.75 * min(1.0, n / peak)
        rgb = (gold * k) if i == pred else (np.array([215, 215, 215]) * k)
        colors.append("rgb({},{},{})".format(*rgb.astype(int)))
    return sizes, colors


def _image_mesh(lay, image):
    """Input image as a flat Mesh3d with one colour per pixel (works for RGB)."""
    H = W = lay.geo.img_size
    img = np.asarray(image)
    rgb = np.repeat(img[..., None], 3, axis=-1) if img.ndim == 2 else img
    xs = lay.img_x0 + lay.img_w - np.arange(W + 1) * PX       # mirrored like the grid
    zs = lay.img_x0 + lay.img_w - np.arange(H + 1) * PX       # row 0 at the top
    X, Z = np.meshgrid(xs, zs)
    vid = lambda r, c: r * (W + 1) + c
    I, J, Kf, colors = [], [], [], []
    for r in range(H):
        for c in range(W):
            col = "rgb({},{},{})".format(*(np.clip(rgb[r, c], 0, 1) * 255).astype(int))
            a, b, d, e = vid(r, c), vid(r, c + 1), vid(r + 1, c), vid(r + 1, c + 1)
            I += [a, b]; J += [b, e]; Kf += [d, d]; colors += [col, col]
    return go.Mesh3d(x=X.ravel(), y=np.full(X.size, IMAGE_Y), z=Z.ravel(), i=I, j=J, k=Kf,
                     facecolor=colors, flatshading=True, hoverinfo="skip", name="Input image",
                     lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0))


def build_figure(trace, attr, classes):
    """Base scene at the final iteration. hero.js updates it in place."""
    geo = trace["geo"]
    lay = Layout(geo)
    it = len(trace["iterations"]) - 1
    c = trace["iterations"][it]["c"]
    K = c.shape[0]
    pred = trace["predicted"]
    u_norm = trace["u_norm"]
    pos = lay.primary_positions()
    dpos = lay.digit_positions(K)

    fig = go.Figure()

    hover = [f"Primary capsule {i}<br>grid ({gx}, {gy}) · type {t}<br>"
             f"Δ if removed: {attr['exact'][i]:+.5f}"
             for i, (gx, gy, t) in ((i, geo.coords(i)) for i in range(geo.N))]
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers", name="Primary capsules",
        marker=dict(size=np.maximum(2, u_norm * (8 if geo.T <= REF_TYPES else 6)),
                    color=A.normalized_entropy(c), cmin=0, cmax=1,
                    colorscale=[[0, "#2F5BD3"], [0.5, T.BLUE], [1, T.TEAL]],
                    opacity=0.75, line=dict(width=0.5, color="rgba(255,255,255,0.25)")),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>"))

    sizes, colors = _digit_style(trace["digit_norms"][it], pred)
    fig.add_trace(go.Scatter3d(
        x=dpos[:, 0], y=dpos[:, 1], z=dpos[:, 2], mode="markers+text", name="Class capsules",
        text=list(classes), textposition="middle right",
        textfont=dict(family=T.SERIF, size=15, color=T.INK),
        marker=dict(size=sizes, color=colors, opacity=0.95,
                    line=dict(width=2, color="rgba(255,255,255,0.8)")),
        customdata=[f"Class capsule {i} · {n}" for i, n in enumerate(classes)],
        hovertemplate="%{customdata}<extra>click to route</extra>"))

    fig.add_trace(_image_mesh(lay, trace["image"]))

    infl = A.grid_sum(geo, attr["exact"])
    norm, lo, hi = _norm01(infl)
    IX, IZ = np.meshgrid([lay.x(g) for g in range(geo.G)], [lay.z(g) for g in range(geo.G)])
    fig.add_trace(go.Surface(
        x=IX, y=np.full_like(IX, INFLUENCE_Y), z=IZ, surfacecolor=norm, customdata=infl,
        cmin=0, cmax=1, colorscale=T.SCALE_HEAT, opacity=0.8, name="Capsule attribution projection",
        showscale=True, colorbar=dict(title=dict(text="attribution", side="right",
                                                 font=dict(family=T.SERIF, color=T.MUTED)),
                                      x=0.98, len=0.35, y=0.3, thickness=10,
                                      tickfont=dict(color=T.MUTED), outlinewidth=0),
        hovertemplate="Σ over types: %{customdata:.5f}<extra></extra>"))

    gx_, gy_, gz_ = [], [], []
    for i in range(geo.G + 1):
        p = i * SPACING
        gx_ += [p, p, None, 0, lay.extent, None]
        gy_ += [0.1, 0.1, None, 0.1, 0.1, None]
        gz_ += [0, lay.extent, None, p, p, None]
    fig.add_trace(go.Scatter3d(x=gx_, y=gy_, z=gz_, mode="lines", hoverinfo="skip",
                               line=dict(color="rgba(200,200,200,0.22)", width=1),
                               showlegend=False, name="grid"))

    fig.add_trace(go.Scatter3d(
        x=[lay.img_x0, 0.0, lay.center],
        y=[IMAGE_Y, STACK_HEIGHT + 0.6, ELEVATION],
        z=[lay.img_x0 + lay.img_w + 0.6, lay.extent + 0.6,
           lay.center + ((K - 1) / 2 + 1.8) * DIGIT_STEP],
        mode="text", hoverinfo="skip", showlegend=False, name="labels",
        text=[f"input  {geo.img_size}×{geo.img_size}",
              f"primary capsules  {geo.G}×{geo.G}×{geo.T} · {geo.d_in}D",
              f"class capsules  {K} · {geo.d_out}D"],
        textfont=dict(family=T.SERIF, size=14, color=T.MUTED), textposition="middle right"))

    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip",
                               showlegend=False, name="focus", line=dict(width=5, color=T.YELLOW)))
    for k in range(N_BINS):
        width, color, opacity = _bin_style(k)
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip",
                                   showlegend=False, name=f"bin{k}", opacity=opacity,
                                   line=dict(width=width, color=color)))

    axis = dict(showticklabels=False, showgrid=False, zeroline=False, showbackground=False,
                title="", showspikes=False)
    fig.update_layout(
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                   bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=1.25, y=1.05, z=1.25), center=dict(x=0, y=-0.1, z=-0.05))),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision="hero",
        font=dict(family=T.SANS, color=T.INK),
        hoverlabel=dict(bgcolor=T.PANEL, bordercolor=T.LINE, font=dict(color=T.INK)),
    )
    return fig


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body{margin:0;height:100%;background:transparent;overflow:hidden;
    font-family:__SANS__;color:__INK__}
  #wrap{position:relative;width:100%;height:__H__px;
    background:radial-gradient(ellipse at 50% 40%,#161B26 0%,#0B0D12 70%);border-radius:12px}
  #hero{position:absolute;inset:0}
  .panel{position:absolute;z-index:5;background:rgba(20,24,33,.82);border:1px solid #2A2F3A;
    border-radius:10px;backdrop-filter:blur(6px);font-size:13px}
  #hint{top:14px;left:16px;padding:8px 12px;color:__MUTED__;max-width:340px;line-height:1.45}
  #hint b{color:__INK__;font-weight:500}
  #iterlabel{top:14px;right:16px;padding:8px 14px;font-family:__SERIF__;font-size:18px}
  #iterlabel span{color:__BLUE__}
  #attr{top:96px;left:16px;padding:10px 12px;width:300px;line-height:1.5}
  #attr select{width:100%;background:#0B0D12;color:__INK__;border:1px solid #2A2F3A;
    border-radius:8px;padding:5px 6px;font:inherit;margin:4px 0 6px 0}
  #attr .k{color:__MUTED__;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
  #attr .m{display:flex;gap:14px;font-family:__SERIF__;font-size:16px}
  #attr .m span{color:__MUTED__;font-family:__SANS__;font-size:12px}
  #bar{left:50%;bottom:16px;transform:translateX(-50%);display:flex;gap:10px;align-items:center;
    padding:8px 12px;flex-wrap:wrap;justify-content:center;max-width:calc(100% - 40px)}
  button{background:transparent;color:__INK__;border:1px solid #2A2F3A;border-radius:999px;
    padding:5px 11px;font:inherit;cursor:pointer;transition:all .15s}
  button:hover{border-color:__BLUE__;color:__BLUE__}
  button.on{border-color:__BLUE__;color:#0B0D12;background:__BLUE__}
  button.digit{min-width:30px;padding:5px 6px;font-family:__SERIF__;font-size:15px}
  button.digit.pred{border-color:__GOLD__;color:__GOLD__}
  button.digit.on{background:__GOLD__;color:#0B0D12;border-color:__GOLD__}
  .sep{width:1px;height:22px;background:#2A2F3A}
  input[type=range]{width:150px;accent-color:__BLUE__}
  #card{top:64px;right:16px;padding:12px 14px;width:250px;display:none}
  #card h4{margin:0 0 6px 0;font-family:__SERIF__;font-weight:400;font-size:17px}
  #card .row{display:flex;justify-content:space-between;color:__MUTED__;margin:2px 0}
  #card .row b{color:__INK__;font-weight:500}
  #legend{left:16px;bottom:16px;padding:10px 12px;line-height:1.7;color:__MUTED__}
  .sw{display:inline-block;width:22px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:8px}
  @media (max-width:900px){#legend,#attr{display:none}#hint{max-width:220px}}
</style></head><body>
<div id="wrap">
  <div id="hint" class="panel"><b>Drag</b> to orbit · <b>click a class capsule</b> (or a label below)
    to see who routes to it · <b>click a primary capsule</b> to inspect its votes.</div>
  <div id="attr" class="panel">
    <div class="k">capsule attribution projection (plane between image and grid)</div>
    <select id="method"></select>
    <div id="agree"></div>
  </div>
  <div id="iterlabel" class="panel">routing iteration <span id="itv">__NITER__</span> / __NITER__</div>
  <div id="card" class="panel"></div>
  <div id="legend" class="panel">
    <div><span class="dot" style="background:#2F5BD3"></span>certain capsule (low entropy)</div>
    <div><span class="dot" style="background:__TEAL__"></span>uncertain capsule</div>
    <div><span class="dot" style="background:__GOLD__"></span>predicted class</div>
    <div><span class="sw" style="background:#5B5FC7;height:2px;opacity:.7"></span>weak coupling</div>
    <div><span class="sw" style="background:__RED__;height:5px"></span>strong coupling</div>
  </div>
  <div id="bar" class="panel">
    <button id="play">▶ play routing</button>
    <input id="scrub" type="range" min="0" max="__MAXIT__" step="0.01" value="__MAXIT__">
    <div class="sep"></div>
    <span id="digits"></span>
    <div class="sep"></div>
    <button id="t-image" class="on">input</button>
    <button id="t-infl" class="on">attribution</button>
    <button id="t-labels" class="on">labels</button>
    <button id="cam">view</button>
  </div>
  __PLOT__
</div>
</body></html>"""


def build_html(trace, attr, classes, height=860):
    fig = build_figure(trace, attr, classes)
    payload = build_payload(trace, attr, classes)
    js = (Path(__file__).with_name("hero.js").read_text()
          .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))))
    plot = fig.to_html(full_html=False, include_plotlyjs=True, div_id="hero",
                       post_script=js, default_width="100%", default_height="100%",
                       config={"displaylogo": False, "responsive": True, "displayModeBar": False})
    subs = {"__H__": str(height), "__NITER__": str(payload["nIter"]),
            "__MAXIT__": str(payload["nIter"] - 1), "__SANS__": T.SANS, "__SERIF__": T.SERIF,
            "__INK__": T.INK, "__MUTED__": T.MUTED, "__BLUE__": T.BLUE, "__GOLD__": T.GOLD,
            "__TEAL__": T.TEAL, "__RED__": T.RED}
    html = _TEMPLATE
    for k, v in subs.items():
        html = html.replace(k, v)
    return html.replace("__PLOT__", plot)
