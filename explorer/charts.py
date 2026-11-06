"""2D figures for the explainer chapters (all use the '3b1b' Plotly template)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analysis as A
from . import theme as T
from .data import squash_np
from .protocol import TOPK_FRACS
from .registry import M_MINUS, M_PLUS


def fmt_pct(frac):
    return f"{100 * frac:g}%"


TOP_A = fmt_pct(TOPK_FRACS[0])          # "1%"


# ===================================================================== helpers
def _img_axes(fig, row=None, col=None, reverse=True):
    kw = dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks="")
    if row is None:
        fig.update_xaxes(**kw, constrain="domain")
        fig.update_yaxes(**kw, autorange="reversed" if reverse else True, scaleanchor="x")
    else:
        fig.update_xaxes(**kw, row=row, col=col)
        fig.update_yaxes(**kw, autorange="reversed" if reverse else True, row=row, col=col)


def image_trace(img, **kw):
    """Greyscale -> Heatmap, RGB -> Image. img in [0, 1]."""
    img = np.asarray(img)
    if img.ndim == 3:
        return go.Image(z=(np.clip(img, 0, 1) * 255).astype(np.uint8), hoverinfo="skip", **kw)
    return go.Heatmap(z=img, colorscale=T.SCALE_INK, zmin=0, zmax=1, showscale=False,
                      hovertemplate="%{z:.2f}<extra></extra>", **kw)


def _colors(pred, base=T.BLUE, K=10):
    return [T.GOLD if i == pred else base for i in range(K)]


def _short(classes, n=9):
    return [c if len(c) <= n else c[:n - 1] + "…" for c in classes]


MAGNITUDE_METHODS = ("taylor", "taylor_frozen")
METHOD_LABELS = {
    "taylor": "Taylor ⟨∂φ/∂u, u⟩", "taylor_frozen": "Taylor, routing frozen",
    "g_A": "‖∂φ/∂u‖", "g_C": "|∂φ/∂c|", "coupling": "coupling c", "act_x_coupling": "‖u‖·c",
    "activation": "activation ‖u‖", "random": "random", "exact": "exact ablation",
}
METHOD_COLORS = {
    "exact": T.INK, "taylor": T.YELLOW, "taylor_frozen": T.GOLD, "g_A": T.BLUE, "g_C": T.TEAL,
    "coupling": T.RED, "act_x_coupling": "#C98B8B",
    "activation": T.GREY, "random": "#555555",
}


# ===================================================================== Ch. I
def pipeline_figure(trace, classes):
    """Pixels -> conv1 -> primary capsules -> class capsules, as stacked planes."""
    geo = trace["geo"]
    img = np.asarray(trace["image"])
    conv = trace["conv1"]
    u_grid = geo.to_grid(trace["u_norm"])
    norms = trace["digit_norms"][-1]
    pred = trace["predicted"]
    top_maps = np.argsort(conv.mean(axis=(1, 2)))[::-1][:6]
    top_types = np.argsort(u_grid.var(axis=(1, 2)))[::-1][:6]
    lo = float(np.quantile(trace["u_norm"], 0.02))

    fig = go.Figure()

    def plane(z, x0, y0, w, cs, zmax=None, opacity=1.0):
        n = z.shape[0]
        fig.add_trace(go.Heatmap(z=z, x0=x0, dx=w / n, y0=y0, dy=w / n, colorscale=cs, zmin=0,
                                 zmax=zmax if zmax is not None else max(z.max(), 1e-6),
                                 showscale=False, opacity=opacity, hoverinfo="skip"))

    n = img.shape[0]
    fig.add_trace(image_trace(img[::-1], x0=0, dx=10 / n, y0=0, dy=10 / n))
    for k, ch in enumerate(top_maps[::-1]):
        off = (len(top_maps) - 1 - k) * 0.9
        plane(conv[ch][::-1], 15 + off, off, 8, T.SCALE_BLUE_TEAL, opacity=0.95)
    for k, t in enumerate(top_types[::-1]):
        off = (len(top_types) - 1 - k) * 0.9
        plane(np.clip(u_grid[t][::-1] - lo, 0, None), 31 + off, 1 + off, 6, T.SCALE_HEAT,
              zmax=1.0 - lo, opacity=0.95)
    K = len(norms)
    ys = np.linspace(9.5, 0.5, K)
    fig.add_trace(go.Bar(x=norms * 7, y=ys, base=49, orientation="h", width=0.7,
                         marker_color=_colors(pred, T.GREY, K), hoverinfo="skip"))
    for i, y in enumerate(ys):
        fig.add_annotation(x=48.6, y=y, text=_short(classes, 8)[i], showarrow=False, xanchor="right",
                           font=dict(family=T.SERIF, size=12, color=T.GOLD if i == pred else T.MUTED))
    H1 = conv.shape[-1]
    labels = [(5, f"input<br><span style='color:{T.MUTED}'>{n} × {n} pixels</span>"),
              (21.5, f"conv1 + ReLU<br><span style='color:{T.MUTED}'>{H1} × {H1} × {conv.shape[0]}</span>"),
              (36.5, f"primary capsules<br><span style='color:{T.MUTED}'>{geo.G} × {geo.G} × {geo.T}, "
                     f"each {geo.d_in}D</span>"),
              (52.5, f"class capsules<br><span style='color:{T.MUTED}'>{K}, each {geo.d_out}D<br>length = ‖v‖</span>")]
    for x, text in labels:
        fig.add_annotation(x=x, y=-1.6, text=text, showarrow=False, yanchor="top",
                           font=dict(family=T.SERIF, size=15, color=T.INK))
    conv_lab = f"{geo.k1}×{geo.k1} conv" + (f", stride {geo.s1}" if geo.s1 > 1 else "")
    prim_lab = f"{geo.k2}×{geo.k2} conv" + (f"<br>stride {geo.s2}" if geo.s2 > 1 else "") + "<br>→ squash"
    for x0, x1, lab in [(10.6, 14.4, conv_lab), (28.2, 30.4, prim_lab)]:
        fig.add_annotation(x=x1, y=5, ax=x0, ay=5, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                           arrowcolor=T.MUTED, text="")
        fig.add_annotation(x=x1, y=6.4, text=lab, showarrow=False, xanchor="right", align="right",
                           font=dict(size=11, color=T.MUTED))
    # the routing arrow starts after the primary-capsule stack
    fig.add_annotation(x=45.0, y=5, ax=42.4, ay=5, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
                       arrowcolor=T.YELLOW, text="")
    fig.add_annotation(x=43.7, y=6.3, text="routing", showarrow=False,
                       font=dict(family=T.SERIF, size=14, color=T.YELLOW))
    fig.update_xaxes(visible=False, range=[-1, 58])
    fig.update_yaxes(visible=False, range=[-5, 11], scaleanchor="x")
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, bargap=0)
    return fig


def receptive_field_figure(image, geo, gx, gy, title=None):
    x0, x1, y0, y1 = geo.receptive_field(gx, gy)
    S = geo.img_size
    fig = go.Figure(image_trace(image))
    for (a, b, c, d) in [(-0.5, S - 0.5, -0.5, y0 - 0.5), (-0.5, S - 0.5, y1 - 0.5, S - 0.5),
                         (-0.5, x0 - 0.5, y0 - 0.5, y1 - 0.5), (x1 - 0.5, S - 0.5, y0 - 0.5, y1 - 0.5)]:
        if b > a and d > c:
            fig.add_shape(type="rect", x0=a, x1=b, y0=c, y1=d, line_width=0,
                          fillcolor="rgba(11,13,18,0.62)")
    fig.add_shape(type="rect", x0=x0 - 0.5, x1=x1 - 0.5, y0=y0 - 0.5, y1=y1 - 0.5,
                  line=dict(color=T.YELLOW, width=2))
    cx = [geo.receptive_field(i, j)[0] + 8 for j in range(geo.G) for i in range(geo.G)]
    cy = [geo.receptive_field(i, j)[2] + 8 for j in range(geo.G) for i in range(geo.G)]
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", hoverinfo="skip",
                             marker=dict(size=5, color=T.BLUE, opacity=0.6)))
    fcx, fcy = geo.rf_center(gx, gy)
    fig.add_trace(go.Scatter(x=[fcx], y=[fcy], mode="markers", hoverinfo="skip",
                             marker=dict(size=11, color=T.YELLOW, line=dict(width=1, color=T.BG))))
    _img_axes(fig)
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=36 if title else 0, b=0),
                      showlegend=False, title=title)
    return fig


def feature_maps_figure(conv1, k=8):
    top = np.argsort(conv1.mean(axis=(1, 2)))[::-1][:k]
    fig = make_subplots(rows=2, cols=k // 2, horizontal_spacing=0.02, vertical_spacing=0.08,
                        subplot_titles=[f"channel {c}" for c in top])
    for n, ch in enumerate(top):
        r, c = divmod(n, k // 2)
        fig.add_trace(go.Heatmap(z=conv1[ch], colorscale=T.SCALE_BLUE_TEAL, showscale=False,
                                 hovertemplate="%{z:.2f}<extra></extra>"), row=r + 1, col=c + 1)
        _img_axes(fig, r + 1, c + 1)
    fig.update_annotations(font=dict(size=11, color=T.MUTED))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=24, b=0))
    return fig


def capsule_types_figure(geo, u_norm):
    """Small multiples: length of every capsule type over the grid (data-range scaled)."""
    grid = geo.to_grid(u_norm)
    lo = float(np.quantile(u_norm, 0.02))
    cols = 8
    rows = int(np.ceil(geo.T / cols))
    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.012, vertical_spacing=0.04,
                        subplot_titles=[f"type {t}" for t in range(geo.T)])
    for t in range(geo.T):
        r, c = divmod(t, cols)
        fig.add_trace(go.Heatmap(z=grid[t], zmin=lo, zmax=1, colorscale=T.SCALE_HEAT,
                                 showscale=(t == 0),
                                 colorbar=dict(title=dict(text="‖u‖", font=dict(family=T.SERIF)),
                                               len=0.6, thickness=10, outlinewidth=0),
                                 hovertemplate=f"type {t}" + " · grid (%{x}, %{y})<br>‖u‖ = %{z:.3f}<extra></extra>"),
                      row=r + 1, col=c + 1)
        _img_axes(fig, r + 1, c + 1)
    fig.update_annotations(font=dict(size=10, color=T.MUTED))
    fig.update_layout(height=118 * rows, margin=dict(l=0, r=0, t=22, b=0))
    return fig


# ===================================================================== Ch. II
def votes_figure(trace, digit, it, classes, focus_cap=None, n_arrows=14):
    """
    Class j's output space (d_out dimensions), flattened onto a plane whose x-axis is the
    direction of v_j: a vote's x-coordinate is its agreement û·v̂.
    """
    u_hat = trace["u_hat"][digit]
    itd = trace["iterations"][it]
    v, s, c = itd["v"][digit], itd["s"][digit], itd["c"][digit]
    e1 = v / (np.linalg.norm(v) + 1e-12)
    resid = u_hat - np.outer(u_hat @ e1, e1)
    _, _, vt = np.linalg.svd(resid, full_matrices=False)
    e2 = vt[0]
    P = lambda X: np.stack([X @ e1, X @ e2], axis=-1)
    pts = P(u_hat)
    agree = itd["agreement"][digit]
    lim = np.abs(agree).max() + 1e-9
    N = u_hat.shape[0]

    fig = go.Figure()
    weight = c * np.linalg.norm(u_hat, axis=1)
    fig.add_trace(go.Scatter(
        x=pts[:, 0], y=pts[:, 1], mode="markers", name="votes û",
        marker=dict(size=3 + 14 * weight / (weight.max() + 1e-12), color=agree,
                    colorscale=T.SCALE_DIVERGING, cmin=-lim, cmax=lim, opacity=0.75,
                    colorbar=dict(title=dict(text="û · v", font=dict(family=T.SERIF)),
                                  thickness=10, outlinewidth=0, len=0.7)),
        customdata=np.stack([np.arange(N), c, agree], axis=1),
        hovertemplate="capsule %{customdata[0]:.0f}<br>c = %{customdata[1]:.3f}"
                      "<br>û·v = %{customdata[2]:.4f}<extra></extra>"))
    for i in np.argsort(weight)[::-1][:n_arrows]:
        fig.add_annotation(x=pts[i, 0], y=pts[i, 1], ax=0, ay=0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1,
                           arrowsize=0.8, arrowcolor="rgba(88,196,221,0.55)", text="")
    if focus_cap is not None:
        fx, fy = pts[focus_cap]
        fig.add_annotation(x=fx, y=fy, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=2.5, arrowcolor=T.TEAL, text="")
        fig.add_annotation(x=fx, y=fy, text=f"û<sub>{digit}|{focus_cap}</sub>", showarrow=False,
                           yshift=12, font=dict(family=T.SERIF, size=14, color=T.TEAL))
    for vec, color, lab, w in [(P(s[None])[0], T.GREY, "s", 2), (P(v[None])[0], T.YELLOW, "v", 4)]:
        fig.add_annotation(x=vec[0], y=vec[1], ax=0, ay=0, xref="x", yref="y", axref="x",
                           ayref="y", showarrow=True, arrowhead=2, arrowwidth=w,
                           arrowcolor=color, text="")
        fig.add_annotation(x=vec[0], y=vec[1], text=f"<b>{lab}</b><sub>{digit}</sub>",
                           showarrow=False, xshift=14, yshift=8,
                           font=dict(family=T.SERIF, size=18, color=color))
    xspan = max(np.abs(pts[:, 0]).max(), np.linalg.norm(v), 1e-3) * 1.15
    yspan = max(np.abs(pts[:, 1]).max(), 1e-3) * 1.3
    fig.update_xaxes(range=[-xspan, xspan], title="along v̂  (agreement direction)", zeroline=True,
                     tickfont=dict(size=17), title_font=dict(size=18))
    fig.update_yaxes(range=[-yspan, yspan], title="orthogonal (stretched)", zeroline=True,
                     tickfont=dict(size=17), title_font=dict(size=18))
    fig.update_layout(height=520, showlegend=False, font=dict(size=19),
                      title=f"Votes for class {classes[digit]} · iteration {it + 1}")
    return fig


def routing_steps_figure(trace, cap, pred, classes):
    its = trace["iterations"]
    K = its[0]["c"].shape[0]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=["logits  b<sub>ij</sub>", "couplings  c<sub>ij</sub> = softmax(b)"])
    shades = ["rgba(88,196,221,0.30)", "rgba(88,196,221,0.6)", T.BLUE]
    x = _short(classes)
    for k, it in enumerate(its):
        name = f"iteration {k + 1}"
        fig.add_trace(go.Bar(x=x, y=it["b"][:, cap], name=name, marker_color=shades[k % 3],
                             legendgroup=name), 1, 1)
        fig.add_trace(go.Bar(x=x, y=it["c"][:, cap], name=name, showlegend=False,
                             marker_color=shades[k % 3], legendgroup=name), 1, 2)
    fig.add_hline(y=1 / K, line=dict(color=T.MUTED, dash="dot", width=1), row=1, col=2,
                  annotation_text=f"uniform 1/{K}", annotation_font_color=T.MUTED)
    for col in (1, 2):
        fig.add_vrect(x0=pred - 0.5, x1=pred + 0.5, fillcolor="rgba(240,172,95,0.10)",
                      line_width=0, row=1, col=col)
    fig.update_layout(height=330, barmode="group", bargap=0.25,
                      legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"))
    fig.update_annotations(font=dict(family=T.SERIF, size=15, color=T.INK))
    return fig


def evidence_maps_figure(trace, it, classes):
    """Where each class's evidence comes from: Σ_types c_ij ‖û_{j|i}‖ over the grid."""
    geo = trace["geo"]
    itd = trace["iterations"][it]
    ev = itd["c"] * np.linalg.norm(trace["u_hat"], axis=-1)
    pred = trace["predicted"]
    K = ev.shape[0]
    grids = [A.grid_sum(geo, ev[j]) for j in range(K)]
    vmax = max(g.max() for g in grids)
    ncols = int(np.ceil(K / 2))
    fig = make_subplots(rows=2, cols=ncols, horizontal_spacing=0.02, vertical_spacing=0.1,
                        subplot_titles=[f"<b>{n}</b>" if j == pred else n
                                        for j, n in enumerate(_short(classes, 11))])
    for j, g in enumerate(grids):
        r, c = divmod(j, ncols)
        fig.add_trace(go.Heatmap(z=g, zmin=0, zmax=vmax, colorscale=T.SCALE_HEAT,
                                 showscale=(j == 0), colorbar=dict(thickness=10, outlinewidth=0, len=0.7),
                                 hovertemplate=f"{classes[j]}" + " · grid (%{x}, %{y})<br>%{z:.3f}<extra></extra>"),
                      row=r + 1, col=c + 1)
        _img_axes(fig, r + 1, c + 1)
    for k, ann in enumerate(fig.layout.annotations):
        ann.font = dict(family=T.SERIF, size=15, color=T.GOLD if k == pred else T.MUTED)
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=26, b=0))
    return fig


# ===================================================================== Ch. III
def squash_figure(s_len, trace, it, classes):
    xs = np.linspace(0, 8, 400)
    ys = xs ** 2 / (1 + xs ** 2)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.08,
                        subplot_titles=["‖v‖ = ‖s‖² / (1 + ‖s‖²)", "one vector, before and after"])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=T.BLUE, width=3),
                             hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=np.minimum(xs, 1.4), mode="lines", hoverinfo="skip",
                             line=dict(color=T.GREY, width=1, dash="dot")), 1, 1)
    fig.add_hline(y=1, line=dict(color=T.MUTED, dash="dash", width=1), row=1, col=1)
    itd = trace["iterations"][it]
    s_n = np.linalg.norm(itd["s"], axis=-1)
    v_n = np.linalg.norm(itd["v"], axis=-1)
    pred = trace["predicted"]
    K = len(s_n)
    names = _short(classes, 6)                 # label only the predicted class; the rest on hover
    fig.add_trace(go.Scatter(x=s_n, y=v_n, mode="markers+text", customdata=names,
                             text=[n if i == pred else "" for i, n in enumerate(names)],
                             textposition="top center", textfont=dict(family=T.SERIF, color=T.INK),
                             marker=dict(size=[14 if i == pred else 9 for i in range(K)],
                                         color=_colors(pred, T.TEAL, K),
                                         line=dict(width=1, color=T.BG)),
                             hovertemplate="%{customdata}<br>‖s‖ = %{x:.3f}<br>‖v‖ = %{y:.3f}<extra></extra>"),
                  1, 1)
    v_len = s_len ** 2 / (1 + s_len ** 2)
    fig.add_trace(go.Scatter(x=[s_len], y=[v_len], mode="markers", hoverinfo="skip",
                             marker=dict(size=16, color=T.YELLOW, symbol="circle-open",
                                         line=dict(width=3))), 1, 1)
    fig.add_shape(type="line", x0=s_len, x1=s_len, y0=0, y1=v_len, row=1, col=1,
                  line=dict(color=T.YELLOW, width=1, dash="dot"))
    fig.update_xaxes(title="‖s‖  (length before squash)", range=[0, 8], row=1, col=1)
    fig.update_yaxes(title="‖v‖  (length after)", range=[0, 1.45], row=1, col=1)
    ang = np.deg2rad(32)
    sv = np.array([np.cos(ang), np.sin(ang)]) * s_len
    vv = squash_np(sv)
    R = max(1.2, s_len * 1.1)
    th = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", hoverinfo="skip",
                             line=dict(color=T.MUTED, width=1, dash="dot")), 1, 2)
    for vec, color, lab, w in [(sv, T.GREY, "s", 2), (vv, T.YELLOW, "v", 4)]:
        fig.add_annotation(x=vec[0], y=vec[1], ax=0, ay=0, xref="x2", yref="y2", axref="x2",
                           ayref="y2", showarrow=True, arrowhead=2, arrowwidth=w,
                           arrowcolor=color, text="")
        fig.add_annotation(x=vec[0], y=vec[1], xref="x2", yref="y2", text=f"<b>{lab}</b>",
                           showarrow=False, xshift=12, yshift=10,
                           font=dict(family=T.SERIF, size=18, color=color))
    fig.update_xaxes(range=[-0.3, R], row=1, col=2, zeroline=True, showticklabels=False)
    fig.update_yaxes(range=[-0.3, R], row=1, col=2, zeroline=True, showticklabels=False,
                     scaleanchor="x2")
    for text in ("‖v‖ = ‖s‖² / (1 + ‖s‖²)", "one vector, before and after"):
        fig.update_annotations(selector=dict(text=text), font=dict(family=T.SERIF, size=16, color=T.INK))
    fig.update_layout(height=420, showlegend=False)
    return fig


# ===================================================================== Ch. IV
def digit_lengths_figure(norms, pred, label, classes):
    fig = go.Figure(go.Bar(x=_short(classes), y=norms, marker_color=_colors(pred, T.BLUE, len(norms)),
                           text=[f"{n:.2f}" for n in norms], textposition="outside",
                           textfont=dict(color=T.MUTED),
                           hovertemplate="%{x}: ‖v‖ = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=M_PLUS, line=dict(color=T.GREEN, dash="dash", width=1),
                  annotation_text=f"m⁺ = {M_PLUS:g}", annotation_font_color=T.GREEN)
    fig.add_hline(y=M_MINUS, line=dict(color=T.RED, dash="dash", width=1),
                  annotation_text=f"m⁻ = {M_MINUS:g}", annotation_font_color=T.RED)
    fig.update_yaxes(range=[0, 1.1], title="length ‖v<sub>j</sub>‖")
    fig.update_layout(height=340, title=f"Class-capsule lengths · true label: {classes[label]}")
    return fig


def image_pair_figure(image, recon):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04,
                        subplot_titles=["input", "reconstruction from the class capsules"])
    for k, z in enumerate([image, recon]):
        fig.add_trace(image_trace(z), 1, k + 1)
        _img_axes(fig, 1, k + 1)
    fig.update_yaxes(scaleanchor="x", row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", row=1, col=2)
    fig.update_annotations(font=dict(family=T.SERIF, size=14, color=T.MUTED))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def attribution_map_figure(grid, title):
    fig = go.Figure(go.Heatmap(z=grid, colorscale=T.SCALE_HEAT, xgap=2, ygap=2,
                               colorbar=dict(thickness=10, outlinewidth=0, exponentformat="e"),
                               hovertemplate="grid (%{x}, %{y})<br>%{z:.5f}<extra></extra>"))
    fig.update_xaxes(dtick=1, title="grid x", showgrid=False, constrain="domain")
    fig.update_yaxes(dtick=1, title="grid y", autorange="reversed", scaleanchor="x",
                     showgrid=False, constrain="domain")
    fig.update_layout(height=340, title=title)
    return fig


def coupling_hist_figure(couplings, name, K):
    fig = go.Figure(go.Histogram(x=couplings, nbinsx=40, histnorm="probability",
                                 marker_color=T.BLUE, opacity=0.8))
    fig.add_vline(x=1 / K, line=dict(color=T.GOLD, dash="dash"), annotation_text=f"uniform 1/{K}",
                  annotation_font_color=T.GOLD)
    fig.update_layout(height=300, bargap=0.05, showlegend=False,
                      title=f"Couplings of all {len(couplings)} capsules to {name}",
                      xaxis_title="c<sub>ij</sub>", yaxis_title="fraction")
    return fig


def bars_figure(values, title, highlight=None, color=T.BLUE, xlabel="", x=None):
    cols = [T.GOLD if i == highlight else color for i in range(len(values))]
    fig = go.Figure(go.Bar(x=x if x is not None else list(range(len(values))), y=values,
                           marker_color=cols, hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    if x is None:
        fig.update_xaxes(dtick=1)
    fig.update_xaxes(title=xlabel)
    fig.update_layout(height=260, title=title)
    return fig


def evolution_figure(trace, cap, pred, classes):
    its = trace["iterations"]
    x = [k + 1 for k in range(len(its))]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["coupling to every class", "routing entropy (bits)"])
    for d in range(its[0]["c"].shape[0]):
        fig.add_trace(go.Scatter(x=x, y=[it["c"][d, cap] for it in its], mode="lines+markers",
                                 name=classes[d],
                                 line=dict(color=T.GOLD if d == pred else "rgba(88,196,221,0.45)",
                                           width=3 if d == pred else 1.2)), 1, 1)
    ent = [A.calculate_entropy(it["c"])[cap] for it in its]
    fig.add_trace(go.Scatter(x=x, y=ent, mode="lines+markers", line=dict(color=T.PURPLE, width=2.5),
                             fill="tozeroy", fillcolor="rgba(154,114,172,0.15)", showlegend=False), 1, 2)
    fig.update_xaxes(dtick=1, title="iteration")
    fig.update_annotations(font=dict(family=T.SERIF, size=15, color=T.INK))
    fig.update_layout(height=320, showlegend=False)
    return fig


# ===================================================================== Ch. V
def attribution_scatter(attr, method):
    """Exact ablation Δ vs. a method's score; the true top-TOPK_FRACS[0] set is highlighted."""
    ex = attr["exact"]
    y = attr[method]
    k = max(1, int(np.ceil(TOPK_FRACS[0] * len(ex))))
    top = np.zeros(len(ex), bool)
    top[np.argsort(-ex)[:k]] = True
    fig = go.Figure()
    for mask, color, name, size in [(~top, "rgba(88,196,221,0.35)", "other capsules", 4),
                                    (top, T.GOLD, f"true top {TOP_A}", 8)]:
        fig.add_trace(go.Scatter(x=ex[mask], y=y[mask], mode="markers", name=name,
                                 marker=dict(size=size, color=color),
                                 customdata=np.flatnonzero(mask),
                                 hovertemplate="capsule %{customdata}<br>Δ = %{x:.5f}<br>score = %{y:.5f}<extra></extra>"))
    if method in MAGNITUDE_METHODS:
        lo, hi = float(min(ex.min(), y.min())), float(max(ex.max(), y.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                                 line=dict(color=T.MUTED, dash="dot", width=1), hoverinfo="skip"))
    ag = A.agreement(y, ex)
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25), font=dict(size=16),
                      title=f"{METHOD_LABELS[method]} vs exact ablation<br>"
                            f"<span style='font-size:13px'>ρ = {ag['rho']:.3f} · "
                            f"top-{TOP_A} recovered {100 * ag['p1']:.0f}%</span>",
                      margin=dict(t=78),
                      xaxis_title="exact Δ<sub>i</sub> = φ − φ(u<sub>i</sub> = 0)",
                      yaxis_title="estimate" if method in MAGNITUDE_METHODS else "score")
    return fig


INSIDE_LABEL_MIN = 0.35        # layout: bars shorter than this share of the axis get their label outside


def method_agreement_figure(attr, methods):
    rows = [(m, A.agreement(attr[m], attr["exact"])) for m in methods if m in attr]
    rows.sort(key=lambda r: r[1]["p1"])
    fig = go.Figure(go.Bar(
        y=[METHOD_LABELS[m] for m, _ in rows], x=[100 * a["p1"] for _, a in rows], orientation="h",
        marker_color=[METHOD_COLORS[m] for m, _ in rows],
        text=[f"{100 * a['p1']:.0f}%  ·  ρ {a['rho']:.2f}".replace("-0.00", "0.00") for _, a in rows],
        textposition=["inside" if a["p1"] > INSIDE_LABEL_MIN else "outside" for _, a in rows],
        insidetextfont=dict(color=T.BG), outsidetextfont=dict(color=T.MUTED), cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(range=[0, 100], title=f"true top-{TOP_A} capsules recovered (%)")
    fig.update_layout(height=80 + 34 * len(rows), title="This image: how well each method<br>finds the capsules that matter",
                      margin=dict(t=78, r=36), font=dict(size=15))
    return fig


def deletion_title(curves):
    """Describe THIS image's exact-ablation deletion curve rather than assert a pattern."""
    frac, dele, _ = curves["exact"]
    part = dele[frac < 1]
    i = int(np.argmin(part))
    rebound = float(part[i:].max() - part[i])
    if rebound > 0.05:
        return f"Removing capsules in groups: φ falls to {part[i]:.2f}·φ₀, then rises by {rebound:.2f}"
    return f"Removing capsules in groups: φ falls to {part[i]:.2f}·φ₀ (no rebound on this image)"


def deletion_figure(curves, n_caps):
    """φ/φ0 as the top-ranked capsules are removed cumulatively (log x)."""
    fig = go.Figure()
    for m, (frac, dele, _) in curves.items():
        k = np.maximum(frac * n_caps, 0.6)
        fig.add_trace(go.Scatter(x=k, y=dele, mode="lines+markers", name=METHOD_LABELS[m],
                                 line=dict(color=METHOD_COLORS[m], width=3 if m in ("exact", "taylor") else 1.8,
                                           dash="dot" if m == "exact" else None),
                                 marker=dict(size=4),
                                 hovertemplate="%{x:.0f} removed → φ/φ0 = %{y:.3f}<extra></extra>"))
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000) if t <= n_caps]
    fig.update_xaxes(type="log", tickvals=ticks, ticktext=[str(t) for t in ticks],
                     title="capsules removed (ranked by each method, cumulative)")
    fig.update_yaxes(title="φ / φ<sub>0</sub>", range=[0, 1.15])
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.28),
                      title=deletion_title(curves))
    return fig
