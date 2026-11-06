#!/usr/bin/env python3
"""
CapsNet Routing Explorer

A full-page, 3Blue1Brown-styled tour of dynamic routing-by-agreement in trained
CapsNets (MNIST, Fashion-MNIST, CIFAR-10), with live gradient-based capsule
attribution validated against exact ablation. The 3D routing cube is the
centrepiece; five chapters below it explain what the cube shows.

Run from this folder:  streamlit run capsnet_routing_explorer.py
"""

import base64

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from explorer import analysis as A
from explorer import attribution as AT
from explorer import charts as C
from explorer import data as D
from explorer import hero as H
from explorer import theme as T
from explorer.protocol import SOFTMAX_TOL
from explorer.registry import DATASETS, LAMBDA, M_MINUS, M_PLUS

st.set_page_config(page_title="CapsNet Routing Explorer", page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")
T.register_template()
st.markdown(T.PAGE_CSS, unsafe_allow_html=True)

CHART = dict(use_container_width=True, config={"displaylogo": False, "displayModeBar": False})
GRADIENT_VIEW = ["taylor", "taylor_frozen", "g_A", "g_C",
                 "coupling", "act_x_coupling", "activation", "random"]


def _lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:          # plotly 6 typed array
            arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"]))
            if "shape" in obj:
                arr = arr.reshape([int(n) for n in str(obj["shape"]).split(",")])
            return arr.tolist()
        return {k: _lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_lists(v) for v in obj]
    return obj


def plain(fig):
    """Plotly 6 serializes numpy arrays as base64 'bdata', which the plotly.js
    bundled with Streamlit 1.29 cannot decode; hand it plain lists instead."""
    return _lists(fig.to_dict())


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner=False)
def get_model(key):
    device = D.get_device()
    spec = DATASETS[key]
    model = spec.build_model(device)
    return model, device


@st.cache_resource(show_spinner=False)
def get_dataset(key):
    ds = DATASETS[key].dataset(train=False)
    return ds, DATASETS[key].targets(ds)


@st.cache_data(show_spinner=False, max_entries=8)
def hero_html(key, idx, _trace, _attr):
    return H.build_html(_trace, _attr, DATASETS[key].classes, height=860)


def load_sample(key, idx):
    model, device = get_model(key)
    ds, _ = get_dataset(key)
    x, label = ds[idx]
    x = x.unsqueeze(0).to(device)
    st.session_state["sample"] = {
        "key": key, "idx": idx, "label": int(label), "x": x,
        "trace": D.trace_routing(model, x, DATASETS[key]),
        "attr": D.attributions(model, x),
    }


def new_random_sample(key=None):
    key = key or st.session_state.get("ds_key", "mnist")
    want = st.session_state.get("want_class", "any")
    classes = DATASETS[key].classes
    cls = None if want == "any" or want not in classes else classes.index(want)
    load_sample(key, D.pick_index(get_dataset(key)[1], cls))


TITLE_TO_KEY = {spec.title: k for k, spec in DATASETS.items()}


def on_dataset_change():
    key = TITLE_TO_KEY[st.session_state["ds_title"]]
    st.session_state["ds_key"] = key
    st.session_state["want_class"] = "any"
    new_random_sample(key)


def chip(text, value, cls=""):
    return f'<span class="chip {cls}">{text} <b>{value}</b></span>'


def narrative(html):
    st.markdown(f'<div class="narrative">{html}</div>', unsafe_allow_html=True)


def capsule_picker(prefix, geo, default_cap, key):
    """(gx, gy, type) selectors -> capsule index ([H, W, type] flattening order)."""
    gx0, gy0, t0 = geo.coords(default_cap)
    c1, c2, c3 = st.columns(3)
    gx = c1.number_input("grid x", 0, geo.G - 1, gx0, key=f"{prefix}_gx_{key}")
    gy = c2.number_input("grid y", 0, geo.G - 1, gy0, key=f"{prefix}_gy_{key}")
    t = c3.number_input("type", 0, geo.T - 1, t0, key=f"{prefix}_t_{key}")
    return geo.index(gx, gy, t)


# ------------------------------------------------------------------ state
if "ds_key" not in st.session_state:
    # ?dataset=cifar10&index=42 opens a specific sample (shareable links, figure export)
    q = st.experimental_get_query_params()
    st.session_state["ds_key"] = q.get("dataset", ["mnist"])[0] if q.get("dataset", [""])[0] in DATASETS else "mnist"
    if "index" in q and q["index"][0].isdigit():
        load_sample(st.session_state["ds_key"], int(q["index"][0]))
if "sample" not in st.session_state or st.session_state["sample"]["key"] != st.session_state["ds_key"]:
    with st.spinner("Loading model and a sample…"):
        new_random_sample(st.session_state["ds_key"])

S = st.session_state["sample"]
key, idx, label = S["key"], S["idx"], S["label"]
spec = DATASETS[key]
classes = spec.classes
tr, attr = S["trace"], S["attr"]
geo = tr["geo"]
pred = tr["predicted"]
n_it = len(tr["iterations"])
ITERS = [str(k + 1) for k in range(n_it)]
last = tr["iterations"][-1]
exact = attr["exact"]
top_cap = int(np.argmax(exact))
unit = "digit" if key == "mnist" else "class"
uid = f"{key}_{idx}"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Sample")
    st.selectbox("True class", ["any"] + list(classes), key="want_class")
    st.button("New random sample", use_container_width=True, on_click=new_random_sample)
    st.number_input("…or test-set index", 0, len(get_dataset(key)[1]) - 1, step=1, key="idx_in")
    st.button("Load index", use_container_width=True,
              on_click=lambda: load_sample(st.session_state["ds_key"], int(st.session_state["idx_in"])))
    st.caption(f"{spec.title} test set · checkpoint {spec.checkpoint().name} · {n_it} routing iterations · "
               f"{geo.N:,} primary capsules")

# ------------------------------------------------------------------ top strip
left, mid, right = st.columns([5.0, 1, 3.4], gap="medium")
with left:
    st.markdown('<p class="hero-title">How capsules <em>agree</em></p>'
                f'<p class="hero-sub">Every one of {geo.N:,} primary capsules votes for what it sees. '
                'Routing-by-agreement listens most to the votes that agree with each other. '
                'The gradient of that process tells us, in one backward pass, which capsules the '
                'decision actually rests on.</p>', unsafe_allow_html=True)
with mid:
    st.image(tr["image"], width=86, clamp=True)
with right:
    st.selectbox("dataset", list(TITLE_TO_KEY), index=list(DATASETS).index(key),
                 key="ds_title", on_change=on_dataset_change, label_visibility="collapsed")
    conf = tr["digit_norms"][-1][pred]
    ok = pred == label
    st.markdown(
        '<div>' + chip("true", classes[label], "blue") + chip("predicted", classes[pred], "gold")
        + chip("‖v‖", f"{conf:.2f}") + "</div><div style='margin-top:.5rem'>"
        + f'<span class="chip {"ok" if ok else "bad"}">{"✓ correct" if ok else "✗ misclassified"}</span>'
        + chip("test index", idx) + "</div>", unsafe_allow_html=True)
    st.button("↻  new sample", key="top_new", on_click=new_random_sample)

# ------------------------------------------------------------------ hero
components.html(hero_html(key, idx, tr, attr), height=870, scrolling=False)
st.markdown(
    f'<p class="caption" style="text-align:center">The floor is the {geo.G}×{geo.G} grid of primary '
    f'capsules; each column stacks its {geo.T} capsule types, directly above the patch of the input it '
    f'sees. Dot colour is routing entropy, dot size is activation ‖u‖. The {unit} capsules float above, '
    f'their size is their length ‖v‖. The plane between the image and the grid shows a capsule attribution projection; switch it '
    f'between exact ablation and the gradient methods. Press <b>play routing</b> to watch the couplings '
    f'sharpen over {n_it} iterations.</p>', unsafe_allow_html=True)

st.markdown("&nbsp;")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ⅰ  Pixels → capsules", "Ⅱ  Votes & agreement",
                                        "Ⅲ  The squash", "Ⅳ  Gradients through routing",
                                        "Ⅴ  Output & deep dive"])

# ================================================================== Chapter I
with tab1:
    narrative(
        'A capsule is a small <span class="b">vector</span> rather than a single number. Two '
        f'convolutions turn the {geo.img_size}×{geo.img_size} image into a {geo.G}×{geo.G} grid, and at '
        f'each grid cell sit <span class="t">{geo.T} capsule types</span>, each a {geo.d_in}-dimensional vector. '
        'Its <span class="y">direction</span> encodes <i>how</i> a feature looks; its '
        '<span class="y">length</span> encodes <i>how sure</i> the network is that the feature is there.')
    st.plotly_chart(plain(C.pipeline_figure(tr, classes)), **CHART)
    st.latex(rf"u_i \;=\; \operatorname{{squash}}\!\big(\mathrm{{Conv}}_{{{geo.k2}\times{geo.k2},\;s={geo.s2}}}\,"
             rf"(\mathrm{{ReLU}}(\mathrm{{Conv}}_{{{geo.k1}\times{geo.k1}}}\,x))\big)_i"
             rf"\qquad u_i\in\mathbb{{R}}^{{{geo.d_in}}},\;\; \|u_i\|\in[0,1)")
    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        st.markdown("#### What one grid cell sees")
        a, b = st.columns(2)
        gx = a.slider("grid x", 0, geo.G - 1, geo.coords(top_cap)[0], key=f"rf_gx_{uid}")
        gy = b.slider("grid y", 0, geo.G - 1, geo.coords(top_cap)[1], key=f"rf_gy_{uid}")
        st.plotly_chart(plain(C.receptive_field_figure(tr["image"], geo, gx, gy)), **CHART)
        st.markdown(f'<p class="caption">Each capsule column looks through a {geo.rf}×{geo.rf} window '
                    f'(yellow). Neighbouring windows are {geo.stride} pixels apart, so they overlap by '
                    f'{geo.rf - geo.stride} pixels.</p>',
                    unsafe_allow_html=True)
    with c2:
        col_norms = [tr["u_norm"][geo.index(gx, gy, t)] for t in range(geo.T)]
        st.markdown(f"#### The {geo.T} capsules at ({gx}, {gy})")
        st.plotly_chart(plain(C.bars_figure(col_norms, "activation ‖u‖ of each capsule type",
                                            highlight=int(np.argmax(col_norms)), color=T.TEAL,
                                            xlabel="capsule type")), **CHART)
        st.markdown("#### Strongest conv1 feature maps")
        st.plotly_chart(plain(C.feature_maps_figure(tr["conv1"])), **CHART)
    st.markdown(f"#### All {geo.T} capsule types across the grid")
    u_lo = float(np.quantile(tr["u_norm"], 0.02))
    narrative(f'Each small map is one capsule type, coloured by length ‖u‖, scaled to this sample\'s range '
              f'(98% of its capsules have ‖u‖ ≥ {u_lo:.3f}). Darker cells are where that type is '
              '<i>less</i> sure its feature is present.')
    st.plotly_chart(plain(C.capsule_types_figure(geo, tr["u_norm"])), **CHART)

# ================================================================== Chapter II
with tab2:
    narrative(
        'Every primary capsule <span class="b">i</span> makes a prediction for every '
        f'{unit} capsule <span class="g">j</span> by multiplying its pose by a learned matrix. These are '
        'the <span class="b">votes</span> û<sub>j|i</sub>. Routing starts out listening to everyone '
        'equally, then repeatedly <span class="y">raises the weight</span> of votes that point the same '
        'way as the consensus <span class="y">v<sub>j</sub></span> and lowers the rest.')
    e1, e2, e3 = st.columns(3)
    e1.latex(r"\hat u_{j|i} = W_{ij}\,u_i")
    e2.latex(r"c_{ij} = \frac{e^{b_{ij}}}{\sum_k e^{b_{ik}}},\quad s_j=\sum_i c_{ij}\,\hat u_{j|i}")
    e3.latex(r"v_j=\operatorname{squash}(s_j),\quad b_{ij}\leftarrow b_{ij}+\hat u_{j|i}\!\cdot v_j")

    a, b, c = st.columns([1.2, 1.2, 2.2])
    vdigit = a.selectbox(f"{unit} capsule j  (predicted: {classes[pred]})", list(range(len(classes))),
                         index=pred, key=f"v_digit_{uid}")
    vit = int(b.radio("iteration", ITERS, index=n_it - 1, horizontal=True, key=f"v_it_{uid}")) - 1
    with c:
        st.markdown('<p class="caption" style="margin-top:.4rem">Focus capsule (defaults to the one whose '
                    'removal hurts most)</p>', unsafe_allow_html=True)
        focus = capsule_picker("v", geo, top_cap, uid)

    l, r = st.columns([1.35, 1], gap="large")
    with l:
        st.plotly_chart(plain(C.votes_figure(tr, vdigit, vit, classes, focus_cap=focus)), **CHART)
    with r:
        it_d = tr["iterations"][vit]
        ag = it_d["agreement"][vdigit]
        narrative(
            f'This is {unit} {classes[vdigit]}\'s {geo.d_out}-dimensional space, flattened so the horizontal '
            f'axis points along <span class="y">v<sub>{vdigit}</sub></span>. Each dot is one primary '
            f'capsule\'s vote, and its horizontal position <i>is</i> its agreement û·v̂. Dot size is how '
            f'much routing currently listens to it (c·‖û‖). <span class="r">Red</span> votes agree and '
            f'get amplified next iteration; <span class="b">blue</span> votes disagree and fade.')
        k1, k2 = st.columns(2)
        k1.metric("votes that agree (û·v > 0)", f"{(ag > 0).mean() * 100:.0f}%")
        k2.metric(f"‖v‖ at iteration {vit + 1}", f"{np.linalg.norm(it_d['v'][vdigit]):.3f}")
        gxf, gyf, tf = geo.coords(focus)
        st.markdown(f'<p class="caption">Focus capsule {focus} = grid ({gxf}, {gyf}), type {tf}: '
                    f'c = {it_d["c"][vdigit, focus]:.3f}, û·v = {ag[focus]:+.4f}</p>',
                    unsafe_allow_html=True)

    st.markdown(f"#### One capsule's routing, step by step: capsule {focus}")
    narrative('The logits b start at zero, so the first couplings are uniform. After each pass the '
              'agreement û·v is added to b, and the softmax sharpens toward the classes this capsule '
              'agrees with. That is the whole algorithm.')
    st.plotly_chart(plain(C.routing_steps_figure(tr, focus, pred, classes)), **CHART)
    st.markdown(f"#### Where each {unit}'s evidence comes from")
    narrative('Summing c<sub>ij</sub>·‖û<sub>j|i</sub>‖ over the capsule types gives, for every class, '
              'a map of which parts of the image are routed to it.')
    st.plotly_chart(plain(C.evidence_maps_figure(tr, vit, classes)), **CHART)

# ================================================================== Chapter III
with tab3:
    narrative(
        'A capsule\'s length should behave like a probability. The <span class="y">squash</span> keeps '
        'each vector\'s direction, shrinks short vectors almost to zero, and lets long ones approach but '
        f'never reach length 1. The dots are this sample\'s {unit} capsules: '
        + (f'only the predicted one is pushed past m⁺ = {M_PLUS:g}. '
           if (tr["digit_norms"][-1] >= M_PLUS).sum() == 1 and tr["digit_norms"][-1][pred] >= M_PLUS
           else f'{int((tr["digit_norms"][-1] >= M_PLUS).sum())} of them reach m⁺ = {M_PLUS:g} on this image. ')
        + 'Note how flat the curve is near 1: small changes to a long s barely change ‖v‖.')
    st.latex(r"\operatorname{squash}(s)=\frac{\|s\|^2}{1+\|s\|^2}\;\frac{s}{\|s\|}")
    a, b = st.columns([2, 1])
    s_len = a.slider("length of s", 0.0, 6.0, 1.0, 0.05, key="squash_s")
    sit = int(b.radio("iteration", ITERS, index=n_it - 1, horizontal=True, key=f"sq_it_{uid}")) - 1
    st.plotly_chart(plain(C.squash_figure(s_len, tr, sit, classes)), **CHART)
    v_len = s_len ** 2 / (1 + s_len ** 2)
    k1, k2, k3 = st.columns(3)
    k1.metric("‖s‖", f"{s_len:.2f}")
    k2.metric("‖v‖ = squash", f"{v_len:.3f}")
    k3.metric("shrink factor ‖v‖/‖s‖", f"{(v_len / s_len) if s_len else 0:.3f}")

# ================================================================== Chapter IV
with tab4:
    narrative(
        'Which capsules does the decision rest on? The honest answer is an intervention: remove capsule '
        '<span class="b">i</span>, re-run all of routing, and measure how much the winning capsule '
        'shrinks. That is <span class="y">exact ablation</span>, and it costs one routing pass per capsule. '
        'The <span class="g">gradient</span> gets an estimate in a single backward pass, because '
        'autograd differentiates through every routing iteration: the couplings c depend on u through '
        'the agreement updates, so ∂φ/∂u already "knows" how routing would react.')
    e1, e2, e3 = st.columns(3)
    e1.latex(r"\varphi=\|v_{\hat\jmath}\|,\qquad \Delta_i=\varphi(u)-\varphi(u\,|\,u_i{=}0)")
    e2.latex(r"\Delta_i\;\approx\;\Big\langle \tfrac{\partial\varphi}{\partial u_i},\,u_i\Big\rangle")
    e3.latex(r"\tfrac{\partial\varphi}{\partial u_i}=\underbrace{\tfrac{\partial\varphi}{\partial u_i}\Big|_{c}}_{\text{votes}}"
             r"+\underbrace{\sum_{k}\sum_{j}\tfrac{\partial\varphi}{\partial c_{kj}}\tfrac{\partial c_{kj}}{\partial u_i}}_{\text{routing}}")

    l, r = st.columns([1.25, 1], gap="large")
    with l:
        label_to_key = {C.METHOD_LABELS[k]: k for k in GRADIENT_VIEW}
        m = label_to_key[st.selectbox("method", list(label_to_key), key=f"gm_{uid}", index=0)]
        st.plotly_chart(plain(C.attribution_scatter(attr, m)), **CHART)
    with r:
        st.plotly_chart(plain(C.method_agreement_figure(attr, GRADIENT_VIEW)), **CHART)
        ag_t, ag_f = A.agreement(attr["taylor"], exact), A.agreement(attr["taylor_frozen"], exact)
        narrative(
            f'On this image the Taylor estimate recovers <span class="y">{100 * ag_t["p1"]:.0f}%</span> of '
            f'the true top-1% capsules (ρ = {ag_t["rho"]:.3f}). Freezing the couplings, i.e. ignoring the '
            f'routing term, gives {100 * ag_f["p1"]:.0f}% (ρ = {ag_f["rho"]:.3f}). It takes one backward '
            f'pass instead of {geo.N:,} routing passes.')

    st.markdown("#### Removing capsules in groups")
    if "curves" not in S:
        model, device = get_model(key)
        u = AT.primary_capsules(model, S["x"])
        S["curves"] = {k: AT.deletion_insertion(model, u, pred, attr[k])
                       for k in ("exact", "taylor", "g_A", "g_C", "activation") if k in attr}
    st.plotly_chart(plain(C.deletion_figure(S["curves"], geo.N)), **CHART)
    fr, de, _ = S["curves"]["exact"]
    part = de[fr < 1]
    i_min = int(np.argmin(part))
    reb = float(part[i_min:].max() - part[i_min])
    narrative(
        'Removing capsules in groups, with routing run again on the capsules that are left, is a different '
        'question from removing them one at a time. '
        + (f'On this image, removing the top-ranked capsules first drives φ down to {part[i_min]:.2f}·φ₀, '
           f'and removing more lets it <span class="y">rise again by {reb:.2f}</span>. ' if reb > 0.05 else
           f'On this image φ falls to {part[i_min]:.2f}·φ₀ without a marked rebound. ')
        + 'Compare how the rankings by sensitivity magnitude (‖∂φ/∂u‖, |∂φ/∂c|) behave over the same budgets.')

# ================================================================== Chapter V
with tab5:
    narrative(
        'The prediction is simply the <span class="g">longest</span> class capsule. Training pushes the '
        f'correct capsule above <span style="color:{T.GREEN}">{M_PLUS:g}</span> and all others below '
        f'<span class="r">{M_MINUS:g}</span> (margin loss, λ = {LAMBDA:g}). A decoder, trained on the '
        f'{geo.d_out} numbers of the correct class capsule, redraws the input; here it reads the predicted one.')
    st.latex(rf"L_k = T_k\,\max(0,\,{M_PLUS:g}-\|v_k\|)^2+\lambda\,(1-T_k)\,\max(0,\,\|v_k\|-{M_MINUS:g})^2")
    l, r = st.columns([1.2, 1], gap="large")
    l.plotly_chart(plain(C.digit_lengths_figure(tr["digit_norms"][-1], pred, label, classes)), **CHART)
    if tr["reconstruction"] is not None:
        r.plotly_chart(plain(C.image_pair_figure(tr["image"], tr["reconstruction"])), **CHART)
        r.markdown('<p class="caption" style="text-align:center">Reconstruction from the predicted '
                   'class capsule only (the other capsules are masked; in training, the correct class capsule was kept).</p>',
                   unsafe_allow_html=True)
    else:
        xin = S["x"].detach().cpu().numpy()
        outside = float(((xin < 0) | (xin > 1)).mean())
        r.markdown('<div class="card"><div class="k">reconstruction</div><p class="caption">Not shown for '
                   f'{spec.title}: its decoder ends in a sigmoid (range [0, 1]) but was trained against the '
                   f'normalized input, and {100 * outside:.0f}% of this image\'s normalized pixel values lie '
                   'outside [0, 1], so the reconstruction cannot be compared with the input.</p></div>',
                   unsafe_allow_html=True)

    st.markdown("#### Which capsules mattered?")
    l, r = st.columns([1, 1.3], gap="large")
    l.plotly_chart(plain(C.attribution_map_figure(A.grid_sum(geo, exact),
                                                  "Exact ablation, summed over capsule types")), **CHART)
    rows = []
    for rank, ci in enumerate(np.argsort(-exact)[:10], 1):
        gxi, gyi, ti = geo.coords(ci)
        rows.append({"rank": rank, "capsule": int(ci), "grid (x, y)": f"({gxi}, {gyi})", "type": ti,
                     "exact Δ": float(exact[ci]), "Taylor": float(attr["taylor"][ci]),
                     "coupling → pred": round(float(last["c"][pred, ci]), 4),
                     "‖u‖": round(float(tr["u_norm"][ci]), 4)})
    r.dataframe(pd.DataFrame(rows).style.format({"exact Δ": "{:.2e}", "Taylor": "{:.2e}"}),
                use_container_width=True, hide_index=True, height=390)

    st.markdown("#### Capsule deep dive")
    cap = capsule_picker("dd", geo, top_cap, uid)
    info = A.get_capsule_info(geo, cap, last["c"], tr["u_norm"], tr["u"])
    gxc, gyc = info["grid_position"]
    votes = info["votes"]
    ent = float(A.calculate_entropy(last["c"])[cap])
    m = st.columns(5)
    for col, (k, v) in zip(m, [("capsule", cap), ("activation ‖u‖", f"{info['activation_norm']:.3f}"),
                               ("exact Δ · rank", f"{exact[cap]:.1e} · {A.rank_of(exact, cap)}"),
                               ("Taylor estimate · rank", f"{attr['taylor'][cap]:.1e} · {A.rank_of(attr['taylor'], cap)}"),
                               ("routing entropy", f"{ent:.2f} bits")]):
        col.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("&nbsp;")
    a, b, c = st.columns(3, gap="medium")
    a.plotly_chart(plain(C.receptive_field_figure(tr["image"], geo, gxc, gyc,
                                                  title=f"receptive field of grid ({gxc}, {gyc})")), **CHART)
    b.plotly_chart(plain(C.bars_figure(votes, "coupling c<sub>ij</sub> to each class", highlight=pred,
                                       xlabel="class", x=[n[:6] for n in classes])), **CHART)
    c.plotly_chart(plain(C.bars_figure(info["activation_vector"], f"{geo.d_in}D pose vector u<sub>i</sub>",
                                       color=T.TEAL, xlabel="dimension")), **CHART)
    st.plotly_chart(plain(C.evolution_figure(tr, cap, pred, classes)), **CHART)

    with st.expander("Coupling distribution & softmax sanity check"):
        hd = st.selectbox("class", list(range(len(classes))), index=pred, key=f"hist_d_{uid}")
        st.plotly_chart(plain(C.coupling_hist_figure(last["c"][hd], classes[hd], geo.K)), **CHART)
        dev = float(np.max(np.abs(last["c"].sum(axis=0) - 1.0)))
        st.markdown(f"Σ<sub>j</sub> c<sub>ij</sub> deviates from 1 by at most **{dev:.2e}** "
                    f"{'✅ softmax over classes holds' if dev < SOFTMAX_TOL else '⚠️'}", unsafe_allow_html=True)
