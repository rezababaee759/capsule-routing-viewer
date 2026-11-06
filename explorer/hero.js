(function () {
function init() {
  const D = __PAYLOAD__;
  const gd = document.getElementById('hero');
  const TR = { primary: 0, digits: 1, image: 2, infl: 3, grid: 4, labels: 5, focus: 6, bin0: 7 };
  const NB = D.bins.length, LAST = D.nIter - 1, NCAP = D.primaryPos.length, K = D.classes.length;
  const C = D.colors;
  const state = { t: LAST, digit: null, cap: null, playing: false, cam: 0 };

  // ------------------------------------------------------------ interpolation
  function split(t) {
    const i = Math.min(Math.floor(t), LAST), j = Math.min(i + 1, LAST);
    return [i, j, t - i];
  }
  function mix(a, b, f) {
    if (f < 1e-6) return a;
    const out = new Array(a.length);
    for (let k = 0; k < a.length; k++) out[k] = a[k] + (b[k] - a[k]) * f;
    return out;
  }
  const at = (arr, t) => { const [i, j, f] = split(t); return mix(arr[i], arr[j], f); };
  const couplingRow = (d, t) => { const [i, j, f] = split(t); return mix(D.coupling[i][d], D.coupling[j][d], f); };
  const couplingCol = (cap, t) => { const out = []; for (let d = 0; d < K; d++) out.push(couplingRow(d, t)[cap]); return out; };

  // ------------------------------------------------------------ geometry
  function arc(p, q, out) {
    const n = D.arc.points, h = D.arc.height;
    for (let s = 0; s < n; s++) {
      const u = s / (n - 1);
      out.x.push(p[0] + (q[0] - p[0]) * u);
      out.y.push(p[1] + u * (q[1] - p[1]) + h * Math.sin(Math.PI * u));
      out.z.push(p[2] + (q[2] - p[2]) * u);
    }
    out.x.push(null); out.y.push(null); out.z.push(null);
  }

  // ------------------------------------------------------------ styles
  function digitStyle(norms) {
    const peak = Math.max(...norms, 1e-8), sizes = [], colors = [];
    norms.forEach((n, i) => {
      sizes.push(Math.max(8, n * 20));
      const k = 0.25 + 0.75 * Math.min(1, n / peak);
      const base = i === D.predicted ? [240, 172, 95] : [215, 215, 215];
      const rgb = base.map(v => Math.round(v * k));
      colors.push(`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`);
    });
    return { sizes, colors };
  }

  // ------------------------------------------------------------ renderers
  // Every renderer mutates gd.data in place; commit() then redraws once.
  // (Each separate Plotly.restyle on a 3D scene costs a full WebGL redraw.)
  let rev = 0, pending = false;
  function commit() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      gd.layout.datarevision = ++rev;
      Plotly.react(gd, gd.data, gd.layout);
    });
  }

  function buildConnections() {
    const bins = Array.from({ length: NB }, () => ({ x: [], y: [], z: [] }));
    if (state.digit !== null) {
      const c = couplingRow(state.digit, state.t);
      const max = D.digitMax[state.digit];
      const q = D.digitPos[state.digit];
      for (let k = 0; k < NCAP; k++) {
        if (c[k] <= D.routeMin || max <= 0) continue;
        const b = Math.min(NB - 1, Math.floor((c[k] / max) * NB));
        arc(D.primaryPos[k], q, bins[b]);
      }
    }
    bins.forEach((b, k) => Object.assign(gd.data[TR.bin0 + k], b));
  }

  function buildFocus() {
    const out = { x: [], y: [], z: [] }, col = [];
    if (state.cap !== null) {
      const votes = couplingCol(state.cap, state.t);
      for (let d = 0; d < K; d++) {
        arc(D.primaryPos[state.cap], D.digitPos[d], out);
        for (let s = 0; s <= D.arc.points; s++) col.push(votes[d]);
      }
    }
    Object.assign(gd.data[TR.focus], out);
    gd.data[TR.focus].line = { width: 5, color: col, colorscale: [[0, C.blue], [1, C.red]],
                               cmin: 0, cmax: Math.max(0.3, ...(col.length ? col : [0])) };
    renderCard();
  }
  function renderConnections() { buildConnections(); commit(); }
  function renderFocus() { buildFocus(); commit(); }

  function renderCard() {
    const el = document.getElementById('card');
    if (state.cap === null) { el.style.display = 'none'; return; }
    const cap = state.cap, t = state.t;
    const spatial = Math.floor(cap / D.T), type = cap % D.T;
    const gx = spatial % D.G, gy = Math.floor(spatial / D.G);
    const votes = couplingCol(cap, t);
    const ent = at(D.entropy, t)[cap];
    const top = votes.indexOf(Math.max(...votes));
    const W = 208, H = 70, bw = W / K;
    const vmax = Math.max(...votes, 0.2);
    let bars = '';
    votes.forEach((v, d) => {
      const h = (v / vmax) * (H - 16);
      const fill = d === D.predicted ? C.gold : C.blue;
      bars += `<rect x="${d * bw + 3}" y="${H - 14 - h}" width="${bw - 6}" height="${h}" rx="2" fill="${fill}" opacity="${d === top ? 1 : 0.6}"/>` +
              `<text x="${d * bw + bw / 2}" y="${H - 2}" fill="${C.muted}" font-size="11" text-anchor="middle" font-family="Georgia,serif">${d}</text>`;
    });
    el.innerHTML = `<h4>Primary capsule ${cap}</h4>
      <div class="row">grid position <b>(${gx}, ${gy})</b></div>
      <div class="row">capsule type <b>${type}</b></div>
      <div class="row">activation ‖u‖ <b>${D.activation[cap].toFixed(3)}</b></div>
      <div class="row">routing entropy <b>${(ent * Math.log2(votes.length)).toFixed(2)} bits</b></div>
      <div class="row">votes most for <b style="color:${top === D.predicted ? C.gold : C.ink}">${D.classes[top]}</b></div>
      <div class="row">Δφ if removed (exact) <b>${D.exact[cap].toExponential(2)}</b></div>
      <div class="row">Taylor estimate <b>${D.taylor[cap].toExponential(2)}</b></div>
      <div style="margin-top:8px;color:${C.muted}">coupling c<sub>ij</sub> to each class</div>
      <svg width="${W}" height="${H}">${bars}</svg>`;
    el.style.display = 'block';
  }

  function renderIteration() {
    const t = state.t;
    const ds = digitStyle(at(D.digitNorm, t));
    gd.data[TR.primary].marker.color = at(D.entropy, t);
    Object.assign(gd.data[TR.digits].marker, { size: ds.sizes, color: ds.colors });
    buildConnections();
    if (state.cap !== null) buildFocus();
    commit();
    const lbl = Math.abs(t - Math.round(t)) < 1e-6 ? String(Math.round(t) + 1) : (t + 1).toFixed(1);
    document.getElementById('itv').textContent = lbl;
    document.getElementById('scrub').value = t;
  }

  // ------------------------------------------------------------ controls
  function setDigit(d) {
    state.digit = (state.digit === d) ? null : d;
    document.querySelectorAll('button.digit').forEach(b => b.classList.toggle('on', +b.dataset.d === state.digit));
    renderConnections();
  }
  function setCap(k) { state.cap = (state.cap === k) ? null : k; renderFocus(); }

  const digitsEl = document.getElementById('digits');
  for (let d = 0; d < K; d++) {
    const b = document.createElement('button');
    b.className = 'digit' + (d === D.predicted ? ' pred' : '');
    const name = D.classes[d];
    b.dataset.d = d; b.textContent = name.length <= 2 ? name : d; b.title = `route to ${name}`;
    b.onclick = () => setDigit(d);
    digitsEl.appendChild(b);
  }

  document.getElementById('scrub').oninput = e => { stop(); state.t = +e.target.value; renderIteration(); };

  let raf = null;
  function stop() {
    state.playing = false; if (raf) cancelAnimationFrame(raf); raf = null;
    document.getElementById('play').textContent = '▶ play routing';
  }
  function play() {
    if (state.playing) return stop();
    state.playing = true;
    document.getElementById('play').textContent = '❚❚ pause';
    if (state.digit === null) setDigit(D.predicted);
    if (state.t >= LAST - 1e-6) state.t = 0;
    const STEP_MS = 1600, HOLD_MS = 600;
    let last = performance.now(), hold = HOLD_MS, lastDraw = 0;
    renderIteration();
    const tick = now => {
      if (!state.playing) return;
      const dt = Math.min(now - last, 60); last = now;
      if (hold > 0) hold -= dt;
      else {
        const prev = Math.floor(state.t + 1e-6);
        state.t = Math.min(LAST, state.t + dt / STEP_MS);
        if (Math.floor(state.t + 1e-6) > prev) { state.t = Math.floor(state.t + 1e-6); hold = HOLD_MS; }
      }
      if (now - lastDraw > 45 || hold === HOLD_MS) { renderIteration(); lastDraw = now; }
      if (state.t >= LAST - 1e-6 && hold <= 0) { renderIteration(); return stop(); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }
  document.getElementById('play').onclick = play;

  function toggle(btnId, traces) {
    const b = document.getElementById(btnId);
    b.onclick = () => {
      b.classList.toggle('on');
      traces.forEach(k => { gd.data[k].visible = b.classList.contains('on'); });
      commit();
    };
  }
  toggle('t-image', [TR.image]);
  toggle('t-infl', [TR.infl]);

  // attribution switcher: which map is drawn on the plane between image and grid
  const sel = document.getElementById('method');
  D.methods.forEach(([k, lab]) => {
    const o = document.createElement('option'); o.value = k; o.textContent = lab; sel.appendChild(o);
  });
  function showMethod(k) {
    const m = D.maps[k];
    gd.data[TR.infl].surfacecolor = m.norm;
    gd.data[TR.infl].customdata = m.raw;
    const el = document.getElementById('agree');
    el.innerHTML = k === 'exact'
      ? `<span style="color:${C.muted}">ground truth: the drop in ‖v<sub>ĵ</sub>‖ when each capsule is removed and routing re-runs</span>`
      : `<div class="k">agreement with exact ablation, this image</div>
         <div class="m">${m.rho.toFixed(3)}<span>Spearman ρ</span> ${(100 * m.p1).toFixed(0)}%<span>top-${D.topLabel} recovered</span></div>`;
    commit();
  }
  sel.onchange = e => showMethod(e.target.value);
  showMethod('exact');
  toggle('t-labels', [TR.labels, TR.grid]);

  const CAMS = [
    { name: 'view: orbit', eye: { x: 1.25, y: 1.05, z: 1.25 }, center: { x: 0, y: -0.1, z: -0.05 }, up: { x: 0, y: 0, z: 1 } },
    { name: 'view: layers', eye: { x: 1.9, y: 0.55, z: 1.35 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } },
    { name: 'view: side', eye: { x: 2.6, y: 0.15, z: 0.0 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } },
    { name: 'view: top', eye: { x: 0.01, y: 2.7, z: 0.01 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } },
  ];
  const camBtn = document.getElementById('cam');
  camBtn.textContent = CAMS[0].name;
  camBtn.onclick = () => {
    state.cam = (state.cam + 1) % CAMS.length;
    const c = CAMS[state.cam];
    camBtn.textContent = c.name;
    Plotly.relayout(gd, { 'scene.camera': { eye: c.eye, center: c.center, up: c.up } });
  };

  gd.on('plotly_click', e => {
    const p = e.points && e.points[0];
    if (!p) return;
    if (p.curveNumber === TR.digits) setDigit(p.pointNumber);
    else if (p.curveNumber === TR.primary) setCap(p.pointNumber);
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { stop(); if (state.digit !== null) setDigit(state.digit); if (state.cap !== null) setCap(state.cap); }
    if (e.key === ' ') { e.preventDefault(); play(); }
  });

  window.__hero = { state, setDigit, setCap, play, renderIteration };
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
