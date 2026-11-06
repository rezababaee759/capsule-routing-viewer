#!/usr/bin/env python3
"""
Captures of the 3D routing cube for the paper (headless Chrome over CDP, 2x DPR).

Needs a headless Chrome with --remote-debugging-port=9333 --remote-allow-origins='*'.
Samples are the rule-selected case-study images (scripts/figure_data.py).

Output: paper/figures/raw/hero_<dataset>.png, the full cube with the app's overlays (orbit view).
"""

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import websocket

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from explorer import data as D, hero as H                                   # noqa: E402
from explorer.registry import DATASETS                                      # noqa: E402
from figstyle import case_index                                             # noqa: E402

RAW = ROOT / "paper" / "figures" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
# Print legibility (hero only): the figure is shrunk to the text width, so the overlays are
# enlarged, the usage hint (how to drag and click) is dropped, and the capsule card moves to the
# empty left side so that it does not cover the scene labels.
PRINT_CSS = ("#hint{display:none!important}#attr{top:14px!important}"
             "#card{top:150px!important;left:12px!important;right:auto!important;width:222px!important}"
             "#attr,#card,#legend,#bar,#iterlabel{zoom:1.4}#card svg text{font-size:15px!important}")
# Print legibility of the scene labels: WebGL text in the 3D scene is drawn at depth, so the class
# names crowd together and the routes or the image plane can hide a label. For print, the labels
# are redrawn as scene annotations (an overlay that nothing covers) at the same 3D anchors: class
# names alternate between a near and a far column, joined to their capsule by a thin leader line,
# the predicted class in its orange; the three scene labels get a dark backing.
LABEL_JS = """(()=>{{const g=document.getElementById('hero'), d=g._fullData[1], L=g._fullData[5];
const f=(d.textfont||{{}}).family||'Georgia,serif', ann=[], P={pred}, only={only};
for(let i=0;i<d.x.length;i++){{ if(only && i!==P) continue;
  ann.push({{x:d.x[i],y:d.y[i],z:d.z[i],text:d.text[i],showarrow:true,arrowhead:0,arrowwidth:1,
   arrowcolor:'rgba(210,210,210,0.55)',ax:(only?34:(i%2?92:30)),ay:0,xanchor:'left',
   font:{{family:f,size:{size},color:i===P?'#F0AC5F':'#E6E6E6'}},
   bgcolor:(only?'rgba(11,13,18,0.7)':'rgba(0,0,0,0)')}}); }}
if(!only) for(let i=0;i<L.x.length;i++) ann.push({{x:L.x[i],y:L.y[i],z:L.z[i],text:L.text[i],showarrow:false,
  xanchor:'left',xshift:(i===2?24:0),font:{{family:f,size:{size},color:(L.textfont||{{}}).color||'#8A8F98'}},bgcolor:'rgba(11,13,18,0.7)'}});
g.data[1].mode='markers'; g.data[5].visible=false; g.layout.scene.annotations=ann;
Plotly.react(g,g.data,g.layout);}})();"""
PRINT_TEXT = ("Plotly.restyle('hero', {'colorbar.tickfont.size': 17, 'colorbar.title.font.size': 18,"
              " 'colorbar.thickness': 14}, [3]);")


class Tab:
    def __init__(self, port=9333):
        tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
        t = [t for t in tabs if t["type"] == "page"][0]
        self.ws = websocket.create_connection(t["webSocketDebuggerUrl"], timeout=180, suppress_origin=True)
        self.n = 0

    def cdp(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                return m.get("result", {})

    def js(self, expr):
        return self.cdp("Runtime.evaluate", expression=expr, returnByValue=True,
                        awaitPromise=True).get("result", {}).get("value")

    def shot(self, path):
        Path(path).write_bytes(base64.b64decode(self.cdp("Page.captureScreenshot")["data"]))


def build_page(key, idx, device):
    spec = DATASETS[key]
    model = spec.build_model(device)
    x = spec.dataset()[idx][0].unsqueeze(0).to(device)
    tr = D.trace_routing(model, x, spec)
    attr = D.attributions(model, x)
    page = RAW / f"page_{key}.html"
    page.write_text(H.build_html(tr, attr, spec.classes, height=860))
    return page, tr["predicted"], int(np.argmax(attr["exact"]))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab = Tab()
    tab.cdp("Page.enable")
    tab.cdp("Emulation.setDeviceMetricsOverride", width=1400, height=860, deviceScaleFactor=2, mobile=False)
    for key in sys.argv[1:] or ["mnist"]:
        idx = case_index(key, "confident_correct")
        page, pred, top = build_page(key, idx, device)
        tab.cdp("Page.navigate", url=page.as_uri())
        for _ in range(60):
            if tab.js("!!window.__hero"):
                break
            time.sleep(0.5)
        time.sleep(2)
        # Hero with overlays: orbit view, predicted class routed, Taylor map, top capsule card.
        # Slightly wider than the app's default view so the overlay card does not cover scene labels.
        tab.js("Plotly.relayout('hero', {'scene.camera': {eye: {x: 1.45, y: 1.2, z: 1.45},"
               " center: {x: -0.08, y: -0.1, z: -0.05}}});")
        tab.js(f"window.__hero.setDigit({pred}); window.__hero.setCap({top});"
               "const s=document.getElementById('method'); s.value='taylor'; s.onchange({target:s});")
        tab.js("const pc=document.createElement('style'); pc.id='printcss'; pc.textContent='" + PRINT_CSS + "';"
               "document.head.appendChild(pc);" + PRINT_TEXT + LABEL_JS.format(pred=pred, only="false", size=19))
        time.sleep(3)
        tab.shot(RAW / f"hero_{key}.png")
        tab.js("document.getElementById('printcss').remove();")
        print(f"{key}: idx {idx}, predicted {DATASETS[key].classes[pred]}, top capsule {top}", flush=True)


if __name__ == "__main__":
    main()
