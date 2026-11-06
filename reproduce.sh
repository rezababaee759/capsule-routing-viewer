#!/usr/bin/env bash
# Rebuilds every artifact from the CapsNet checkpoints (models/) and the datasets (data/).
#   ./reproduce.sh            everything
#   ./reproduce.sh <stage>    one of: tests experiments summary figures captures numbers paper
# Order matters: each stage reads the previous stages' outputs.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DATASETS="mnist fashionmnist cifar10"
VARIANTS="seed42 seed123 seed456"

tests()        { $PY tests/test_engine.py && $PY tests/test_stats.py; }
experiments()  {
  for d in $DATASETS; do
    for v in $VARIANTS; do $PY experiments/run_fidelity.py --dataset "$d" --variant "$v" --set main; done
    $PY experiments/run_fidelity.py --dataset "$d" --set misclassified   # the reference run
    for v in $VARIANTS; do $PY experiments/run_margin.py --dataset "$d" --variant "$v"; done
    $PY experiments/run_randomization.py --dataset "$d"
    $PY experiments/run_additivity.py --dataset "$d"
  done
  $PY experiments/run_cost.py          # timing: run on an otherwise idle machine
}
summary()      { $PY experiments/summarize.py && $PY experiments/run_gates.py; }
figures()      {
  $PY scripts/figure_data.py
  (cd scripts && $PY make_figures.py && $PY fig_cases.py)
}
captures()     {
  # Screenshot of the routing scene: headless Chrome loads the scene page that capture_app.py writes.
  local tmp; tmp=$(mktemp -d)
  google-chrome --headless=new --remote-debugging-port=9333 --remote-allow-origins='*' \
    --user-data-dir="$tmp" --no-first-run --use-angle=swiftshader --enable-unsafe-swiftshader \
    --window-size=1440,900 about:blank >/dev/null 2>&1 & local chrome=$!
  trap 'kill $chrome 2>/dev/null || true' RETURN
  sleep 4
  (cd scripts && $PY capture_app.py mnist && $PY compose_app_figures.py)
}
numbers()      { $PY scripts/make_numbers.py; }
paper()        {
  # The manuscript sources are not distributed with the code while it is under review.
  [ -f paper/main.tex ] || { echo "paper: paper/main.tex not present; skipped"; return 0; }
  $PY scripts/check_paper.py && (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null \
                   && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null \
                   && pdflatex -interaction=nonstopmode main.tex >/dev/null) && echo "paper/main.pdf built"; }

if [ $# -eq 0 ]; then tests; experiments; summary; figures; captures; numbers; paper
else "$@"; fi
