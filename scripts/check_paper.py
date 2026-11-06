#!/usr/bin/env python3
"""
Machine gate for the manuscript (paper/main.tex). Exits non-zero on any violation.

1. No bare numbers.  Outside \\V{key}, references, citations, labels and layout
   arguments, prose contains no digits except in whitelisted names (CIFAR-10, 3D...).
   Inside math, only the notational digits 0, 1, 2 are allowed (squares, 1 + x, [0, 1)).
2. Every \\V{key} exists in paper/generated/numbers.json (which make_numbers.py writes
   from the result files).
3. Every \\cite key is in paper/references.bib AND marked verified in
   paper/citations.yaml (checked against the real publication).
4. Every \\claim{id} has an entry in paper/claims.yaml whose evidence resolves:
     - number:  key present in numbers.json
     - file:    path exists (optional `contains:` text must occur in it)
     - json:    file + slash-path that exists in that JSON
   and every paragraph that prints a number (\\V) carries at least one \\claim.
5. Every \\includegraphics target exists.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
WHITELIST = ["CIFAR-10", "Fashion-MNIST", "3D", "2D", "conv1", "Plotly.js", "ReLU"]
MATH_OK = set("012")


def load_yaml(path):
    """Minimal YAML subset reader (lists of mappings, scalars, nested `evidence` lists)."""
    import yaml                                                              # noqa: F401
    return yaml.safe_load(path.read_text())


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def body(tex):
    a = tex.index(r"\abstract{")
    b = tex.index(r"\reftitle") if r"\reftitle" in tex else len(tex)
    return tex[a:b]


def remove_allowed(text):
    patterns = [
        r"\\V\{[^}]*\}", r"\\(?:c|C)?ref\{[^}]*\}", r"\\eqref\{[^}]*\}", r"\\label\{[^}]*\}",
        r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{[^}]*\}", r"\\claim\{[^}]*\}",
        r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", r"\\begin\{[^}]*\}(?:\[[^\]]*\])?(?:\{[^}]*\})?",
        r"\\end\{[^}]*\}", r"\\(?:h|v)space\*?\{[^}]*\}", r"\{-?\\extralength\}\{[0-9.]+cm\}",
        r"\(G[1-9]\)",                                   # design-goal labels (identifiers)
        r"\{-?[0-9.]+(?:cm|mm|pt|em)\}",                  # pure length arguments
        r"\\setlength\{[^}]*\}\{[^}]*\}",
        r"\\(?:url|href)\{[^}]*\}", r"\\(?:multicolumn|multirow)\{[^}]*\}\{[^}]*\}",
        r"\\(?:small|footnotesize|scriptsize|normalsize|centering|toprule|midrule|bottomrule|cmidrule(?:\([^)]*\))?\{[^}]*\})",
        r"\d*\.?\d+\\(?:textwidth|linewidth|columnwidth)", r"\[[0-9.]+(?:pt|em|ex|cm|mm)\]",
    ]
    for p in patterns:
        text = re.sub(p, " ", text)
    for w in WHITELIST:
        text = text.replace(w, " ")
    return text


def split_math(text):
    """Yield (is_math, chunk)."""
    pat = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{(equation|align|gather)\*?\}.*?\\end\{\2\*?\})",
                     re.S)
    pos = 0
    for m in pat.finditer(text):
        yield False, text[pos:m.start()]
        yield True, m.group(0)
        pos = m.end()
    yield False, text[pos:]


def line_of(full, needle_pos):
    return full.count("\n", 0, needle_pos) + 1


def main():
    tex_path = PAPER / "main.tex"
    raw = tex_path.read_text()
    tex = strip_comments(raw)
    errors = []
    numbers = json.loads((PAPER / "generated" / "numbers.json").read_text())

    # 1. bare numbers
    b = body(tex)
    for is_math, chunk in split_math(b):              # detect math BEFORE stripping environments
        chunk = remove_allowed(chunk)
        digits = re.finditer(r"\d+(?:\.\d+)?", chunk)
        for d in digits:
            tok = d.group(0)
            if is_math and set(tok) <= MATH_OK and len(tok) == 1:
                continue
            ctx = chunk[max(0, d.start() - 40): d.end() + 40].replace("\n", " ")
            errors.append(f"bare number '{tok}' {'in math' if is_math else 'in text'}: …{ctx}…")

    # 2. number keys
    for m in re.finditer(r"\\V\{([^}]*)\}", tex):
        if m.group(1) not in numbers:
            errors.append(f"line {line_of(tex, m.start())}: unknown number key \\V{{{m.group(1)}}}")

    # 3. citations
    bib = (PAPER / "references.bib").read_text()
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cit_path = PAPER / "citations.yaml"
    verified = {}
    if cit_path.exists():
        for e in load_yaml(cit_path) or []:
            verified[e["key"]] = e.get("status")
    cited = set()
    for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            cited.add(k.strip())
    for k in sorted(cited):
        if k not in bib_keys:
            errors.append(f"citation '{k}' not in references.bib")
        elif verified.get(k) != "verified":
            errors.append(f"citation '{k}' not marked verified in citations.yaml (status: {verified.get(k)})")

    # 4. claims
    claims_path = PAPER / "claims.yaml"
    ledger = {c["id"]: c for c in (load_yaml(claims_path) or [])} if claims_path.exists() else {}
    used = set(re.findall(r"\\claim\{([^}]*)\}", tex))
    for cid in sorted(used):
        c = ledger.get(cid)
        if c is None:
            errors.append(f"claim {cid} used in main.tex but missing from claims.yaml")
            continue
        for ev in c.get("evidence", []):
            if "number" in ev:
                if ev["number"] not in numbers:
                    errors.append(f"claim {cid}: evidence number key '{ev['number']}' unknown")
            elif "file" in ev:
                f = ROOT / ev["file"]
                if not f.exists():
                    errors.append(f"claim {cid}: evidence file '{ev['file']}' missing")
                    continue
                if "contains" in ev and ev["contains"] not in f.read_text(errors="ignore"):
                    errors.append(f"claim {cid}: '{ev['contains']}' not found in {ev['file']}")
                if "json" in ev:
                    try:
                        d = json.loads(f.read_text())
                        for k in ev["json"].split("/"):
                            d = d[int(k)] if isinstance(d, list) else d[k]
                    except Exception:                                    # noqa: BLE001
                        errors.append(f"claim {cid}: json path '{ev['json']}' not in {ev['file']}")
            else:
                errors.append(f"claim {cid}: evidence entry without number/file")
        if not c.get("evidence"):
            errors.append(f"claim {cid}: no evidence listed")
    for cid in sorted(set(ledger) - used):
        errors.append(f"claim {cid} in claims.yaml is not used in main.tex (stale)")
    for para in re.split(r"\n\s*\n", b):
        if r"\V{" in para and r"\claim{" not in para and r"\begin{table" not in para \
                and r"\caption" not in para and r"\abstract" not in para:
            snippet = re.sub(r"\s+", " ", para.strip())[:90]
            errors.append(f"paragraph prints numbers without a \\claim: {snippet}…")

    # 5. figures
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        f = PAPER / m.group(1)
        if not (f.exists() or f.with_suffix(".pdf").exists()):
            errors.append(f"figure file missing: {m.group(1)}")

    n_numbers = len(re.findall(r"\\V\{", tex))
    print(f"check_paper: {n_numbers} generated numbers, {len(used)} claims, {len(cited)} citations")
    if errors:
        print(f"{len(errors)} violation(s):")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
