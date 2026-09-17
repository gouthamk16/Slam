#!/usr/bin/env python3
"""Build docs/slam2.0-architecture.{html,pdf} from docs/notes.src.html.

Code listings are pulled out of the real source files at build time, so a listing in
the document is always exactly what the repository contains. Placeholders:

    {{func:src/slam/tracking.py:_track_motion_model}}   one function or method
    {{lines:src/slam/optimize/solver.py:43-60}}         a line range

Needs `pip install weasyprint` (HTML/CSS -> PDF, no JavaScript).
"""

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/notes.src.html"
OUT_HTML = ROOT / "docs/slam2.0-architecture.html"
OUT_PDF = ROOT / "docs/slam2.0-architecture.pdf"


def _func(path: str, name: str) -> str:
    lines = (ROOT / path).read_text().splitlines()
    pat = re.compile(rf"^\s*(def|class) {re.escape(name)}\b")
    start = next((i for i, l in enumerate(lines) if pat.match(l)), None)
    if start is None:
        raise SystemExit(f"{path}: no def/class named {name}")
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        end += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _lines(path: str, spec: str) -> str:
    a, b = (int(x) for x in spec.split("-"))
    return "\n".join((ROOT / path).read_text().splitlines()[a - 1 : b])


LONG = 28  # listings taller than this cannot fit beside text on one page, so let them split


def _resolve(m: re.Match) -> str:
    kind, path, arg = m.groups()
    code = _func(path, arg) if kind == "func" else _lines(path, arg)
    cls = "code long" if code.count("\n") + 1 > LONG else "code"
    return (
        f'<figure class="{cls}"><figcaption>{html.escape(path)}</figcaption>'
        f"<pre><code>{html.escape(code)}</code></pre></figure>"
    )


def main() -> None:
    text = re.sub(r"\{\{(func|lines):([^:]+):([^}]+)\}\}", _resolve, SRC.read_text())
    OUT_HTML.write_text(text)
    from weasyprint import HTML

    HTML(string=text, base_url=str(ROOT / "docs")).write_pdf(OUT_PDF)
    print(f"wrote {OUT_HTML.relative_to(ROOT)} and {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
