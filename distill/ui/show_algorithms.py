#!/usr/bin/env python3
"""
Renders all distillation algorithms in LaTeX via a self-contained HTML/MathJax page.
Opens automatically in the default browser.

Usage:
    python scripts/show_algorithms.py
    python scripts/show_algorithms.py --output algorithms.html   # save only, no open
"""

import argparse
import tempfile
import webbrowser
from pathlib import Path

from .algorithm_defs import ALGORITHMS  # noqa: F401 — re-exported for callers


def build_html(algorithms: list) -> str:
    sections = []
    for i, alg in enumerate(algorithms):
        param_rows = "".join(
            f"<tr><td>\\({p}\\)</td><td>{desc}</td></tr>"
            for p, desc in alg.get("params", [])
        )
        param_table = (
            f"""
            <table class="params">
              <thead><tr><th>Symbol</th><th>Meaning</th></tr></thead>
              <tbody>{param_rows}</tbody>
            </table>"""
            if param_rows else ""
        )
        eq_blocks = "".join(
            f"""
            <div class="eq-block">
              <div class="eq-label">{b['label']}</div>
              <div class="eq">\\[ {b['eq']} \\]</div>
            </div>"""
            for b in alg["blocks"]
        )
        sections.append(f"""
        <section>
          <div class="section-header">
            <span class="num">{i+1}</span>
            <div>
              <h2>{alg['title']}</h2>
              <span class="file-tag">{alg['file']}</span>
            </div>
          </div>
          <p class="desc">{alg['description']}</p>
          {eq_blocks}
          {param_table}
        </section>
        """)

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Distillation Algorithms — LaTeX Reference</title>
<script>
  MathJax = {{
    tex: {{
      inlineMath: [['\\\\(','\\\\)']],
      displayMath: [['\\\\[','\\\\]']],
      tags: 'ams',
    }},
    options: {{ skipHtmlTags: ['script','noscript','style','textarea'] }},
  }};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e;
    --accent: #7c6af7; --accent2: #4fc3f7; --text: #e2e4f0;
    --muted: #8b8fa8; --tag: #2d2f45; --num-bg: #7c6af7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.6; padding: 2rem 1rem; }}
  header {{ text-align: center; margin-bottom: 3rem;
    padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); }}
  header h1 {{ font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.4rem; }}
  header p {{ color: var(--muted); font-size: 0.95rem; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  section {{ background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.8rem 2rem; margin-bottom: 2rem; }}
  .section-header {{ display: flex; align-items: flex-start;
    gap: 1rem; margin-bottom: 1rem; }}
  .num {{ display: inline-flex; align-items: center; justify-content: center;
    min-width: 2rem; height: 2rem; background: var(--num-bg); color: #fff;
    font-weight: 700; font-size: 0.9rem; border-radius: 50%;
    flex-shrink: 0; margin-top: 0.15rem; }}
  h2 {{ font-size: 1.2rem; font-weight: 600; color: var(--text); margin-bottom: 0.25rem; }}
  .file-tag {{ font-size: 0.75rem; background: var(--tag); color: var(--accent2);
    padding: 0.15rem 0.6rem; border-radius: 4px;
    font-family: 'SF Mono', 'Fira Code', monospace; }}
  .desc {{ color: var(--muted); font-size: 0.92rem; margin-bottom: 1.4rem; padding-left: 3rem; }}
  .eq-block {{ background: #13151f; border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 6px;
    padding: 0.8rem 1.2rem; margin-bottom: 0.9rem; }}
  .eq-label {{ font-size: 0.78rem; font-weight: 600; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }}
  .eq {{ overflow-x: auto; }}
  .params {{ width: 100%; border-collapse: collapse; margin-top: 1.2rem; font-size: 0.88rem; }}
  .params thead tr {{ background: var(--tag); color: var(--muted);
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }}
  .params th, .params td {{ padding: 0.5rem 0.9rem; text-align: left;
    border-bottom: 1px solid var(--border); }}
  .params td:first-child {{ font-family: 'SF Mono', 'Fira Code', monospace;
    color: var(--accent2); white-space: nowrap; }}
  .params tr:last-child td {{ border-bottom: none; }}
  footer {{ text-align: center; color: var(--muted); font-size: 0.8rem;
    margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Distillation Algorithms</h1>
    <p>Mathematical reference — rendered from source code</p>
  </header>
  {body}
  <footer>Generated by show_algorithms.py &nbsp;&bull;&nbsp; MathJax 3</footer>
</div>
</body>
</html>"""


def build_latex(algorithms: list) -> str:
    lines = [
        r"\documentclass[12pt]{article}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{xcolor}",
        r"\title{Distillation Algorithms}",
        r"\author{show\_algorithms.py}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
    ]
    for i, alg in enumerate(algorithms):
        lines.append(r"\section{" + alg["title"].replace("&", r"\&") + "}")
        lines.append(r"\texttt{" + alg["file"].replace("_", r"\_") + r"}")
        lines.append(r"\par\medskip")
        lines.append(alg["description"].replace("\\(", "$").replace("\\)", "$"))
        lines.append(r"\par")
        for b in alg["blocks"]:
            lines.append(r"\paragraph{" + b["label"].replace("$", "") + "}")
            lines.append(r"\begin{equation*}")
            lines.append(b["eq"])
            lines.append(r"\end{equation*}")
        if alg.get("params"):
            lines.append(r"\begin{center}\begin{tabular}{ll}")
            lines.append(r"\hline\textbf{Symbol} & \textbf{Meaning} \\ \hline")
            for p, desc in alg["params"]:
                p_clean = p.replace("\\\\", "\\")
                lines.append(f"${ p_clean }$ & { desc } \\\\")
            lines.append(r"\hline\end{tabular}\end{center}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Show distillation algorithms in LaTeX")
    parser.add_argument("--output", type=str, default=None,
                        help="Save HTML to this path instead of a temp file")
    parser.add_argument("--latex", type=str, default=None,
                        help="Also export a .tex source file to this path")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open browser automatically")
    args = parser.parse_args()

    html = build_html(ALGORITHMS)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
        tmp.write(html)
        tmp.close()
        out_path = Path(tmp.name)

    out_path.write_text(html, encoding="utf-8")
    print(f"HTML written → {out_path}")

    if args.latex:
        latex_path = Path(args.latex).resolve()
        latex_path.write_text(build_latex(ALGORITHMS), encoding="utf-8")
        print(f"LaTeX written → {latex_path}")

    if not args.no_open:
        webbrowser.open(out_path.as_uri())
        print("Opened in browser.")


if __name__ == "__main__":
    main()
