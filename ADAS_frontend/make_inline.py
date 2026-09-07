"""
make_inline.py
==============
Regenerates dist/index_inline.html from the normal Vite build, as a single
self-contained file (CSS + JS inlined) that the backend serves. Because there
are no external /assets references, the page can never render blank due to an
asset-path or content-type problem.

Run from ADAS/ (after building the frontend):
    python ../ADAS_frontend/make_inline.py        # or update the dist path
"""

import re
import sys
from pathlib import Path

# Candidate dist locations (same logic as the server).
ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ROOT / "dist",                                      # ADAS_frontend/dist
    Path(__file__).resolve().parents[1] / "ADAS/dist",  # <workspace>/ADAS/dist
]
dist = next((p for p in CANDIDATES if (p / "index.html").exists()), None)
if dist is None:
    print("No dist/index.html found — run `npm run build` first.")
    sys.exit(0)

html = (dist / "index.html").read_text()
if not (dist / "assets").exists():
    print("No assets/ folder — run `npm run build` first.")
    sys.exit(0)

css = next((dist / "assets" / f).read_text()
           for f in sorted(p.name for p in (dist / "assets").iterdir())
           if f.endswith(".css"))
js = next((dist / "assets" / f).read_text()
          for f in sorted(p.name for p in (dist / "assets").iterdir())
          if f.endswith(".js"))

new_html = re.sub(r'<script type="module"[^>]*src="[^"]*"[^>]*></script>', '', html)
new_html = re.sub(r'<link rel="stylesheet"[^>]*>', '', new_html)
new_html = new_html.replace("</head>", f"  <style>\n{css}\n  </style>\n  </head>")
new_html = new_html.replace("</body>", f'  <script type="module">\n{js}\n  </script>\n  </body>')

out = dist / "index_inline.html"
out.write_text(new_html)
print(f"wrote {out}  ({len(new_html)} bytes, single self-contained file)")
