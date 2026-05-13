"""
Script to extract all cells and outputs from cat&ada_regularized.ipynb
and save plots as PNG files.
"""

import json
import base64
import re
import os
from pathlib import Path
from html.parser import HTMLParser

# ── Paths ──────────────────────────────────────────────────────────────────────
NOTEBOOK_PATH = Path(r"C:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\notebooks\cat&ada_regularized.ipynb")
OUTPUT_DIR    = Path(r"C:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\notebooks\catboost_info")
TEXT_FILE     = OUTPUT_DIR / "notebook_content.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── HTML → plain-text helper ───────────────────────────────────────────────────
class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self._text_parts = []

    def handle_data(self, d):
        self._text_parts.append(d)

    def get_text(self):
        return "".join(self._text_parts)

def strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    return s.get_text()

# ── Load notebook ───────────────────────────────────────────────────────────────
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells   = nb.get("cells", [])
lines   = []          # lines to write into the .txt file
png_idx = 0           # counter for saved PNG files

# ── Process each cell ──────────────────────────────────────────────────────────
for cell_num, cell in enumerate(cells, start=1):
    cell_type = cell.get("cell_type", "unknown")
    source    = "".join(cell.get("source", []))

    lines.append(f"{'='*80}")
    lines.append(f"CELL #{cell_num}  [{cell_type.upper()}]")
    lines.append(f"{'='*80}")

    # ── Source ────────────────────────────────────────────────────────────────
    if cell_type == "markdown":
        lines.append("[MARKDOWN]")
    else:
        lines.append("[SOURCE CODE]")

    lines.append(source)
    lines.append("")

    # ── Outputs (code cells only) ─────────────────────────────────────────────
    outputs = cell.get("outputs", [])
    if outputs:
        lines.append("[OUTPUTS]")
        for out in outputs:
            out_type = out.get("output_type", "")

            # -- stream (stdout / stderr) --------------------------------------
            if out_type == "stream":
                text = "".join(out.get("text", []))
                lines.append(text)

            # -- display_data / execute_result ---------------------------------
            elif out_type in ("display_data", "execute_result"):
                data = out.get("data", {})

                # Try to save image/png
                if "image/png" in data:
                    png_idx += 1
                    img_b64   = "".join(data["image/png"])
                    img_bytes = base64.b64decode(img_b64)
                    png_name  = f"plot_{png_idx:03d}.png"
                    png_path  = OUTPUT_DIR / png_name
                    with open(png_path, "wb") as img_f:
                        img_f.write(img_bytes)
                    lines.append(f"[PLOT saved as: {png_name}]")

                # Prefer plain text representation
                if "text/plain" in data:
                    txt = "".join(data["text/plain"])
                    lines.append(txt)
                elif "text/html" in data:
                    # Fall back: strip HTML tags
                    html = "".join(data["text/html"])
                    lines.append(strip_html(html))

            # -- error ---------------------------------------------------------
            elif out_type == "error":
                ename  = out.get("ename", "")
                evalue = out.get("evalue", "")
                lines.append(f"[ERROR] {ename}: {evalue}")
                traceback = out.get("traceback", [])
                # Strip ANSI escape codes
                ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
                for tb_line in traceback:
                    lines.append(ansi_escape.sub("", tb_line))

        lines.append("")

lines.append(f"{'='*80}")
lines.append(f"END OF NOTEBOOK — {png_idx} plot(s) saved to: {OUTPUT_DIR}")
lines.append(f"{'='*80}")

# ── Write .txt file ────────────────────────────────────────────────────────────
with open(TEXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[OK]  Text file written : {TEXT_FILE}")
print(f"[OK]  PNG files saved   : {png_idx} file(s) in {OUTPUT_DIR}")
