"""One-time export of evidence screenshots from the PRIVATE run folder.

    python src/export_screenshots.py --eval-dir <private research/eval>

Only screenshots taken during a task's scored attempt are eligible (file time
inside that attempt's window; later re-runs overwrote 14 of the 31). Of those,
only the ones that show evidence for their own task are published, each
cropped to the relevant window so that no browser tabs, bookmarks, window
titles (which showed an email address), taskbar or other applications remain.
Exclusions and their reasons are listed in evidence/README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evidence" / "screenshots"
MAX_W = 1400

# Crop boxes (left, top, right, bottom) in the original 3200x1800 screenshots.
# Browser pages: below the browser chrome (y 315) and above the taskbar
# (y 1697), without the overlay toolbar at the right edge.
PAGE = (400, 315, 2800, 1695)
CROPS = {
    "EVAL-HW-01": (340, 505, 2125, 1570),    # Word window, below its title bar
    "EVAL-HW-02": (340, 505, 2125, 1570),
    "EVAL-HW-03": (560, 560, 2640, 1695),    # maximised Word: the page only
    "EVAL-HW-07": (500, 315, 2700, 1695),
    "EVAL-READ-01": PAGE,
    "EVAL-READ-02": PAGE,
    "EVAL-TEACH-01": PAGE,
    "EVAL-TEACH-02": PAGE,
    "EVAL-TEACH-05": PAGE,
    "EVAL-XLATE-02": (200, 315, 2400, 1695),  # Notepad window
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for task, box in CROPS.items():
        im = Image.open(args.eval_dir / "evidence" / f"{task}.png").convert("RGB").crop(box)
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        im.save(OUT / f"{task}.png", optimize=True)
        print(task, im.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
