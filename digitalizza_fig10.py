"""
Digitisation of NASA TM-81232, Figure 10 (wake geometry), used for dati_sperimentali.SCIA.

Not part of the model: kept only so the numbers can be reproduced and checked.
Usage:
    pdftoppm -r 400 -f 53 -l 53 -png 19820004169.pdf fig10     (PDF from ntrs.nasa.gov)
    python digitalizza_fig10.py fig10-53.png
Prints, every 20 deg of wake age, the centres of the dark runs crossed by a 7-px wide
pixel column. Each row lists the curves top to bottom (z: 5, 8, 12 deg; r: 12, 8, 5 deg);
extra entries come from tick marks, labels and label pointers and were discarded by hand.
The calibration constants below are specific to the 400-dpi render.
"""
import sys
import numpy as np
from PIL import Image

def runs(idx):
    if len(idx) == 0: return []
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v != p + 1: out.append((s, p)); s = v
        p = v
    out.append((s, p)); return out

im = np.array(Image.open(sys.argv[1]).convert("L")) < 128
H, W = im.shape
# axis lines: long vertical segments and long horizontal segments
col_sum = im.sum(0); xa = int(np.argmax(col_sum[: W // 3]))
seg = runs(np.where(im[:, xa])[0]); seg = [s for s in seg if s[1] - s[0] > 200]
print("y-axis column", xa, "vertical segments (row start, end):", seg)
row_sum = im.sum(1)
long_rows = [r for r in range(H) if row_sum[r] > 0.4 * W]
print("long horizontal rows:", runs(np.array(long_rows)))

# x ticks hang below the z = 0 axis (rows 1027-1045) and rise above the r = 1 axis
for name, rows in (("top ticks", slice(1028, 1046)), ("bottom ticks", slice(2362, 2378))):
    c = im[rows, :].sum(0)
    print(name, [int((a + b) / 2) for a, b in runs(np.where(c >= 12)[0])])

# calibration (pixels of the 400-dpi render)
X0, PX_DEG = 758.4, 254.3 / 50          # psi = 0 column, pixels per degree (ticks 100..350 deg)
Z0, PX_Z = 1022.0, (1867 - 1022) / 0.4  # z/R = 0 row, pixels per unit z/R (axis 0 .. 0.4)
R1, PX_R = 2384.0, (2384 - 1967) / 0.2  # r/R = 1 row, pixels per unit r/R (axis 1.0 .. 0.8)
print("\npsi  | z/R curve centres (top to bottom)      | r/R curve centres (top to bottom)")
for psi in range(0, 381, 20):
    x = int(round(X0 + psi * PX_DEG))
    x = min(x, 2695)
    band = im[:, x - 3:x + 4].any(1)    # 7-px wide column, catches dashes
    zr = [((a + b) / 2 - Z0) / PX_Z for a, b in runs(np.where(band[1030:1860])[0] + 1030)]
    rr = [1 - (R1 - (a + b) / 2) / PX_R for a, b in runs(np.where(band[1972:2378])[0] + 1972)]
    print(f"{psi:4d} | {', '.join(f'{v:.3f}' for v in zr):40s} | {', '.join(f'{v:.3f}' for v in rr)}")
