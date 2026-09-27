"""O-K5: measure the poll cross of Fig. 1 (bads-cartoon.png): the extent of
the black pixels along the horizontal and the vertical line through the
current point, in the left panel."""
import numpy as np
from PIL import Image

im = np.asarray(
    Image.open(
        "/home/user/pybads-review/docsrc/source/_static/bads-cartoon.png"
    ).convert("RGB")
).astype(int)
print("image size (h, w):", im.shape[:2])
black = np.all(im < 40, axis=2)
# Left panel, the region of the cross
sub = black[480:720, 90:400]
ys, xs = np.nonzero(sub)
ys += 480
xs += 90
# The centre: the row and column with the most black pixels in the region
row = np.bincount(ys).argmax()
col = np.bincount(xs).argmax()
r = np.flatnonzero(black[row, 90:400]) + 90
c = np.flatnonzero(black[480:720, col]) + 480
print(f"centre row {row}, column {col}")
print(
    f"horizontal arm: x from {r.min()} to {r.max()}, half-widths {col - r.min()} and {r.max() - col}"
)
print(
    f"vertical arm: y from {c.min()} to {c.max()}, half-heights {row - c.min()} and {c.max() - row}"
)
# Panel frame: the axes lines (long black runs)
colsum = black[:, :800].sum(axis=0)
rowsum = black[:, :800].sum(axis=1)
print(
    "left axis column (most black pixels in x < 100):",
    int(np.argmax(colsum[:100])),
    "bottom axis row:",
    int(np.argmax(rowsum)),
)
