import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for key in sorted(k[:-2] for k in a.files if k.endswith("_X")):
    same = all(
        a[key + s].shape == b[key + s].shape
        and np.array_equal(a[key + s], b[key + s], equal_nan=True)
        for s in ("_X", "_Y", "_res")
    )
    print(
        f"{key}: rows {a[key + '_X'].shape[0]} vs {b[key + '_X'].shape[0]}, "
        f"every evaluated point, value and result identical: {same}"
    )
