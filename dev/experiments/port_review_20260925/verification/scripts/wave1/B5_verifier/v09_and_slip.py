"""F9 (internal) / F11 (comparison): value of the search's add condition."""
import numpy as np

n_try = 3
for size in (0, 3):
    for count in (-1, 0, 2, 3, 50):
        v = size > 0 & count < n_try
        print(
            f"size={size} count={count}: expression -> {v};  'and' form -> {size > 0 and count < n_try}"
        )
