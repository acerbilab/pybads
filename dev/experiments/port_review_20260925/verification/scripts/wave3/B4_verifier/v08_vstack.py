"""K7: what np.vstack(u_poll, u_poll_new) does if reached."""
import numpy as np
import vhdr

print("numpy", np.__version__)
a = np.zeros((2, 3))
b = np.ones((2, 3))
try:
    print(np.vstack(a, b))
except Exception as e:
    print(type(e).__name__, e)
