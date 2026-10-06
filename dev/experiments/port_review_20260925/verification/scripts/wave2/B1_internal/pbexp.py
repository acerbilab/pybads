from common import *

for x in [0.0, 0.005, 0.02, 1.0, 1.99, 2.5]:
    b = BADS(
        quad,
        np.array([x, 5.0]),
        np.array([0.0, 0.0]),
        np.array([10.0, 10.0]),
        np.array([2.0, 2.0]),
        np.array([8.0, 8.0]),
        options={"display": "off", "random_seed": 0},
    )
    print(
        f"x0={x:6.3f} -> x0 after check {b.x0.ravel()[0]:.4f}, plb {b.optim_state['plb_orig'].ravel()[0]:.4f}, u0 {b.u[0]:.4f}"
    )
