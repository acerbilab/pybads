from common import *

from pybads.variable_transformer import VariableTransformer

rng = np.random.default_rng(1)
for k in range(400):
    lbv = -1e10 * rng.uniform(1, 9.99)
    plb = rng.uniform(-3, 0)
    pub = plb + rng.uniform(0.1, 3)
    try:
        VariableTransformer(
            1,
            np.array([[lbv]]),
            np.array([[-lbv]]),
            np.array([[plb]]),
            np.array([[pub]]),
            np.full((1, 1), np.nan),
        )
    except ValueError:
        print("refused:", lbv, -lbv, plb, pub)
        x0 = 0.5 * (plb + pub)
        trycall(
            "BADS same bounds",
            lambda: BADS(
                quad,
                np.array([[x0, 0.0]]),
                np.array([[lbv, -1.0]]),
                np.array([[-lbv, 1.0]]),
                np.array([[plb, -0.5]]),
                np.array([[pub, 0.5]]),
                options={"display": "off", "random_seed": 0},
            ).var_transf.lb,
        )
        break
