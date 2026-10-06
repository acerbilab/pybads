from common import *

from pybads.variable_transformer import VariableTransformer

trycall(
    "apply_log_t scalar",
    lambda: VariableTransformer(
        2,
        np.array([[1.0, 1.0]]),
        np.array([[100.0, 100.0]]),
        np.array([[2.0, 2.0]]),
        np.array([[50.0, 50.0]]),
        0,
    ).apply_log_t,
)
trycall(
    "python float bounds",
    lambda: VariableTransformer(2, 1.0, 100.0, 2.0, 50.0).apply_log_t,
)
trycall(
    "np.float64 bounds",
    lambda: VariableTransformer(
        2,
        np.float64(1.0),
        np.float64(100.0),
        np.float64(2.0),
        np.float64(50.0),
    ).apply_log_t,
)
