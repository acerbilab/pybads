import warnings

import numpy as np
import pytest

from pybads.variable_transformer import VariableTransformer

D = 3


def test_init_no_lower_bounds():
    with pytest.raises(ValueError):
        VariableTransformer(D=D)


def test_init_lower_bounds():
    with pytest.raises(ValueError):
        VariableTransformer(D=D, lower_bounds=np.ones((1, D)))


def test_init_no_upper_bounds():
    with pytest.raises(ValueError):
        VariableTransformer(D=D)


def test_init_upper_bounds():
    with pytest.raises(ValueError):
        VariableTransformer(D=D, upper_bounds=np.ones((1, D)))


def test_init_bounds_check():
    with pytest.raises(ValueError):
        VariableTransformer(
            D=D,
            lower_bounds=np.ones((1, D)) * 3,
            upper_bounds=np.ones((1, D)) * 2,
        )
    with pytest.raises(ValueError):
        VariableTransformer(
            D=D,
            lower_bounds=np.ones((1, D)) * 0,
            upper_bounds=np.ones((1, D)) * 10,
            plausible_lower_bounds=np.ones((1, D)) * -1,
        )
    with pytest.raises(ValueError):
        VariableTransformer(
            D=D,
            lower_bounds=np.ones((1, D)) * 0,
            upper_bounds=np.ones((1, D)) * 10,
            plausible_upper_bounds=np.ones((1, D)) * 11,
        )
    with pytest.raises(ValueError):
        VariableTransformer(
            D=D,
            lower_bounds=np.ones((1, D)) * 0,
            upper_bounds=np.ones((1, D)) * 10,
            plausible_lower_bounds=np.ones((1, D)) * 100,
            plausible_upper_bounds=np.ones((1, D)) * -20,
        )


def test_init_():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)),
        upper_bounds=np.ones((1, D)) * 2,
    )
    assert np.all(parameter_transformer.apply_log_t == 0)


def test_direct_transform__within_positive():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * 0.3
    assert np.all(np.isclose(Y, Y2, atol=1e-04))


def test_direct_transform__on_boundaries():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 10
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * 1.0
    assert np.all(np.isclose(Y, Y2, atol=1e-04))

    X = np.ones((10, D)) * -10
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * -1.0
    assert np.all(np.isclose(Y, Y2, atol=1e-04))


def test_direct_transform_within_negative():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * -0.4
    assert np.all(np.isclose(Y, Y2))


def test_inverse_within():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * 0.3
    X = parameter_transformer.inverse_transf(Y)
    X2 = np.ones((10, D)) * 3.0
    assert np.all(np.isclose(X, X2))


def test_inverse_within_negative():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -0.4
    X = parameter_transformer.inverse_transf(Y)
    X2 = np.ones((10, D)) * -4.0
    assert np.all(np.isclose(X, X2))


def test_inverse_on_boundaries():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -1
    X = parameter_transformer.inverse_transf(Y)
    X2 = np.ones((10, D)) * -10.0
    assert np.all(np.isclose(X, X2))


def test_1D_transform():
    """Test 1D variable transformation"""
    parameter_transformer = VariableTransformer(
        D=1,
        lower_bounds=np.array([[-10]]),
        upper_bounds=np.array([[10]]),
    )
    X = np.array([[3]])
    Y = parameter_transformer(X)
    Y2 = np.array([[0.3]])
    assert np.all(np.isclose(Y, Y2, atol=1e-04))


def test_inverse_min_space():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -500
    X = parameter_transformer.inverse_transf(Y)
    assert np.all(X == np.ones((1, D)) * -10)


def test_inverse_max_space():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * 3000
    X = parameter_transformer.inverse_transf(Y)
    assert np.all(X == np.ones((10, D)) * 10)


def test_transform_inverse():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 0.05
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse_transf(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))

    U = np.ones((10, D)) * 0.2
    X = parameter_transformer.inverse_transf(U)
    U2 = parameter_transformer(X)
    assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))


def test_transform_inverse_largeN():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    X = np.ones((10 ^ 6, D)) * 0.4
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse_transf(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))


@pytest.mark.parametrize(
    "bounds, log_t",
    [
        ((-9.53e10, 9.53e10, -2.06, -0.74), False),
        ((1e-3, 2e9, 1.0, 10.0), True),
    ],
    ids=["linear", "log"],
)
def test_bounds_of_large_magnitude(bounds, log_t):
    """The transform's self-test allows an error relative to a bound of large
    magnitude, so that rounding alone does not refuse the bound: the error
    of the round trip at `ub` is one or a few units in the last place, 1.5e-5
    and 2.4e-6 here, above an absolute 1e-6."""
    lb, ub, plb, pub = (np.array([[bound]]) for bound in bounds)
    parameter_transformer = VariableTransformer(
        1, lb, ub, plb, pub, np.full((1, 1), np.nan)
    )
    assert parameter_transformer.apply_log_t.item() == log_t
    np.testing.assert_allclose(parameter_transformer.plb, -1.0, atol=1e-12)
    np.testing.assert_allclose(parameter_transformer.pub, 1.0, atol=1e-12)
    X = parameter_transformer.inverse_transf(parameter_transformer.ub)
    np.testing.assert_allclose(X, ub, rtol=1e-12)


@pytest.mark.parametrize(
    "lb, ub",
    [((1.0, -5.0), (1000.0, np.inf)), ((1.0, -1e3), (1000.0, 1e3))],
    ids=["infinite", "finite"],
)
def test_log_beside_linear_gives_no_overflow_warning(lb, ub):
    """The inverse of a mix of log-scaled and linear variables takes the
    exponential of every variable and masks out the linear ones. That of a
    linear variable overflows above about 709, at a bound of 1e3 or at the
    1/sqrt(eps) where the self-test puts an infinite one, harmlessly: it
    gives no warning, and the values are those of the transform."""
    lb, ub = np.array([lb]), np.array([ub])
    plb, pub = np.array([[2.0, -1.0]]), np.array([[500.0, 1.0]])
    x = np.array([[10.0, 800.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parameter_transformer = VariableTransformer(2, lb, ub, plb, pub)
        X = parameter_transformer.inverse_transf(parameter_transformer(x))
    assert np.all(parameter_transformer.apply_log_t == [[True, False]])
    # The log-scaled variable maps [2, 500] to [-1, 1] and [1, 1000] to
    # [-log(1000) / log(250), log(1000) / log(250)]; the linear one is
    # unchanged
    r = np.log(1000) / np.log(250)
    np.testing.assert_allclose(parameter_transformer.lb, [[-r, lb[0, 1]]])
    np.testing.assert_allclose(parameter_transformer.ub, [[r, ub[0, 1]]])
    np.testing.assert_allclose(parameter_transformer.plb, [[-1.0, -1.0]])
    np.testing.assert_allclose(parameter_transformer.pub, [[1.0, 1.0]])
    np.testing.assert_allclose(X, x, rtol=1e-12)
