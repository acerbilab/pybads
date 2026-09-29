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
    X = np.ones((10**6, D)) * 0.4
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


@pytest.mark.parametrize(
    "plausible", [True, False], ids=["plausible", "plausible_omitted"]
)
def test_integer_bounds_are_taken_as_floats(plausible):
    """The log of a log-scaled variable's bounds is written into the
    transformer's copies of the bounds, which are floats, so that integer
    bounds give the transform of the same values as floats."""
    bounds = [[[1, -10]], [[1000, 10]]]
    if plausible:
        bounds += [[[2, -5]], [[500, 5]]]
    transformers = [
        VariableTransformer(2, *(np.array(b, dtype=dtype) for b in bounds))
        for dtype in (int, float)
    ]
    x = np.array([[10.0, 3.0], [200.0, -7.0]])
    int_transformer, float_transformer = transformers
    assert np.all(float_transformer.apply_log_t == [[True, False]])
    assert np.all(int_transformer.apply_log_t == float_transformer.apply_log_t)
    names = ["lb", "ub", "plb", "pub"]
    names += ["orig_" + name for name in names]
    for name in names:
        np.testing.assert_array_equal(
            getattr(int_transformer, name), getattr(float_transformer, name)
        )
    u = float_transformer(x)
    np.testing.assert_array_equal(int_transformer(x), u)
    np.testing.assert_array_equal(
        int_transformer.inverse_transf(u), float_transformer.inverse_transf(u)
    )
    assert all(getattr(int_transformer, name).dtype == float for name in names)


@pytest.mark.parametrize(
    "bounds",
    [
        (np.array([1.0, -10.0]), np.array([1000.0, 10.0])),
        ([1.0, -10.0], [1000.0, 10.0]),
    ],
    ids=["1d_arrays", "lists"],
)
def test_bounds_of_d_elements_are_rows(bounds):
    """Bounds of D elements given as 1-D arrays or lists give the transform
    of the same bounds given as arrays of shape (1, D)."""
    rows = [np.atleast_2d(np.asarray(b, dtype=float)) for b in bounds]
    transformer = VariableTransformer(2, *bounds)
    reference = VariableTransformer(2, *rows)
    for name in ["lb", "ub", "plb", "pub", "orig_lb", "orig_ub"]:
        assert getattr(transformer, name).shape == (1, 2)
        np.testing.assert_array_equal(
            getattr(transformer, name), getattr(reference, name)
        )
    x = np.array([[10.0, 3.0]])
    np.testing.assert_array_equal(transformer(x), reference(x))


@pytest.mark.parametrize(
    "scalar", [np.float64, float, int], ids=["numpy", "float", "int"]
)
def test_scalar_bounds_are_replicated(scalar):
    """Scalar bounds, NumPy's or Python's, stand for the same bound in each
    dimension, the plausible bounds omitted too."""
    transformer = VariableTransformer(D, scalar(1), scalar(1000))
    reference = VariableTransformer(
        D, np.ones((1, D)), np.full((1, D), 1000.0)
    )
    assert np.all(transformer.apply_log_t)
    for name in ["lb", "ub", "plb", "pub", "orig_plb", "orig_pub"]:
        np.testing.assert_array_equal(
            getattr(transformer, name), getattr(reference, name)
        )


@pytest.mark.parametrize("apply_log_t", [True, False, np.nan])
def test_scalar_apply_log_t_applies_to_every_variable(apply_log_t):
    """A scalar `apply_log_t` applies to every variable, NaN leaving the
    choice to the bounds, as the default does."""
    bounds = (np.ones((1, D)), np.full((1, D), 1000.0))
    transformer = VariableTransformer(D, *bounds, apply_log_t=apply_log_t)
    expected = True if np.isnan(apply_log_t) else apply_log_t
    assert transformer.apply_log_t.shape == (1, D)
    assert np.all(transformer.apply_log_t == expected)


@pytest.mark.parametrize(
    "lower_bounds",
    [np.zeros((1, D + 1)), np.zeros((D, 1)), np.zeros(D - 1)],
    ids=["too_long", "column", "too_short"],
)
def test_bounds_of_another_size_are_refused(lower_bounds):
    """A bound that is neither a scalar nor an array of D elements in a row
    is refused with a message that names it."""
    with pytest.raises(ValueError, match="lower_bounds needs to be"):
        VariableTransformer(D, lower_bounds, np.ones((1, D)))


@pytest.mark.parametrize(
    "lower_bounds",
    ["1", ["1", "2", "3"], [[1.0], [1.0, 2.0]], object()],
    ids=["string", "strings", "ragged", "object"],
)
def test_bounds_that_are_not_numbers_are_refused(lower_bounds):
    """A bound that is not a number, a string included, which NumPy would
    convert, is refused with a message that names it."""
    with pytest.raises(ValueError, match="lower_bounds needs to be a number"):
        VariableTransformer(D, lower_bounds, np.full((1, D), 10.0))


def _transformer_with_fixed_values():
    """A transform of D variables, the first on a log scale, within points
    of five variables whose first and fourth are fixed."""
    return VariableTransformer(
        D,
        np.array([0.01, -10.0, -10.0]),
        np.array([100.0, 10.0, 10.0]),
        np.array([0.1, -5.0, -5.0]),
        np.array([10.0, 5.0, 5.0]),
        fixed_values=np.array([2.0, np.nan, np.nan, -1.5, np.nan]),
    )


@pytest.mark.parametrize("shape", [(D,), (1, D), (4, D)])
def test_fixed_values_are_put_back_by_the_inverse(shape):
    """With `fixed_values`, the inverse returns points of all the
    variables, the fixed ones at their values and the others as the
    transform without them gives them, and the transform takes such points
    and leaves their fixed coordinates out, for one point or several."""
    transformer = _transformer_with_fixed_values()
    without = VariableTransformer(
        D,
        transformer.orig_lb,
        transformer.orig_ub,
        transformer.orig_plb,
        transformer.orig_pub,
    )
    u = np.random.default_rng(0).uniform(-1, 1, shape)
    x = transformer.inverse_transf(u)
    assert x.shape == shape[:-1] + (5,)
    assert np.all(x[..., [0, 3]] == [2.0, -1.5])
    x_free = x[..., [1, 2, 4]]
    assert np.array_equal(x_free, without.inverse_transf(u))
    assert np.array_equal(transformer(x), without(x_free))
    assert np.allclose(transformer(x), u)


@pytest.mark.parametrize(
    "fixed_values",
    [
        np.array([2.0, np.nan, np.nan]),
        np.array([2.0, np.nan, np.nan, np.nan, np.nan]),
        np.array([np.inf, np.nan, np.nan, np.nan]),
        np.array([[2.0, np.nan], [np.nan, np.nan]]),
        "2, nan, nan, nan",
    ],
    ids=["too_few_free", "too_many_free", "infinite", "2d", "string"],
)
def test_fixed_values_are_refused_unless_nan_at_the_d_variables(fixed_values):
    """`fixed_values` is a row, NaN at exactly D variables and finite at the
    others."""
    with pytest.raises(ValueError, match="fixed_values needs to be a row"):
        VariableTransformer(
            D, -np.ones(D), np.ones(D), fixed_values=fixed_values
        )
