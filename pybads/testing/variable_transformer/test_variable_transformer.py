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
    Y2 = np.ones((10, D)) * 1.
    assert np.all(np.isclose(Y, Y2, atol=1e-04))
    
    X = np.ones((10, D)) * -10
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * -1.
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
    X2 = np.ones((10, D)) * 3.
    assert np.all(np.isclose(X, X2))

def test_inverse_within_negative():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -0.4
    X = parameter_transformer.inverse_transf(Y)
    X2 = np.ones((10, D)) * -4.
    assert np.all(np.isclose(X, X2))
    
def test_inverse_on_boundaries():
    parameter_transformer = VariableTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -1
    X = parameter_transformer.inverse_transf(Y)
    X2 = np.ones((10, D)) * -10.
    assert np.all(np.isclose(X, X2))

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
