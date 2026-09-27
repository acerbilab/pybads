p = "pybads/function_logger/function_logger.py"
s = open(p).read()
old = """            else:
                fval_orig = fun_res
                fsd = None

            if isinstance(fval_orig, np.ndarray):
                # fval_orig can only be an array with size 1 since we support just single evaluation
                fval_orig = fval_orig.item()
            if isinstance(fsd, np.ndarray):
                # fsd can only be an array with size 1 since we support just single evaluation
                fsd = fsd.item()
        except Exception as err:
"""
new = """            else:
                fval_orig = fun_res
                fsd = None
        except Exception as err:
"""
assert old in s
s = s.replace(old, new)
old = '''        # if fval is an array with only one element, extract that element
        if not np.isscalar(fval_orig) and np.size(fval_orig) == 1:
            fval_orig = np.array(fval_orig).flat[0]

        # Check function value
        if np.any(
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(fval_orig)))

        # Check returned function SD
        if self.he_noise_flag and (
            not np.isfinite(fsd) or not np.isreal(fsd) or fsd <= 0.0
        ):
'''
new = '''        # A value or an SD in an array or a list of one element is taken as
        # that element. The conversion and the checks, the logger's own, stay
        # out of the try above, whose note is for the target's errors, and
        # come before anything is recorded
        fval_orig = _as_scalar(fval_orig)
        if self.he_noise_flag:
            fsd = _as_scalar(fsd)

        # Check function value
        if not _is_finite_real_scalar(fval_orig):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(fval_orig)))

        # Check returned function SD
        if self.he_noise_flag and not (
            _is_finite_real_scalar(fsd) and fsd > 0.0
        ):
'''
assert old in s
s = s.replace(old, new)
s = (
    s.rstrip("\n")
    + '''


def _as_scalar(value):
    """Return the element of an array or a sequence of one element, and any
    other value unchanged."""
    if np.isscalar(value):
        return value
    try:
        array = np.asarray(value)
    except ValueError:  # A ragged sequence
        return value
    return array.item() if array.size == 1 else value


def _is_finite_real_scalar(value):
    """Whether ``value`` is a finite scalar of a boolean, integer or floating
    type: a value of a complex type is not real, whatever its imaginary part,
    as for MATLAB's ``isreal``."""
    return (
        np.isscalar(value)
        and np.asarray(value).dtype.kind in "biuf"
        and bool(np.isfinite(value))
    )
'''
)
open(p, "w").write(s)
