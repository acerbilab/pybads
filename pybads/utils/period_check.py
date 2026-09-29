import numpy as np


def period_check(u, lb, ub, periodic_vars):
    """
    Wrap the periodic coordinates of ``u`` into their period, ``[lb, ub)``,
    as MATLAB BADS's ``periodCheck``.

    A periodic variable's period is the width of its bounds, ``ub - lb``,
    and ``lb`` and ``ub`` are the same point of it: a coordinate beyond
    either bound comes back in from the other one.

    Parameters
    ----------
    u : np.ndarray
        The points, one per row (or one point, as a 1-D array), in the
        coordinates of ``lb`` and ``ub``.
    lb, ub : np.ndarray
        The lower and upper bounds, of shape ``(1, D)`` or ``(D,)``.
    periodic_vars : np.ndarray or None
        The boolean mask of the periodic variables, of shape ``(1, D)`` or
        ``(D,)``, or ``None`` when no variable is periodic.

    Returns
    -------
    u_wrapped : np.ndarray
        ``u`` itself when no variable is periodic; otherwise a copy of
        ``u`` whose periodic coordinates lie in ``[lb, ub)``, and the
        others as they are.
    """
    if periodic_vars is None or not np.any(periodic_vars):
        return u
    mask = np.ravel(periodic_vars).astype(bool)
    lb_p = np.ravel(lb)[mask]
    ub_p = np.ravel(ub)[mask]
    u_wrapped = np.array(u, dtype=float, copy=True)
    # A view of the copy: a 1-D point is one row
    rows = np.atleast_2d(u_wrapped)
    wrapped = lb_p + np.mod(rows[:, mask] - lb_p, ub_p - lb_p)
    # np.mod returns the period itself for a tiny negative difference, and
    # lb + a shift within rounding of the period can round to ub: both are
    # the point lb
    rows[:, mask] = np.where(wrapped >= ub_p, lb_p, wrapped)
    return u_wrapped
