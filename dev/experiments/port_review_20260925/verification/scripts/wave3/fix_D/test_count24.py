import numpy as np
import pytest

from pybads import BADS

D = 3


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _make_bads(fun=_sphere, **options):
    opts = {"display": "off", "max_fun_evals": 80, "random_seed": 3}
    opts.update(options)
    return BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    )


def test_poll_points_on_the_search_mesh(monkeypatch):
    """The poll steps along LTMADS directions in units of the search mesh
    size (`poll_mads_2n`; MATLAB BADS's poll steps along one coordinate at a
    time): at default options each poll point is a step of `mesh_size` along
    one coordinate, tilted along the others, and the points lie on the
    search mesh when the incumbent does."""
    import pybads.bads.bads as bads_module

    bads = _make_bads(max_fun_evals=100)
    original_check = bads_module.contraints_check
    polls = []

    def check(U, lb, ub, tol_mesh, function_logger, proj, non_box_cons):
        if not proj:  # the poll's points, before the check
            polls.append(
                (
                    bads.u.copy(),
                    U.copy(),
                    bads.optim_state["mesh_size"],
                    bads.optim_state["search_mesh_size"],
                )
            )
        return original_check(
            U, lb, ub, tol_mesh, function_logger, proj, non_box_cons
        )

    def on_mesh(u, search_mesh_size):
        units = u / search_mesh_size
        return np.allclose(units, np.round(units), rtol=0, atol=1e-6)

    monkeypatch.setattr(bads_module, "contraints_check", check)
    bads.optimize()
    n_tilted, n_on_mesh = 0, 0
    for u, U, mesh_size, search_mesh_size in polls:
        steps = U - u
        assert np.allclose(np.max(np.abs(steps), axis=1), mesh_size)
        n_tilted += np.sum(
            np.sum(np.abs(steps) > search_mesh_size / 2, axis=1) > 1
        )
        if on_mesh(u, search_mesh_size):
            n_on_mesh += 1
            assert on_mesh(U, search_mesh_size)
    print(
        "COUNTS",
        len(polls),
        n_tilted,
        n_on_mesh,
        sum(len(p[1]) for p in polls),
        [np.log2(p[2]) for p in polls],
    )
