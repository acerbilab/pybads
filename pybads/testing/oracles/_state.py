"""Snapshot codec and rebuilders for the oracles.

A *snapshot* is the state of a PyBADS run at the start of a search or poll
step, stored as plain arrays: a ``.npz`` file (read with ``allow_pickle=False``)
holding every array, and a JSON sidecar holding the scalars, strings,
booleans, ``None``, lists and the tree structure, with the string marker
``"@@npz:<key>"`` where an array belongs. Nothing in a snapshot depends on
the layout of a Python class, so a fixture survives refactors and can be
read by another implementation, such as a harness that runs MATLAB BADS on
the same states.

A snapshot holds the function logger's filled rows, the ``optim_state``
(through the codec), the GP's training data, hyperparameters, bounds,
priors and ``temporary_data``, the effective options, the incumbent, the
hedge's gains, the state of the run's generator (PCG64's state is plain
JSON), the inputs that some oracles take beside the state, a candidate
set, and the reference outputs of the oracles.

:func:`snapshot_from_bads` takes the state out of a live :class:`BADS`;
``build_*`` rebuild the objects through the public constructors (a gpyreg
``GP`` from its data and hyperparameters, a ``FunctionLogger`` filled row by
row, ``Options`` from the option files); :func:`save_snapshot` and
:func:`load_snapshot` do the files. The fixture generator
(``dev/scripts/make_oracle_fixtures.py``) and the tests share this module,
so that the references and the checked outputs are computed from states
rebuilt the same way.
"""

import copy
import json
import numbers
import types
from pathlib import Path

import gpyreg as gpr
import numpy as np

import pybads.bads
from pybads.bads.bads import BADS
from pybads.bads.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _meanfun_name_to_mean_function,
)
from pybads.bads.options import Options
from pybads.function_logger import FunctionLogger

from ._recipes import NON_BOX_CONS

ARRAY_MARKER = "@@npz:"
FLOAT_TAG = "@float"
CALLABLE_TAG = "@callable"
OPTION_FILES = Path(pybads.bads.__file__).resolve().parent / "option_configs"


# --------------------------------------------------------------------------
# Codec: trees of dict / list / tuple / scalar / ndarray / None
# --------------------------------------------------------------------------


def _encode_float(x):
    if np.isfinite(x):
        return float(x)
    # Strict JSON has no inf or nan: they are tagged.
    return {FLOAT_TAG: "nan" if np.isnan(x) else ("inf" if x > 0 else "-inf")}


def encode(obj, prefix, arrays):
    """Encode ``obj`` into a JSON-able tree, its arrays into ``arrays``.

    Raises ``TypeError`` on a type it does not know, so that a new kind of
    state is added to the codec deliberately and never dropped. Tuples
    become lists.
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError(f"cannot encode {prefix}: an object array")
        arrays[prefix] = obj
        return ARRAY_MARKER + prefix
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, (bool, str)) or obj is None:
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return _encode_float(float(obj))
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"{prefix}: non-string dict key {k!r}")
            out[k] = encode(v, f"{prefix}/{k}", arrays)
        return out
    if isinstance(obj, (list, tuple)):
        return [encode(v, f"{prefix}/{i}", arrays) for i, v in enumerate(obj)]
    raise TypeError(f"cannot encode {prefix}: {type(obj).__name__}")


def decode(tree, arrays):
    """The inverse of :func:`encode` (tuples come back as lists)."""
    if isinstance(tree, str) and tree.startswith(ARRAY_MARKER):
        key = tree[len(ARRAY_MARKER) :]
        if key not in arrays:
            raise KeyError(f"array marker {tree!r} has no array in the .npz")
        return np.array(arrays[key])
    if isinstance(tree, dict):
        if set(tree) == {FLOAT_TAG}:
            return float(tree[FLOAT_TAG])
        return {k: decode(v, arrays) for k, v in tree.items()}
    if isinstance(tree, list):
        return [decode(v, arrays) for v in tree]
    return tree


# --------------------------------------------------------------------------
# Snapshot extraction
# --------------------------------------------------------------------------


def _encode_options(options, arrays):
    """The effective options; a callable becomes ``{"@callable": name}``
    (the rebuild keeps the option files' callable), and the set of the
    names the user set is left out."""
    out = {}
    for key in options:
        if key == "useroptions":
            continue
        value = options[key]
        if callable(value):
            out[key] = {CALLABLE_TAG: getattr(value, "__name__", "?")}
        else:
            out[key] = encode(value, f"options/{key}", arrays)
    return out


def snapshot_from_bads(bads, gp, meta):
    """Take the state of ``bads`` at the start of a search or poll step
    whose GP is ``gp``. Returns ``(arrays, tree)`` for
    :func:`save_snapshot`; the tree has empty ``inputs``, ``cand`` and
    ``ref`` entries, which the generator fills."""
    arrays = {}
    optim_state = bads.optim_state
    fl = bads.function_logger
    n = fl.X_max_idx + 1

    tree = {"meta": dict(meta)}
    tree["options"] = _encode_options(bads.options, arrays)
    tree["optim_state"] = encode(
        copy.deepcopy(optim_state), "optim_state", arrays
    )
    tree["incumbent"] = encode(
        {
            "u": np.array(bads.u, dtype=float),
            "yval": bads.yval,
            "fval": bads.fval,
            "fsd": bads.fsd,
            "mesh_size": bads.mesh_size,
            "search_mesh_size": bads.search_mesh_size,
            "sufficient_improvement": bads.sufficient_improvement,
            "lower_bounds": np.array(bads.lower_bounds, dtype=float),
            "upper_bounds": np.array(bads.upper_bounds, dtype=float),
            "gamma_uncertain_interval": bads.gamma_uncertain_interval,
        },
        "incumbent",
        arrays,
    )
    tree["logger"] = encode(
        {
            "D": int(fl.D),
            "noise_flag": bool(fl.noise_flag),
            "uncertainty_handling_level": int(fl.uncertainty_handling_level),
            "cache_size": int(fl.X.shape[0]),
            "Xn": int(fl.Xn),
            "X_max_idx": int(fl.X_max_idx),
            "func_count": int(fl.func_count),
            "cache_count": int(fl.cache_count),
            "Y_max": float(fl.Y_max),
            "X_orig": np.array(fl.X_orig[:n]),
            "Y_orig": np.array(fl.Y_orig[:n]),
            "X": np.array(fl.X[:n]),
            "Y": np.array(fl.Y[:n]),
            "S": np.array(fl.S[:n]) if fl.noise_flag else None,
            "n_evals": np.array(fl.n_evals[:n]),
            "X_flag": np.array(fl.X_flag[:n]),
        },
        "logger",
        arrays,
    )
    tree["gp"] = encode(
        {
            "X": np.array(gp.X),
            "y": np.array(gp.y),
            "s2": None if gp.s2 is None else np.array(gp.s2),
            "hyp": gp.get_hyperparameters(as_array=True),
            "bounds": gp.get_bounds(),
            "priors": gp.get_priors(),
            "temporary_data": copy.deepcopy(gp.temporary_data),
            "cov_fun": optim_state["gp_cov_fun"],
            "mean_fun": optim_state["gp_mean_fun"],
            "noise_fun": list(optim_state["gp_noisefun"]),
        },
        "gp",
        arrays,
    )
    hedge = bads.search_es_hedge
    tree["hedge"] = (
        None
        if hedge is None
        else encode(
            {
                "g": np.array(hedge.g),
                "count": hedge.count,
                "chosen_hedge": np.array(hedge.chosen_hedge),
            },
            "hedge",
            arrays,
        )
    )
    tree["rng_state"] = encode(bads.rng.bit_generator.state, "rng", arrays)
    tree["inputs"] = {}
    tree["cand"] = {}
    tree["ref"] = {}
    return arrays, tree


# --------------------------------------------------------------------------
# Rebuilders
# --------------------------------------------------------------------------


def build_options(D, user_options, effective):
    """The options of a run: the option files evaluated at ``D`` with the
    user's options, as ``BADS`` loads them, then the run's effective value
    of every option but the callables, which keep the files' value."""
    options = Options(
        str(OPTION_FILES / "basic_bads_options.ini"),
        evaluation_parameters={"D": D},
        user_options=user_options,
    )
    options.load_options_file(
        str(OPTION_FILES / "advanced_bads_options.ini"),
        evaluation_parameters={"D": D},
    )
    for key, value in effective.items():
        if key not in options:
            raise ValueError(f"the snapshot's option {key!r} does not exist")
        if isinstance(value, dict) and set(value) == {CALLABLE_TAG}:
            if not callable(options[key]):
                raise ValueError(f"option {key!r} is no longer a callable")
            continue
        options[key] = value
    return options


def build_transformer(optim_state, options, D):
    """The run's ``VariableTransformer``, built by ``BADS``'s own method
    from the original bounds."""
    stand_in = types.SimpleNamespace(
        options=options,
        D=D,
        lower_bounds=optim_state["lb_orig"],
        upper_bounds=optim_state["ub_orig"],
        plausible_lower_bounds=optim_state["plb_orig"],
        plausible_upper_bounds=optim_state["pub_orig"],
    )
    return BADS._variable_transformer_(stand_in)


def _missing_fun(x):
    raise RuntimeError("a rebuilt FunctionLogger has no target")


def build_logger(d, variable_transformer):
    """A ``FunctionLogger`` holding the snapshot's rows, without a target."""
    fl = FunctionLogger(
        _missing_fun,
        d["D"],
        d["noise_flag"],
        d["uncertainty_handling_level"],
        cache_size=d["cache_size"],
        variable_transformer=variable_transformer,
    )
    n = d["X_max_idx"] + 1
    for name in ("X_orig", "Y_orig", "X", "Y", "n_evals", "X_flag"):
        getattr(fl, name)[:n] = d[name]
    if d["noise_flag"]:
        fl.S[:n] = d["S"]
    fl.Xn = d["Xn"]
    fl.X_max_idx = d["X_max_idx"]
    fl.func_count = d["func_count"]
    fl.cache_count = d["cache_count"]
    fl.Y_max = d["Y_max"]
    return fl


def _as_tuples(tree):
    """The bounds and priors of a GP as gpyreg takes them: tuples, from the
    lists that the codec gives back."""
    return {
        k: None if v is None else tuple(_as_tuples_inner(x) for x in v)
        for k, v in tree.items()
    }


def _as_tuples_inner(x):
    return tuple(x) if isinstance(x, list) else x


def new_gp(D, cov_fun, mean_fun, noise_fun):
    """A gpyreg ``GP`` without data, with the covariance, mean and noise
    functions that ``init_and_train_gp`` gives a run's GP from
    ``optim_state["gp_cov_fun"]``, ``["gp_mean_fun"]`` and
    ``["gp_noisefun"]``."""
    noise = gpr.noise_functions.GaussianNoise(
        constant_add=noise_fun[0] == 1,
        user_provided_add=noise_fun[1] == 1,
        scale_user_provided=noise_fun[1] == 2,
        rectified_linear_output_dependent_add=noise_fun[2] == 1,
    )
    return gpr.GP(
        D=D,
        covariance=_cov_identifier_to_covariance_function(cov_fun),
        mean=_meanfun_name_to_mean_function(mean_fun),
        noise=noise,
    )


def build_gp(d):
    """A gpyreg ``GP`` with the snapshot's functions, bounds, priors, data
    and hyperparameters, its posterior computed from them, and its
    ``temporary_data``."""
    X = np.array(d["X"])
    gp = new_gp(X.shape[1], d["cov_fun"], d["mean_fun"], d["noise_fun"])
    gp.set_bounds(_as_tuples(d["bounds"]))
    gp.set_priors(_as_tuples(d["priors"]))
    gp.update(
        X_new=X,
        y_new=np.array(d["y"]),
        s2_new=None if d["s2"] is None else np.array(d["s2"]),
        hyp=np.atleast_2d(np.array(d["hyp"])),
    )
    gp.temporary_data = copy.deepcopy(d["temporary_data"])
    return gp


def build_state(snap):
    """Rebuild every object of a decoded snapshot, on a private copy.

    Returns a dict with ``options``, ``optim_state``, ``var_transf``,
    ``logger``, ``gp``, ``incumbent``, ``hedge``, ``rng_state``,
    ``inputs``, ``cand``, ``non_box_cons``, ``meta`` and ``ref``.
    """
    snap = copy.deepcopy(snap)
    meta = snap["meta"]
    D = int(meta["D"])
    options = build_options(D, meta["user_options"], snap["options"])
    optim_state = snap["optim_state"]
    var_transf = build_transformer(optim_state, options, D)
    name = meta.get("non_box_cons")
    return {
        "options": options,
        "optim_state": optim_state,
        "var_transf": var_transf,
        "logger": build_logger(snap["logger"], var_transf),
        "gp": build_gp(snap["gp"]),
        "incumbent": snap["incumbent"],
        "hedge": snap["hedge"],
        "rng_state": snap["rng_state"],
        "inputs": snap["inputs"],
        "cand": snap["cand"],
        "non_box_cons": None if name is None else NON_BOX_CONS[name],
        "meta": meta,
        "ref": snap["ref"],
    }


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def snapshot_files(path):
    """The ``.npz`` and ``.json`` files of the snapshot at ``path``."""
    path = Path(path)
    return (
        path.parent / (path.name + ".npz"),
        path.parent / (path.name + ".json"),
    )


def save_snapshot(path, arrays, tree):
    """Write ``<path>.npz`` and ``<path>.json`` (strict JSON, LF)."""
    npz, js = snapshot_files(path)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    with open(js, "w", encoding="utf-8", newline="\n") as f:
        json.dump(tree, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")


def load_arrays(path):
    """The arrays of the snapshot at ``path``, by key."""
    npz, _ = snapshot_files(path)
    with np.load(npz, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_tree(path):
    """The JSON tree of the snapshot at ``path``, not decoded."""
    _, js = snapshot_files(path)
    return json.loads(js.read_text(encoding="utf-8"))


def load_snapshot(path):
    """Read a snapshot back into a decoded tree (arrays in place)."""
    return decode(load_tree(path), load_arrays(path))


def snapshot_names(fixtures_dir):
    """The names of the snapshots in a directory, sorted."""
    return sorted(
        p.name[: -len(".json")] for p in Path(fixtures_dir).glob("*.json")
    )
