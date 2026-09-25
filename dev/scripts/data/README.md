# Data of the real-data benchmark targets

The archives read by the `timing` and `multisensory_s1` targets of
`dev/scripts/benchmark_targets.py`, and the reference minima of those
targets.

The two `.npz` files are plain NumPy archives: every array is float64 or
int64, and they load with `np.load(path, allow_pickle=False)`. They are
copies of the files of the same names in PyVBMC's `dev/scripts/data/`
(`acerbilab/pyvbmc`, as of commit `66d9b364`), which exports them with its
`dev/scripts/export_benchflow_data.py` from the lab-private
[benchflow](https://github.com/acerbilab/benchflow) repository.
`provenance.json`, copied with them, records the benchflow commit and the
SHA-256 of each source `.mat` file. The `.mat` files are not in either
repository.

## `timing.npz`

Bayesian time-interval reproduction (Acerbi, Wolpert & Vijayakumar 2012),
one subject of Experiment 3 (uniform interval distribution; subject 2 in
the paper's numbering), 1512 trials, the "Bayesian timing" problem of the
2020 noisy-VBMC paper.
Source: `benchflow/tasks/task_info/timing/timing.mat`.

| key | dtype, shape | content |
| --- | --- | --- |
| `stim_index` | int64 (1512,) | 0-based index of the trial's interval into `stimuli` (252 trials each) |
| `response` | float64 (1512,) | reproduced interval in seconds, binned at `bin_size` |
| `stimuli` | float64 (6,) | the six intervals, 0.6 to 0.975 s |
| `bin_size` | float64 () | response discretization, 0.02 s |
| `lb`, `ub`, `plb`, `pub` | float64 (5,) | the paper's bounds (Table S2 of the 2020 paper) for `(w_s, w_m, mu_p, sigma_p, lambda)` |
| `paper_mle`, `paper_mle_val` | (5,), () | the paper's maximum-likelihood point and its log-likelihood |
| `paper_ln_z`, `paper_post_mean`, `paper_post_cov`, `paper_post_mode`, `paper_post_mode_val` | (), (5,), (5, 5), (5,), () | the 2020 paper's MCMC ground truth under a uniform prior |
| `paper_marginal_bounds`, `paper_marginal_pdf` | (2, 5), (5, 8192) | the paper's marginal posteriors on grids over the hard bounds |

`make_reference_optima.py` compares its reference minimum with
`paper_mle`; the posterior arrays are not used here.

## `multisensory.npz`

Visuo-vestibular unity judgments (Acerbi, Dokka, Angelaki & Ma 2018),
subjects 1 and 2 of the 2020 paper (benchflow subject indices 0 and 1;
1069 and 857 trials), the "Multisensory causal inference" problems. The
benchmark uses subject 1.
Source: `benchflow/tasks/task_info/multisensory_6D/acerbidokka2018_data.mat`,
columns 3 to 5 of benchflow's per-level arrays (the first two columns are
trial ids and a constant).

| key | dtype, shape | content |
| --- | --- | --- |
| `s{s}_c{c}_stim` | float64 (n, 2) | vestibular and visual directions in degrees, subject `s` in 1, 2, coherence level `c` in 1, 2, 3 |
| `s{s}_c{c}_resp` | int64 (n,) | response code as benchflow's likelihood reads it: 2 when the subject judged the cues to come from different sources, 1 otherwise |

## `reference_optima.json`

The reference minimum of each real-data target, the `f_min` and `x_min`
against which `benchmark_targets.py` measures the error of a run, written
by `dev/scripts/make_reference_optima.py` (its docstring describes the
procedure). Regenerate it when a likelihood or its data changes; `--check`
of `benchmark_targets.py` fails when `f_true(x_min)` no longer equals
`f_min`.

`targets` holds one entry per target:

| key | content |
| --- | --- |
| `D`, `x_min`, `f_min` | the dimension, the reference point and the negative log-likelihood there (the minimum of `multisensory_s1` is a curve, and `x_min` one point of it) |
| `source` | the candidate chosen as the reference: a BADS restart, the polish of one, or (for `timing`) the paper's point |
| `restart_spread` | the restart values above `f_min`: minimum, quartiles, maximum, and the counts within the solved tolerance and within 1e-3 |
| `restarts` | each BADS restart: seed, start point, result, value, evaluations, iterations, message, wall time |
| `polishes` | each polish of a restart result: the value before, after L-BFGS-B and after Nelder-Mead, the evaluations, the point |
| `paper_mle` | `timing` only: the paper's point, its value in the paper and here, and the best value found minus the value there |
| `settings`, `provenance`, `elapsed_s` | the generator's settings; git state, versions, the source and commit of the imported PyBADS and gpyreg, the SHA-256 of the data archives, start and end times; the run time |
