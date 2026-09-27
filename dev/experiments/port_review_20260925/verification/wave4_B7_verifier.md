<!-- Report of the verifier of wave 4, slice B7 (the two B7 reports and the items kept from the reviewers, given as B7-K1 onwards, briefs/wave4_kept_B7.md), reading PyBADS at 0d866e8 in ../pybads-review, MATLAB BADS at 74919c0 and gpyreg v1.3.3 (98ab5a4), with the complete history, in a cloud session; saved verbatim from its final message on 2026-09-27. Its scratch paths are the sandbox's; its check scripts and outputs are copied to scripts/wave4/B7_verifier/. -->

# Wave 4 verification: B7

I read the code at `0d866e8` (`/home/user/pybads-review`) and MATLAB at `74919c0`. Each finding was reproduced with my own scripts in `/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad/wave4/B7_verifier/` (named `v*.py`, each writing a `v*.out` log). Every script printed `pybads /home/user/pybads-review/pybads/__init__.py` and `gpyreg /home/user/gpyreg-v1.3.3/gpyreg/__init__.py`. The environment was NumPy 2.4.6 and SciPy 1.17.1 on x86_64 Linux, with one BLAS thread. For dating only, `v110/` holds a `git archive` copy of v1.1.0's package.

The two reports describe the same behaviours several times over, so I verified each behaviour once, under a letter (A to N):
- "I-Fn" is a finding of the internal-track report; "C-Fn" is one of the comparison report.
- Kept items K1 to K9 are mapped onto the same letters.

## 1. Summary

| | Finding (reports, kept items) | Classification | Reached at default options | Dating | Confidence |
|---|---|---|---|---|---|
| A | Seed comes from the integer parts of `u0`, so there is one design per D for every start inside the plausible box (I-F1, C-F1 interior part; K1; K2's seed claims) | **needs MATLAB** for the effect (the PyBADS facts are confirmed); a design decision follows | yes, every run, levels 0, 1, 2 | never agreed: Python `c7c88ab` (2022-06-02); MATLAB `initSobol.m:9-15` unchanged since `6c93629` (2017-03-14) | high (Python); medium (MATLAB arithmetic, from documentation excerpts) |
| B | Undefined float to `uint64` cast for a coordinate u ≤ −1 makes the design depend on the platform (I-F2, C-F1 platform part; K2's cast claims; K3) | **confirmed defect** | yes, at levels 0, 1, 2, for a start with u ≤ −1 among its first 11 coordinates (x0 ≤ plb) | `c7c88ab`; reached at default since then, including at 1.1.0. It was **not** introduced by W2-4. MATLAB has no counterpart | high (x86); medium (arm64, by reading) |
| C | Design doubled when 2^m = D (I-F3, C-F2, K4) | **design question** | yes: level 0 at D ∈ {1, 2, 4, 8, 16}; levels 1 and 2 only at D = 32 | never agreed: `c7c88ab`, unchanged | high |
| D | `init_sobol` returns the exponent; docstring errors (I-F4, C-F7 part, K6) | **confirmed, inert** | the call yes; the value is discarded | `4e6a001` (2022-11-15) | high |
| E | Level-2 merge unreachable; KD-B7-3's reasoning is out of date (I-F5, C-F3, K9) | **confirmed, inert** (the records need correcting) | no, at no option | merge since `c7c88ab` (`199d787`; row fix `032dfcb` in `ab4dded`); unreachable since `149d528` (W3-1, in `0d866e8`); MATLAB never merges | high |
| F | Noise test leaves `n_evals[0] = 2` and an averaged time in row 0 (I-F6, C-F6b, K7) | `n_eff`: **confirmed defect** (minor, internal); time row: **confirmed, inert** | yes, levels 0 and 1 (`uncertainty_handling=None`) | matched MATLAB at `c7c88ab` (direct call); diverged in `f9e9326`/`8ff10f5` (Nov 2022) | high |
| G | `overhead`: final samples and merges (K8); the noise test counted as optimizer time (I-F7) | K8: **no longer holds** (`17e65ee`, in `8aecb6a`); I-F7: **confirmed shared defect**, negligible, kept as MATLAB | I-F7: yes, levels 0 and 1 | MATLAB's accounting since `603da99` (2017); the Python fix is dated 2026-09-26 | high |
| H | Malformed target outputs do not raise the documented `ValueError` (I-F8, C-F7 SD part) | **confirmed port discrepancy** (minor) | no | Python `c7c88ab`/`9037851`; MATLAB's `isscalar` since `a3b6ebd` (2021); never agreed | high |
| I | `finalize` does not trim `n_evals`; `reset_fun_eval_time` gives the wrong length (I-F9) | **confirmed, inert** | no | `c7c88ab` | high |
| J | `add`/`__call__` docstrings omit the space; `add` differs from `__call__` (I-F10, C-F7 `add` part) | **confirmed, inert** | no | `c7c88ab` | high |
| K | `period_check` call sites are inconsistent; the poll discards the result (I-F11, C-F5) | **confirmed, inert** (latent) | no | `c7c88ab`; MATLAB assigns at every site since 2017 | high |
| L | The log grows; MATLAB's is a ring of `CacheSize` (C-F4) | **intentional difference** (missing from the sheet) | no at D ≤ 20 | never agreed | high |
| M | The noise test's value goes through the logger's checks (C-F6a) | **confirmed port discrepancy** (benign) | no (needs a non-finite second value) | matched MATLAB at `c7c88ab`; diverged in `9037851` (2022-09-22) | high |
| N | A noisy small-budget run never takes the final samples it reserved (K5) | **confirmed shared defect**, widened by the port's rounding of the design | no (small `max_fun_evals`), levels 1 and 2 | both sides since the port (MATLAB `iter > 1` since `3f9522b`, 2017); the widening since `c7c88ab` | high |

How the kept items map:
- **K1:** under A, together with B's cast clause.
- **K2:** under A and B. Its NumPy 1.x/Windows 32-bit clause **no longer holds** since `1c8c71d` (#65, "require NumPy 2"; `pyproject.toml:12`). I found no `c044fea` in either repository.
- **K3:** under B. Its facts are confirmed but its dating is wrong.
- **K4 = C**, **K5 = N**, **K6 = D**, **K7 = F**, **K8 = G**, **K9 = E**.

## 2. Per finding

### A. The seed ignores the start and the generator (I-F1, C-F1, K1, K2)

**PyBADS side.** `init_sobol.py:52-62` casts `u0[:11]` to `uint64`, applies `array2string`, multiplies the character codes in int64, then takes the result mod 997, plus 1.

`v1_seed.py`, 20 random interior grid starts per D:
```
D= 1 seeds [49]   D= 2 [948]   D= 3 [967]   D= 4 [241]   D= 5 [748]   D= 6 [843]
D= 7 [431]; int64 product 3170534137668829184 (exact 630359832643793584128, wraps: True)
D= 8 .. 20 [1]; int64 product 0
interior starts give the same design: True
second return value: 2 for 4 rows
```
- The reports' seeds are right. The one they do not give is D = 7: the product already wraps there, to a nonzero value that gives seed 431.
- The branch that draws from the generator (line 64) is dead code in a BADS run. `bads.py:323-339` replaces any non-finite `x0` before this point, so `u0` is always finite.

Consequence, measured:
- `v7_misc.py` (1): random `x0`, D = 3, 20 seeds. Output: `distinct x0 20; designs identical in all 20: True; start after the design: 16 of 20 at the same design point, 4 stay at x0`.
- `v6_seed_effect.py`: at a fixed `x0`, seeds 0 and 1 evaluate identical points up to row 11 (x0, 4 design points and the first poll's 6), then diverge.
- For a fixed `x0` MATLAB's design is fixed too, since MATLAB has no seed. The difference from MATLAB is therefore only the dependence on the start.

**K2's numerical claims hold here:**
- The int32 product, emulated, overflows at D = 4 (seed 515) and is 0 from D = 5.
- The int64 product is 0 from D = 8, as stated. It already wraps at D = 7, which K2 does not say.
- The int32 case only described NumPy 1.x on Windows, which PyBADS no longer accepts.

**MATLAB side (K1).** What the documentation settles:
- `uint64(strseed)` gives the character codes.
- MATLAB's documented default output type of `prod` is double for every input except `single`, and the operation is performed in that type. The MathWorks pages are blocked by this sandbox's proxy, so I read this from search excerpts of the `prod` page ("returns output as single when the input is single, but for all other numeric and logical data types, it returns double"). The MATLAB 7 note on `sum` agrees: 'double' "is the default for integer data types".
- So `prod` neither keeps `uint64` nor saturates. The saturating reading (seed 961 for every multi-D start) is contrary to the documentation.
- `i4_sobol_generate.m` takes points `seed … seed+Ninit−1` of the unscrambled sequence, as W0-19 says.

My transcription (`v1_seed.py`, `v1b_matlab_mod.py`) uses `num2str` with 5 significant digits, blank-trimmed, which holds for |u| ≤ 1 in every version I know. It gives:
- When the product p is below 2^53, the seed is exact:

  | Start | D = 1 | D = 2 | D = 3 | D = 4 |
  |---|---|---|---|---|
  | centre, `u0 = 0` (`'0  0 …'`) | 49 | 395 | 161 | 982 |

  For a 1-D non-integer start, `'0.25'` gives 805.
- For **every non-integer start with D ≥ 2**, p/997 exceeds 2^53. For example `'0.25         -0.5'` gives p = 1.08e27. The seed then depends on how MATLAB's `mod` treats a double beyond flintmax:
  - If `mod` returns the exact remainder (as C's `fmod`), [0.25, −0.5] gives 966. Over 500 random grid starts there are 378 to 395 distinct seeds (D = 2, 3, 5, 10).
  - If `mod` evaluates its documented formula `x − floor(x/997)*997`, or applies MathWorks' round-off compensation (a quotient within eps of an integer counts as that integer, giving 0), the seed is 1 for 500 of 500 starts at every D ≥ 2. MATLAB would then also have one design per D.
- So the survey's conclusion that "the port differs in mechanism more than in effect" may still hold, through `mod` rather than through a saturating `prod`.
- The comparison report's statement "if it accumulates in double, the seed follows the digits" holds only if `mod` is exact.

**What only MATLAB can settle:**
- `mod(prod(uint64(num2str([0.25 -0.5]))),997)+1`: 966 means exact, 1 means not.
- The rounding order of `prod` once the odd part of p exceeds 2^53.
- `class(prod(uint64('ab')))`, as a confirmation of the documentation.

**Sheet.** This contradicts KD-B1-1's headline, "Every random draw comes from one … Generator". The scrambling draws from SciPy's `default_rng(seed)`, and KD-B1-1 cites `init_sobol.py:64` as a draw site although no run reaches that line. The same contradiction is in the BADS docstring (`bads.py:142`), `index.rst:23` and CHANGELOG 1.1.0 (line 625). KD-B7-1 leaves the derivation open; nothing contradicts it.

**Recommended disposition: needs MATLAB (the one call above), then decide the design.**
- If MATLAB gives 1: the port matches MATLAB in effect. Keep it, settle KD-B7-1, fix only B, and correct the statements that every random draw comes from the generator.
- If MATLAB gives 966: either transcribe MATLAB's digits-based seed (format plus an exact integer product mod 997), or seed the scrambling from `bads.rng` (a deliberate departure).
- Either change moves every default run from its first design point. The gate is a population comparison at default options on both platform references, plus `tolerance_sweep.py` for the optimization tests.

### B. The undefined cast (I-F2, C-F1 platform part, K2, K3)

**Lines.** `init_sobol.py:55` does `astype(np.uint64)`. C11 6.3.1.4 makes the conversion undefined when the integral part is not representable, so −0.99 gives a defined 0 and −1.0 is undefined.

**On this x86 machine** (`v1_seed.py` §3), no warnings are raised:
```
 -0.99 -> 0   -1.00 -> 18446744073709551615   -1.50 -> 18446744073709551615   -2.00 -> 18446744073709551614
u0=[-1.0, 0.5]: string '18446744073709551615                    0', int64 product 0, seed 1
```

**What NumPy documents.**
- The `astype` docstring is silent on the value.
- The 1.24 release notes say an out-of-range float to int cast "may return an undefined result with a warning: 'RuntimeWarning: invalid value encountered in cast'", that "the precise behavior is subject to the C99 standard and its implementation in both software and hardware", and that the warnings are platform dependent.
- AArch64's `FCVTZU` saturates negatives to 0 and raises the invalid-operation flag. That gives the interior string and seed: 948 at D = 2, 967 at D = 3. This matches the survey's macOS arm64 warning. I could not run arm64 here.

**Reached at default** (`v2_bound_start.py`, `v9_extra.py`, `v10_small_noisy_seed.out`):
- `x0 = lb` with the plausible bounds omitted, with either transform: `u0 on the grid [-1.0, 0.1669921875], u0[0] == -1 exactly: True`. The start is not moved: in u, lb is −1 as well. A default run passes it to `init_sobol`: `cast [18446744073709551615, 0] … seed 1`, against 948 for an interior start.
- `x0 = plb` given explicitly inside the hard bounds: `u0 [-1.0, 0.25]`.
- `test_small_noisy_func` starts at `x0 = −3` with plb = −2, which is u0 = −1.5. On x86 its seed is 1; on arm64 it would be 967.

**K3's dating does not hold.** At v1.1.0, `x0 = lb` with plb omitted also gave `u0 = -1` exactly. The effective-bound block moved `x0` and `plb` together (`v2_bound_start_v110.out`: `x0 [-2.994, 0.5], plb (orig) [-2.994, -2.994] … u0[0] == -1 exactly: True`). So W2-4 (`a31a9be`) did not make the cast reachable. Any `x0 ≤ plb` has reached it since `c7c88ab`.

**Recommended disposition: fix**, either inside A's redesign or on its own.
- `u0[:11].astype(np.int64).astype(np.uint64)` is well defined: float to int64 truncates for |u| < 2^63, and int64 to uint64 wraps. It reproduces x86's current values on every platform.
- With that form, x86 runs are unchanged, which the fingerprint gate shows; only arm64 runs from such starts change.
- A fix that follows arm64 instead changes x86 runs from starts with u ≤ −1 only, including `test_small_noisy_func`. That needs its tolerance sweep, and a population comparison only if the benchmark's starts reach u ≤ −1.

### C. The doubling (I-F3, C-F2, K4)

**Lines and history.** `init_sobol.py:73-76`. `git log -L` shows it arrived with the whole file in `c7c88ab` ("Init porting (#1)"), with no message or comment explaining it. The commented-out lines 71-72 (`# n_samples = fun_eval_start` / `# samples = sobol_sampler.random(n_samples)`) show an earlier form. Owen's comment (66-68) argues only for a power of two. MATLAB draws `Ninit` (`evalinitmesh.m:101-104`).

**Design sizes** (`v3_design_size.py`, running `_init_mesh_` only at default options):

| D | 1 | 2 | 3 | 4 | 5-7 | 8 | 9-15 | 16 | 17-20 |
|---|---|---|---|---|---|---|---|---|---|
| Level 0 design (at 0d866e8) | 2 | 4 | 4 | 8 | 8 | 16 | 16 | 32 | 32 |
| Rounding alone | 1 | 2 | 4 | 4 | 8 | 8 | 16 | 16 | 32 |
| MATLAB | D | D | D | D | D | D | D | D | D |

- At levels 1 and 2 the design is 32 at every D ≤ 20 (MATLAB: 20). There the doubling acts only at D = 32.
- User-set sizes jump: D = 4 gives `2:2 3:8`; D = 16 gives `8:8 9:32`.
- Geometric necessity: the search needs more than D rows (`bads.py:1400-1404`). x0 plus the rounded design already has at least D + 1, so the doubling is not needed for that.

**The decision.**
- *Keep:* the costs are D extra evaluations at D ∈ {1, 2, 4, 8, 16} (0.2-0.4% of 500·D; 4 of 61 evaluations in my D = 2 run) and the jumps in user sizes. It then needs a stated reason, and none exists.
- *Remove:* the design becomes `2**ceil(log2(fes))`, still a power of two and still at least `fes`. Level-0 default runs at those D change from the design on. Every other D, and every noisy run at D ≤ 20, stays bit-identical.
- A removal also updates the `fun_eval_start` description, `AGENTS.md`, KD-B7-1 (whose formula already omits the `+1`) and KD-B2-6, and gets a changelog entry.

**Recommended disposition: decide the design. I recommend removal.** Its gate is a population comparison whose problems include level-0 runs at a power-of-two D, plus an unchanged fingerprint if the fingerprint's runs avoid those D.

### D. `init_sobol`'s return value and docstring (I-F4, C-F7 part, K6)

`v1`: `second return value: 2 for 4 rows`. The docstring (44-49) calls it "Number of samples"; the caller discards it (`bads.py:1112`, `u1, _ = …`).

Also confirmed in the docstring:
- The parameter defaults are types (8-13).
- `lb` and `ub` are unused.
- `plb`/`pub` are described as "Lower/Upper bounds for the parameters".
- `fun_eval_start` is described as "Number of initial function evaluations", which ignores the rounding and the doubling.

Dated `4e6a001`. **Disposition: fix the docstring**, or return the number of points. The gate is the fingerprint.

### E. The level-2 merge is unreachable (I-F5, C-F3, K9)

**KD-B7-3 describes the code at `0d866e8`:**
- `function_logger.py:406-436` merges into the row that matches in every coordinate, by precision weighting.
- It returns the merged value with the new observation's SD. `Y_orig` and `Y_max` are not updated.
- After a merge, `add_and_update_gp` (`gaussian_process_train.py:1315-1372`) would add `(x, merged value, new SD²)` as a new GP row beside the old one. The new observation would then weigh (b/(a+b))² instead of b/(a+b), as the survey says.

**No BADS run reaches it at any option:**
- Every recorded evaluation after x0 passes `contraints_check` (`constraints_check.py:34-49`). It removes a candidate whose `tol_mesh/2` bin holds a logged point, and an exact repeat always shares that bin.
- At level 2 there is no noise test: `specify_target_noise` sets `uncertainty_handling=True` (`bads.py:855-860`).
- So at level 2 the only evaluations that repeat a point are the final samples, 10 at default. They take `record_duplicate_data=False` (390-404): `n_evals` gets +1 and the time is averaged, with no merge and no GP row.
- `add` has no caller.

`v5_logger.py` (c), D = 3, 200 evaluations each:
```
level 2 seed 0: func_count 200, rows 190, distinct rows 190, calls by path {'new row': 190, 'unrecorded (repeat)': 10}
level 1 seed 0: func_count 165, rows 154, distinct rows 154, calls by path {'new row': 154, 'unrecorded (repeat)': 11}
level 0 seed 0: func_count 77, rows 76, distinct rows 76, calls by path {'new row': 76, 'unrecorded (repeat)': 1}
```
The other seeds are alike.

**What the records get wrong.**
- KD-B7-3's "Until the next rebuild, `add_and_update_gp` then adds…" and its population evidence (20 of 90 runs) describe runs before W3-1 (`149d528`, 2026-09-27). At `0d866e8` neither form changes any run.
- The same holds for CHANGELOG line 158, "Repeated points with user-specified noise". It conflicts with line 586-590, "neither repeats any now".
- The same holds for `AGENTS.md`'s FunctionLogger bullet.

**Disposition: correct the records.**
- KD-B7-3 should say that since W3-1 only direct use of `FunctionLogger` reaches the merge. The survey row can then be closed by it.
- Fold the changelog entry into "Points evaluated again".
- Optionally, remove the merge path. That moves no run; the gate is the fingerprint.

### F. The noise test's trace in row 0 (I-F6, C-F6b, K7)

`v5` (b), D = 2, with a timer giving 1 s, 2 s, and so on:
```
n_evals[0] 2, fun_eval_time[0] 1.5 (calls timed 1 s and 2 s) ...
after 0 more evaluations: n_eff 6, eff_starting_points 5, n_budget 65: init_N 123 (counting rows only: 128)
after 10 more evaluations: n_eff 16, ...: init_N 77 (counting rows only: 81)
```
- `n_eff` (`gaussian_process_train.py:1133`) counts the noise test.
- `eff_starting_points` (`bads.py:1175`) does not, although the comment at 1145-1147 calls x "the fraction of the budget used after the initial design".
- `init_N` is PyBADS's own schedule.
- The per-row time has no reader: `t_train` (line 92) is unused.
- MATLAB calls the target directly for the test (`evalinitmesh.m:41`), and `funevaltime` is written only for `'iter'` calls (`funlogger.m:128`).

**Dating.** At `c7c88ab` the test called `function_logger.fun` directly, as MATLAB does. `9037851` recorded it as a row. `f9e9326`/`8ff10f5` made it unrecorded but kept the `n_evals` increment.

**Disposition:**
- `n_eff`: fix, with low priority. A fix shifts `init_N` in every default level-0/1 run, so its gate is a population comparison at default options, which the benchmark reaches.
- Time row: optional; the gate is the fingerprint.

### G. `overhead` (K8, I-F7)

`v5` (a), with a fixed 1 s per timed call:
```
level 0 (noise test): func_count 51, rows 50, total target time 50 s
level 1 (noise test, final samples): func_count 100, rows 89, total target time 99 s
level 2 (final samples): func_count 100, rows 90, total target time 100 s
```
- The code fixed in `17e65ee` (`function_logger.py:383-387`, `bads.py:1059-1065`) matches MATLAB. `funlogger.m:130` times `'iter'` and `'single'` calls, so the final samples count; the noise test is an untimed direct call.
- **K8 no longer holds.**
- I-F7 is true on both sides: MATLAB's `totaltime = toc(t0)` (`bads.m:1186`) also spans the untimed test (`bads_output.m:48`).
- **Disposition: keep, as the W2-20 ruling did**, and optionally add one sentence to the `overhead` description. No run changes.

### H. Malformed outputs (I-F8, C-F7 SD part)

`v7` (2):
```
level 2, output (1.0, [0.5]): TypeError: '<=' not supported between instances of 'list' and 'float'
level 2, output (1.0, None): TypeError ...;  (1.0, [0.5, 0.6]): ValueError: The truth value of an array ...
level 2, output (1.0, array([0.5, 0.6])): ValueError: can only convert an array of size 1 ... (+FuncError note)
level 0, output (1+0j): TypeError: float() argument must be ...; Xn 0, X_max_idx 0, func_count 0
level 0, output np.complex128(1+0j): accepted -> np.complex128(1+0j)
```
- The SD check (173-179) lacks MATLAB's `isscalar` (`funlogger.m:108`), which `add` has (268-277).
- A Python complex with zero imaginary part leaves a half-written row.

**Disposition: fix.** Convert and check the SD as the value is converted and checked, validate before writing the row, and drop the FuncError note on the logger's own conversion. The gate is the fingerprint.

### I. `finalize` and `reset_fun_eval_time` (I-F9)

`v8_finalize.py`:
```
after finalize: X (4, 2), X_flag (4,), n_evals (5, 1) ...
n_evals[X_flag] raises IndexError ...
one more call: X (6, 2), n_evals (7, 1)
reset_fun_eval_time after growth: fun_eval_time (3, 1) vs X (8, 2)
```
No code calls either method. **Disposition: fix or remove the methods.** The gate is the fingerprint.

### J. `add` and `__call__` docstrings (I-F10, C-F7 `add` part)

`v7` (4):
- `add(np.array([1.]))` raises `ValueError`, where `__call__` accepts the same value.
- With the SDs held, a missing SD becomes 1.
- Without them, a given SD is dropped.

Neither docstring (79-81, 208-210) says that `x` is in u space. **Disposition: correct the docstrings now**; settle the semantics when `fun_values` is ported.

### K. `period_check` call sites (I-F11, C-F5)

- The design passes `options["periodic_vars"]` (None) at `bads.py:1130-1135`.
- The search passes the mask at 1837-1842.
- The poll discards the result at 2220-2225.
- MATLAB assigns the result at every call site (`bads.m:552, 611, 807`; `evalinitmesh.m:107`; `searchES.m:128`; `gpupdate.m:122`).

It is inert: the stub returns its input and periodic variables are refused. **Disposition: fix with the port of periodic variables**, or assign the poll's result now. The gate is the fingerprint.

### L. Log growth against MATLAB's ring (C-F4)

- `v9` (3) transcribes `funlogger.m:120-121` for nmax = 5: `rows written [1, 2, 3, 4, 1, 2, 3, 4, 1], Xmax 5`. Row nmax is never written.
- PyBADS grows by 50% (303-340). Its docstring calls `cache_size` "The initial size", which makes the growth deliberate; the `.ini` line 20, "Size of cache for storing fcn evaluations", misdescribes it.
- It is unreached at D ≤ 20: at most 9999 `'iter'` evaluations are logged there.
- **Disposition: keep and document** (a sheet entry, and the `.ini` wording). No run changes.

### M. The noise test's validation (C-F6a)

- `v7` (5): `second value nan: ValueError at call 2: FunctionLogger:InvalidFuncValue`, and the same for inf.
- MATLAB (`evalinitmesh.m:41-47`) reads NaN as deterministic (the comparison is false) and inf as noisy, and carries on.
- PyBADS's behaviour is the safer one, but no record makes the difference deliberate.
- **Disposition: keep and document it on the sheet.**

### N. Reserved final samples left unused (K5)

`v4_noisy_budget.py`, D = 2, level 1:
```
max_fun_evals 38: target calls 34, iterations 1, noise_final_samples reserved 4, max_fun_evals after the reserve 34, yval_vec [0.14349133], fsd 1
budget 44: calls 34, reserved 10, taken False | 46: calls 36, False | 48: calls 38, False | 50: calls 50, True
```
- The final samples need `poll_iteration > 0` (`bads.py:1631`). MATLAB's condition is the same, `iter > 1` (`bads.m:1138`, since 2017).
- In MATLAB (design 20) the design alone ends the run in its first iteration only for budgets up to 32. In PyBADS (design 32) it happens for budgets up to 44, and the first poll extends that to 48.
- `dev/TODO.md`'s wording ("design leaves fewer evaluations than `noise_final_samples`") is narrower than this: it misses budgets 45-48.
- **Disposition: decide the design** (take the samples at the incumbent, or reserve none when the loop gets no evaluations), and correct the TODO wording. It changes only noisy small-budget runs. The gate is a configuration with such a budget, plus an unchanged fingerprint for default runs.

## 3. Met while verifying (unverified unless stated)

1. **Unverified.** After MATLAB's ring wraps, `U(1:Xmax)` and `Y(1:Xmax)` include the never-written NaN row (`uCheck.m:23`, `gpupdate.m:30`). It is probably harmless unless `ntrain` reaches `Xmax`.
2. **Unverified.** Under the literal `mod` formula, one transcribed MATLAB seed came out as −1.09e40. `i4_sobol.m:249-250` clamps a negative seed to 0, which would start the design at the Sobol origin (the PLB corner). Only MATLAB can say whether this happens.
3. **Seen in `v4`; MATLAB side unverified.** A level-1 run that ends in its first iteration reports `fsd = noise_size` (1), a default rather than an estimate. MATLAB appears to do the same (`bads.m:448-452`).
4. **Reproduced (`v9` (2)).** These come from the internal report's answers, not from its findings: `periodic_vars=[]` is refused, although MATLAB treats it as no periodic variables; and with a random `x0`, an index out of range raises `IndexError` in `_variable_transformer_` before the refusal.
