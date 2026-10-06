# The PyBADS film: notes

The decisions behind the film and the facts it rests on: the story, the
rules for the pictures, how PyBADS behaves on the film's problem, the
landscape, the run, the illustrations and the record, and what was tried
and set aside. [README.md](README.md) says how the film is made, and
[STORYBOARD.md](STORYBOARD.md) what it shows, line by line, with the
numbers behind each claim.

## The story

Bayesian optimization is smart and fast, but brittle: its surrogate can be
wrong. Direct search (mesh adaptive direct search) is slow but steady: it
uses no surrogate and cannot be fooled. BADS combines them. Its SEARCH stage
is Bayesian optimization near the best point. When the SEARCH fails several
times in a row, its POLL stage takes a round of direct search, which moves
on without the surrogate and collects the points that help build a better
surrogate for the next SEARCH. The story follows Acerbi & Ma (2017),
section 3, and Singh & Acerbi (2024). The script has 478 words.

## Decisions

- Line 1.5 says why the problem needs a method like BADS: for many
  computational models the error is approximated numerically, or noisy,
  so its slope is no guide, and each evaluation can take seconds; such
  models are black boxes (2026-10-05). The line gives the reason a slope
  fails, not that it is missing: a slope can be computed, by hand or by
  automatic differentiation, or estimated from nearby points, and a model
  whose error has a reliable slope suits a gradient method, as the FAQ
  says. "Many" makes it a property of some models; "computational" and
  "the error" take the modeller's terms, likelihoods included (line 1.2
  makes "the error" stand for both). The claim stays at the level of the
  FAQ's list of the problems that PyBADS suits, "rough (nonsmooth),
  typically due to numerical approximations or noise" and "at least
  moderately expensive". It does not speak of the film's landscape, which
  is smooth: on it, SciPy's BFGS (`scipy.optimize.minimize`, default
  options, SciPy 1.18.1) finds the minimum from 145 of 200 random starts in
  the plausible box.
  PyBADS's own time, 15 to 35 ms per evaluation
  (`dev/results/2026-09-28-where-pybads-spends-its-time.md`), which gpyreg
  1.4.0 cuts further (`dev/results/2026-09-29-bit-identical-speedups.md`),
  is small beside an evaluation that takes seconds.
- The landscape is a valley, lower is better, as PyBADS minimizes
  (2026-10-01).
- Scenes 2 and 3, which show Bayesian optimization and direct search on
  their own, are illustrations computed on the landscape of the real run
  (2026-10-03).
- The film does not present optimization as the only way to fit a model:
  inferring the whole posterior, as PyVBMC does, is another. So line 1.1
  asks which parameters fit best rather than defining model fitting as
  finding them; line 1.2 says what "best" means, which is also what the
  landscape is, the error; and line 1.3, "This is an optimization problem",
  on its own, keeps viewers who do not fit models (2026-10-03). "Make your
  data most probable" is maximum likelihood said plainly; "the likelihood of
  your data" would misname it, as the likelihood is that probability read
  as a function of the parameters.
- Line 2.2 says where Bayesian optimization evaluates in the landscape's
  terms, "looks low, or might be", because it explores as well as exploits
  (2026-10-03).
- Scene 4 credits MADS for the SEARCH and POLL stages with a footnote, and
  shows each stage at work in a panel (2026-10-03).
- The film is about PyBADS 1.5, and the end card says so (2026-10-03).
- The film's voice is Kokoro's `am_michael`, the voice of the animatic
  (2026-10-06), credited on the end card as "Voice: Kokoro". Kokoro's
  weights are under Apache 2.0, which sets no condition on the audio they
  generate, so the credit is a courtesy.
- The film ends on how BADS performs, then zooms out to the studies that
  have used it, and closes on "Join them." over their field (scene 7,
  2026-10-03). The scope of up to twenty parameters on rugged or noisy
  landscapes is that of BADS, of which PyBADS is a port.
- Line 7.3 names a field only for studies whose own text shows that they
  used BADS or PyBADS, and its footnote cites them; a study that only
  cites BADS does not count (2026-10-04).

Settled when the script was locked (2026-10-03):

1. Scene 5 has no line "The SEARCH still misses, so PyBADS polls again,
   with longer steps.": the run shown polls once before the SEARCH works
   again.
2. Lines 4.3 and 5.7 simplify the run: taken literally they do not hold
   for all of it: after evaluation 29, eight searches (35, 36, 40, 42,
   47, 55, 63 and 72) still lower the best value a little, but PyBADS
   counts each as an *incremental* improvement, smaller than the sufficient
   improvement that makes a search a success, so each of those rounds still
   ends in a POLL. The difference is too fine for the narration.
3. The run's first POLL (evaluations 7 and 8, before any SEARCH) plays
   without its indicator, among the "few evaluations" of line 5.2.
4. Line 5.4 says nothing of the order in which the POLL tries its steps,
   although the surrogate expected +52.5 at the step that landed at −24.75
   and ranked it third of four.

Still open:

5. Some labels describe what happens ("far lower", "lower ground: move
   here", "each failed poll: shorter steps", and the captions of scene 4's
   arrows) rather than name what the narration points at; whether they
   stay is open.
6. Conventions proposed on 2026-10-01 and not built: presenting BADS as
   Bayesian optimization with a safety net, and a footnote crediting
   Bayesian optimization (Jones, Schonlau & Welch 1998) beside the MADS
   one.

## Rules for the pictures

At every moment it is obvious what the narration is talking about. That
thing is lit, what gives context is dimmed, and everything else is off.
Things appear when they are introduced and step back when they are done.
The frame stays uncluttered without being empty (2026-10-03).

- **Elements, each with one meaning.** Evaluations are amber dots. The
  surrogate is one green wireframe sheet, bright where it is sure and faint
  where it is not, shown while it is in use. The best point so far is a
  white ring. The POLL is a purple cross of four steps on the mesh, drawn on
  the plane through the best point. The true landscape is a grey wireframe,
  and so are the small landscapes of lines 1.5 and 7.1; in line 1.5 a
  landscape's local slope is a white arrow, and its bottom a white point.
  A search's try shows where the surrogate put it (a green dot on the
  sheet) and where it landed (amber), joined by a dashed white line.
- **Text on screen:** SEARCH and POLL at top left, with four marks for the
  current round (a cross for a failed try, a dot for a success), and short
  labels for what the narration points at; a term that a label introduces
  ("black box", "surrogate", "mesh") is set in italic serif. No readouts, no top view, and no
  numbers beyond what the narration says. A credit is an asterisk on the
  caption and a footnote at the bottom right.
- **The true landscape** is shown in scene 1, faintly in line 2.4 to show
  the trench that the illustration's surrogate misses, hidden during the
  run, and revealed at line 6.1.
- **Scene 4** is a diagram (STORYBOARD.md). The cards light in turn with
  the lines, and a dim card's panel dims with it; at line 4.1 the panels
  carry only the names of the two methods, and the stage names arrive at
  line 4.2. Each panel has its own camera (azimuth −40°, elevation 60° for
  SEARCH and 52° for POLL) and its own height map, which stretches the
  small differences of value near the start. The SEARCH panel replays the
  run's first round of searches (trace steps 1 to 4), the POLL panel the
  illustration's polls 5 to 7.
- **The camera** is calm and orthographic, looking down the valley
  (azimuth −40°, elevation 35°), higher during the POLL.
- **The frame** is letterboxed 16:9, with the captions in the lower bar.

## Mood and sound

Cyberpunk, after the intro of Syndicate (1993, Amiga) and the original Deus
Ex, carried by the music, the pacing and the darkness, not by props. Nothing
on screen that is not a quantity or an object of the algorithm: no glitch
effects, fake terminals, scrolling hex, voice filters or "agent" framing.
The narration stays plain. The black-and-gold palette of Deus Ex: Human
Revolution was set aside as not fitting. The score is tracker-style music
in E minor that follows the film's events (README.md, "The score"); its
level, 17 LU under the voice, was set by ear on 2026-10-03.

## How PyBADS behaves on this problem

Facts from `pybads/bads/bads.py` and the default options
(`pybads/bads/option_configs/advanced_bads_options.ini`) that the script
and the pictures rely on, for two parameters:

- A round of SEARCH is `search_n_try` = max(D, floor(3 + D/2)) = 4 tries. If any try in a round is a
  success, the next round starts and no POLL runs; after a round without a
  success, BADS polls. The first stage after the initial design is a POLL,
  because the search count starts full.
- A try is a success when it lowers the best value by more than a
  sufficient improvement, `tol_improvement`·mesh^1.5 with a floor at
  `tol_fun`. A smaller improvement still moves the best point
  (`sloppy_improvement`) but counts as *incremental*, not as a success.
- The POLL tries its four steps, along each parameter in both directions, in
  the order of the surrogate's lower confidence bound. After a success it
  stops early only when the remaining steps have a negligible probability of
  improving (`tol_poi`).
- The mesh starts at its largest size (`init_mesh_size_integer` and
  `max_poll_grid_number` are both 0). A
  successful POLL can grow it only after it has shrunk; a failed POLL halves
  it, and halves it again when the last iterations stalled
  (`accelerate_mesh`).
- Up to about 50 evaluations every evaluation is in the surrogate's
  training set; from then on the training set holds the points nearest the
  best one, 50 to 60 of them in this run (`n_train_min` = 50, `n_train_max`
  = 50 + 10·D = 70), so the surrogate of the run's last part, and the one
  revealed at line 6.1, is fitted around the trench. The film does not
  mention this locality. The hyperparameters are refitted at intervals,
  checked at every
  SEARCH and POLL step (`_is_gp_refit_time_`); a POLL's points enter the
  surrogate at its next rebuild.

## The landscape

`scripts/landscape.py`: a long diagonal valley from the start down to A,
and a narrow trench that leaves A along +x₁ and deepens to the minimum.

f(x) = a·(d·r)² + b·(n·r)² − S·max(0, r₁)·exp(−r₂²/(2w²)), with r = x − A,
d = (1, 1)/√2, n = (1, −1)/√2, and S = (a + b)·t\*, which puts the bottom of
the trench's centre line at distance t\* from A along +x₁.

The film's variant: a = 0.08, b = 4, A = (−0.5, −0.5), t\* = 4, w = 0.2, so
S = 16.32. Bounds [−5, 5]², plausible box [−4, 4]², start (−4, −4),
f(start) = 1.96, f(A) = 0. The true minimum is −32.715224 at
(3.504618, −0.490405), slightly off the trench's centre line; `illus.js`
holds it (key `truth`), while `landscape.make` and `trace.js` hold the
nominal point (3.5, −0.5), at −32.64. The valley is easy for a smooth
surrogate and slow for steps along the parameters. The trench is too narrow
for a smooth surrogate and lies along a parameter axis.

Three variants were swept on 30 seeds each. "Within 1" counts evaluations
until the best value is within 1 of the nominal minimum; for direct search
the range covers its two orders of steps, +x₁ first and +x₂ first. A "full
arc" is a run with a successful SEARCH round, then a successful POLL after
the first iteration, then a successful SEARCH round.

| Variant | Direct search, within 1 | PyBADS, within 1 (median) | Full arc |
|---|---|---|---|
| a = 0.12, b = 1.5, A = (−1, −1), t\* = 4.75 | 31 to 45 | 36 | 60 % |
| a = 0.10, b = 3, A = (−1, −1), t\* = 4.75 | 57 to 86 | 34.5 | 47 % |
| **a = 0.08, b = 4, A = (−0.5, −0.5), t\* = 4** | **68 to 98** | **39** | **63 %** |

## The run: seed 25

PyBADS with default options and `random_seed=25` on the film's variant. It
ends at the minimum after 87 evaluations and is within 1 of it after 35.
Among the 19 seeds with a full arc, seed 25 tells the story most simply: one
failed round with large mispredictions between successful rounds down the
valley, and a POLL rescue after which the next search lands near the bottom
(seeds 4, 18, 19 and 25 were compared). STORYBOARD.md, scene 5, gives its
numbers line by line; beyond them:

- Evaluation 2 repeats the start as PyBADS's test for noise, and the film
  does not draw it. The first POLL (7 and 8) takes the mesh step from 4
  to 2.
- The first round of searches (trace steps 1 to 4) lands lower, higher,
  lower and higher, the last at the corner (−5, −5). Its third try,
  predicted +1.69 ± 1.62 where the best value was +0.90, was worth trying
  only for the surrogate's uncertainty there, and landed at +0.30.
- The surrogate is refitted during the POLL of evaluations 25 to 28, and
  after evaluation 29 the incremental improvements and the failed POLLs of
  decision 2 follow, until the mesh has shrunk and the run stops.

The trace keeps only the surrogate states that a step used, renumbered in
order, so a state of the run's own count can have another index in the
trace. README.md, "The data", says how it was recorded.

## The illustrations (scenes 2 and 3)

`scripts/illustrations.py` writes `illus.js`; it reads `trace.js` for the
run's initial design.

- **Plain Bayesian optimization:** a Gaussian process with a
  squared-exponential ARD kernel on standardized values, hyperparameters by
  maximum marginal likelihood, and the lower confidence bound μ − 0.5σ
  minimized on a 161 × 161 grid. It starts from the run's initial design
  (five distinct points) and takes 16 steps, down the valley to its floor,
  and never comes near the deep part of the trench. With μ − 2σ it first
  spends four steps on the edges of the box, three of them corners, one at
  f = 200, so the illustration uses 0.5.
- **Plain direct search:** steps along the parameters tried in the order
  +x₁, −x₁, +x₂, −x₂, first step 2, longest 4, doubled after a success and
  halved after a failure, stopping below 4·2⁻⁷. It zigzags down the valley
  in a staircase, finds the trench at the valley's end and follows it: 96
  evaluations in all, within 1 of the minimum after 68.

## The record (scene 7)

**The benchmark's data.** `scripts/bench_export.py` reads the 2017
benchmark's result files, which are not public: for each of the six studies
of the CCN17 set, a cache that holds every optimizer's curve per dataset and
the raw runs behind it. It averages the curves over each study's six
datasets as the 2017 code did (201 log-spaced points; evaluations / D from
10 to 500 for the deterministic studies, the error tolerance from 10 to 0.1
at the end of the budget for the noisy ones), and writes `bench.js`;
`--check` plots the six panels, which match the published Figs 2 and 3 up
to small differences, none of which changes the order of the curves. The
film shows BADS's curve without its overhead correction, as the paper's
headline curve. Each chart shows between 6 and 13 other optimizers, and
plain Bayesian optimization (MATLAB's `bayesopt`), whose curve stays at zero
in four studies: the three deterministic ones and word recognition memory.
Two of the other optimizers have their own colours and names in the
legend, in every study where they ran: CMA-ES and `fmincon`, named
"gradient-based", each drawn by the best of its variants in the study; one
of the two is the best of the others in every study, and the others are
faint (STORYBOARD.md, line 7.2).

The charts have no numbers on their axes; the footnote names them.

## The animatic

The animatic plays the whole film on Kokoro's `am_michael`, which became
the film's voice, and is judged on whether it reads, not on its look.
v1 ran 3 min 10 s and was too slow, its start above all; from v2 the voice is 12 % faster (20 % in scene 1, 15 % in scene 2), with
shorter pauses. In v3 every transition between scenes takes the same 0.7 s,
where v2's took 0.9 to 2.4 s, and the gaps between lines are a quarter
shorter. v4 opens line 3.1 on "Instead", which tells that another method
follows line 2.4; v5 adds line 1.2 and names the landscape "the error" at
line 1.4; v6 adds the score, and v7 lowers it by 2 dB. v8 voices line 7.3
again, with quantum computers as its second field, and lights each of its
labels as its word is spoken. In v9 the labels and overlays fade with their
shots at every line change: until v8, the labels of the shot before
vanished as a line started, and an overlay that both shots showed (scene
4's diagram, the HUD of scene 5, scene 7's tiles and charts) blinked off and
came back; in line 7.2 the tiles' frames and labels were hidden. In v10
the surrogate of line 5.3 caves in once, from 2.8 s into its shot, where
the score's fooled surrogate begins: until v9 it took its new shape at the
end of evaluation 20's try, went back to the old one, and then caved in.
v11 (2 min 56 s) rewrites line 1.5 to say why the problem needs a method
like BADS (the first of the decisions above): the model is a black box,
which gives only heights, and each evaluation can take seconds; and in
line 7.2 it names the best of the other optimizers in each chart, brings
the legend in with BADS's curve rather than near the line's end, and holds
the finished charts 2.5 s after the line, instead of 0.5 s. v11 named the
best of the others in a key beside each chart's title, under a legend that
named BADS and Bayesian optimization alone, and the names went unseen;
in v12 the legend names CMA-ES and gradient-based too, each in its own
colour in every chart, and builds up as the curves draw. v13 (2 min 58 s)
gives line 1.5 its present wording: v11 and v12 said "To the optimizer, your
model is a black box: each evaluation gives only the height at a single
point, not the slope, and can take seconds.", which read as true of every
model and left open why the slope is not computed. Its caption, too long
for the lower bar, shows in two parts. In v14 line 1.5 shows what it says,
where v13 held an empty floor for 7 s: two small landscapes, one under
bumps and one under noise, each with its bottom and an arrow for its local
slope, and the evaluation dropping slowly as each evaluation "can take
seconds". v15 credits the voice on the end card as "Voice: Kokoro", the
animatic's voice having become the film's. The masters are v15 at
1920 × 1080, the 1280 × 720 layout drawn at 1.5 device pixels per pixel,
one with the captions and one without, at CRF 18.

Whisper (`verify_voice.py`) hears most lines of the voice as written,
the others mostly differing in spelling ("20" for "twenty", "pole" for
"poll"); a check of each line cut from the mix found the same words with
the score under them. Worth a listen: line
3.3, whose second sentence Whisper drops; line 4.3, "keeps" heard as
"keep"; and line 5.4, where this voice runs "Pie-Bads polls" together. A
new take of a line moves every line after it.

## What was tried and set aside

- **The style frame (2026-10-02).** Forty seconds of an earlier run (seed
  21 on the earlier landscape below), with the surrogate as dashed contour
  lines, hard cuts on the beat, a HUD with readouts and a top view, and
  short captions. It did not explain the method, the screen was cluttered,
  and its narration was vague: a review of implementation details (for instance, that
  the POLL orders its steps by the surrogate) had turned its captions into
  hedged fragments. The narration tells the paper's story plainly; a detail
  enters only when the story needs it. Kept from the style frame: the
  palette, the dark ground and glow, the letterbox and captions, the
  SEARCH/POLL indicator, the mesh, the POLL's cross, the falling points, the
  bracket on a search's pick, and the score, whose instruments and mix are
  `scripts/synth.py`.
- **The earlier landscape:** a bowl with an axis-aligned trench,
  0.4·|x − A|² − 5.2·max(0, x₁ + 3)·exp(−x₂²/(2·0.2²)) with A = (−3, 0).
  Plain direct search gets within 1 of the minimum in 5 to 10 evaluations,
  against 33 for PyBADS (seed 21), because the trench lies along the
  direction it steps in: scene 3 would have called direct search slow over
  a landscape where it is fast.
- **Other landscapes.** On smooth or merely rugged ones (bowl, Rosenbrock,
  egg-crate, terraces, plateau, cliff), in two dimensions the SEARCH does
  nearly all the work, and a POLL succeeds after the first iteration in 0
  to 5 % of seeds: no rescue to show. "Crumpled" landscapes have POLL
  rescues in 80 to 85 % of seeds, but near the minimum, in runs of 136 to
  156 evaluations. With a tilted or curved trench the POLL succeeds in 5 to
  10 % of seeds: the trench has to lie along a parameter axis for the POLL
  to land in it, so the narration never claims that the POLL finds any
  valley.
