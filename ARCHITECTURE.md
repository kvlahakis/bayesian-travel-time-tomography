# Architecture

## Dependency chain

```
             geometry
            /        \
           v          v
       forward       prior
            \        /
             v      v
            inversion
                |
                v
           diagnostics
                |
                v
            experiments
                |
                v
              plots
                |
                v
     scripts / notebook
```

- `geometry.py` has no dependencies on other project modules. It defines the
  grid, cell-ordering convention, acquisition geometry, and the sensitivity
  matrix `A`. Every other module depends on it, directly or indirectly.
- `forward.py` depends on `geometry.py` (it maps `s` through `A` to `t`, and
  needs cell centers to build synthetic truth fields).
- `prior.py` depends on `geometry.py` (needs cell centers to build the
  squared-exponential covariance) but not on `forward.py`.
- `inversion.py` depends on `forward.py` (via `A`, `d`) and `prior.py` (via
  `s0`, `Cs`).
- `diagnostics.py` depends on `inversion.py`'s outputs (`s_post`, `C_post`) and
  a ground truth from `forward.py`.
- `experiments.py` wires `geometry`, `forward`, `prior`, `inversion`, and
  `diagnostics` together into Experiments I-III, using `config.py` for
  reproducible parameterization.
- `plots.py` consumes `experiments.py` outputs and produces figures.
- `scripts/run_experiment.py` and `notebooks/report.ipynb` are thin consumers
  at the top of the chain: YAML config in, `experiments.py` + `plots.py` out.
  No mathematical logic lives above `experiments.py`.

## Cell ordering convention

Cell index `k` corresponds to grid position `(i, j)` via `k = i * n + j`
(row-major), with `i` indexing the `y` direction and `j` indexing the `x`
direction. `cell_centers[k]` corresponds exactly to column `k` of `A`, element
`k` of `s`, and row/column `k` of `Cs`. This is fixed once in `geometry.py` and
never deviated from elsewhere.

## Two different meanings of "calibrated"

> Experiment II generates the truth from the same prior the inversion assumes
> (`s_true ~ N(s0, Cs)`, `d | s_true ~ N(A s_true, Sigma_d)`). Under this
> generative model, the posterior is exactly the correct conditional
> distribution, so `Q ~ chi2(n_cells)` is *guaranteed* by the model being
> self-consistent — this experiment validates the implementation, not a deeper
> statistical claim. Experiment III fixes `s_true` as a deterministic field not
> regarded as a draw from the prior. There is *no general guarantee* that
> `Q ~ chi2(n_cells)` here; whether the posterior remains well-calibrated depends on
> how well the fixed truth matches what the prior expects. Never write "the
> posterior is calibrated" without specifying which of these two senses is
> meant.

## Spatial correlation and pooled calibration diagnostics

Per-cell standardized errors `z_j` are spatially correlated within a single
realization: the squared-exponential prior `Cs` induces spatial dependence
between nearby cells, and this dependence is then modified (not simply
inherited) by the likelihood/posterior update that produces `C_post` — the
posterior's correlation structure is not assumed to equal the prior's. A
diagnostic that pools `z` values across cells (e.g. a
goodness-of-fit test against `N(0, 1)`) cannot treat those pooled values as
independent observations, and is not used in this project's tests for that
reason. (A pooled Kolmogorov-Smirnov test of this kind was tried during
development and removed from `test_calibration.py` after confirming that it
failed reliably at the project's real grid scale even though the underlying
model and inference code were correct — the small p-value reflected the
violated independence assumption, not miscalibration.)

Per-realization statistics such as `Q` do not have this problem: each
realization's `Q` is computed from one independent draw of `s_true` and
`epsilon`, so pooling `Q` *across realizations* is valid, and
`test_calibration.py`'s `Q`-vs-`chi2(n_cells)` check relies on exactly that.
Where per-cell behavior needs checking, `test_calibration.py` instead runs
separate KS tests against `N(0, 1)` for a small number of individual,
deterministically chosen cells (each test pooling only across realizations
for that one cell), Bonferroni-corrected across those tests.

(This project has not attempted to quantify an "effective independent
sample size" for the pooled case — only that treating the pooled count as
the true sample size is wrong.)

## Why Experiment III's per-cell `z` is not `N(0, 1)`, even when well matched

The per-cell KS/z-score diagnostics used in Experiment II rely on `s_true`
itself being a draw from the prior: conditional on the data, `s_true - s_post
~ N(0, C_post)` is exact, so standardizing by `sqrt(diag(C_post))` gives
`z_j ~ N(0, 1)` by construction. Experiment III does not have this property.
There, `s_true` is fixed and only the observation noise `epsilon` is
resampled across realizations, so the *frequentist sampling distribution* of
`z_j` (over repeated noise draws, truth held fixed) is governed by a
different covariance than `C_post`.

Write the posterior mean's fixed-truth error as

$$
s_{\rm true} - s_{\rm post} = (I - KA)(s_{\rm true} - s_0) - K\epsilon,
\qquad
K = C_s A^\top (A C_s A^\top + \Sigma_d)^{-1},
$$

which decomposes into a **deterministic regularization bias**
`(I - KA)(s_true - s0)` (fixed, since `s_true` is fixed — it does not vary
across realizations) and a **noise-driven term** `-K epsilon` (the only
random part, since only `epsilon` varies). The repeated-noise sampling
covariance of `s_post` is therefore

$$
V_{\rm post} = C_{\rm post}\, A^\top \Sigma_d^{-1} A\, C_{\rm post},
$$

which is generally *not* equal to `C_post` itself (`C_post` is the Bayesian
posterior covariance, which also accounts for prior uncertainty about which
`s_true` might be true — a component that is simply absent when `s_true` is
fixed). Standardizing the fixed-truth sampling variability by `C_post`'s
diagonal instead of `V_post`'s gives a per-cell ratio

$$
r_j = \sqrt{\frac{(V_{\rm post})_{jj}}{(C_{\rm post})_{jj}}},
$$

which is the *theoretically expected* `std(z_j)` under repeated noise for a
fixed truth — not 1. Concretely:

- for a well-matched fixed truth (the smooth Gaussian-anomaly truth), `r_j`
  is well below 1 (empirically `r_j` averages ≈0.27 across cells for the
  frozen Experiment III smooth baseline, matching the empirical `std(z_j)`
  per cell to within ≈0.01), producing a *systematically narrow* `z`
  distribution;
- for a poorly matched fixed truth (the sharp checkerboard truth), the
  deterministic bias `(I - KA)(s_true - s0)` can be large relative to the
  posterior standard deviation, producing strongly off-center per-cell `z`
  distributions and substantial spatial variation in their means across
  cells. **This should not be interpreted as increased within-cell sampling
  variance across repeated noise draws; the within-cell repeated-noise
  variance is governed by `V_post`**, which does not depend on `s_true` at
  all (it is identical for the smooth and sharp truths — only the bias term
  differs between them). Concretely, for the frozen Experiment III
  baselines, `tr(C_post^-1 V_post) ≈ 32.93` is the same for both cases,
  while the bias term `b^T C_post^-1 b` is `≈0.28` for the smooth truth and
  `≈2.195e7` for the sharp truth — the entire ~7-order-of-magnitude gap in
  `E[Q]` is a bias effect, not a within-cell-variance effect. The key
  distinction to keep explicit:
  - deterministic bias shifts the *mean* of the repeated-noise `z`
    distribution for a given cell;
  - `V_post` determines the *within-cell* repeated-noise variability;
  - spatial variation in the bias across cells can produce substantial
    variation in the observed per-cell `z` means;
  - none of this is evidence that the poorly matched case has larger
    within-cell sampling variance — do not describe it as "wider" without
    this distinction.

**Neither behavior should be interpreted using the Experiment-II `chi2(n_cells)`
calibration benchmark**: fixing `s_true` changes the repeated-noise sampling
distribution, removing the prior's contribution to the variance decomposition that
the Experiment-II guarantee relies on. Note also that the *pooled*
`std(z)` reported elsewhere for a fixed truth (pooling across cells, not
just realizations) is not directly comparable to `r_j`: by the law of total
variance, `Var(pooled z) = mean_j(Var(z_j)) + Var_j(mean(z_j))`, and the
per-cell theoretical `r_j` only predicts the first (within-cell) term — the
second term (the spread, across cells, of the *bias* `(I-KA)(s_true-s0)`,
which varies spatially for a non-uniform truth like the Gaussian anomaly)
adds further pooled variance on top.

**Coverage and the fixed-truth behavior of `Q` are the appropriate Experiment
III diagnostics** (see the "Two different meanings of calibrated" note
above). The five-cell Bonferroni-corrected KS tests in `test_calibration.py`
are Experiment-II-specific reference checks — under Experiment III they are
expected to fail regardless of how well-matched the fixed truth is, for the
reasons above, and must not be read as a coverage/calibration verdict for
Experiment III.

## Experiment III: investigating the boundary/interior coverage reversal

The sharp/checkerboard Experiment III baseline (`block_size=4`) shows a
boundary-vs-interior difference in empirical coverage that is small but
consistently present, and ad hoc exploration at larger block sizes
(`block_size=8`, `10`, diagnostic only — no baseline was frozen for either)
found the reversal growing stronger. This section documents the follow-up
investigation into that reversal, using the fixed-truth decomposition and
notation (`K`, `b`, `V_post`) introduced in the section above, and states
plainly what was and was not established.

### Theoretical `E[Q]` validation

Taking the expectation of `Q = (s_true - s_post)^T C_post^-1 (s_true -
s_post)` over the repeated-noise randomness, with `s_true - s_post = b -
K epsilon` as above, gives

$$
E[Q] = \operatorname{tr}\!\left(C_{\rm post}^{-1} V_{\rm post}\right)
       + b^\top C_{\rm post}^{-1} b,
$$

i.e. a noise-driven trace term plus a deterministic-bias term. Both terms
were computed directly from the frozen Experiment III configuration's `A`,
`C_post`, and `Sigma_d`, and compared against the empirical mean `Q` from
the frozen `.npz` baselines:

| case | trace term `tr(C_post^-1 V_post)` | bias term `b^T C_post^-1 b` | theoretical `E[Q]` | empirical mean `Q` | relative discrepancy |
|---|---|---|---|---|---|
| smooth | 32.928 | 0.282 | 33.210 | 33.173 | 0.112% |
| sharp | 32.928 | 21,954,034.70 | 21,954,067.63 | 21,954,065.25 | 0.00001% |

Both cases agree with the empirical mean to well within Monte Carlo sampling
noise for `N=1000` realizations. This confirms — quantitatively, not just
qualitatively — that the ~660,000x gap between the smooth and sharp mean
`Q` is explained by the bias term alone: the trace/noise term is *identical*
between the two cases (it depends only on `A`, `C_post`, and `Sigma_d`, none
of which involve `s_true`), so every order of magnitude of the gap comes
from `b^T C_post^-1 b`.

### Bias-normalized spatial comparison (`block_size=4`)

Ray density and marginal posterior variance (`diag(C_post)`) were compared
between boundary and interior cells and found nearly indistinguishable (see
the ray-density and posterior-variance checks elsewhere in this
investigation's history). To check whether the *bias* or *repeated-noise*
terms specifically differ spatially even when their raw magnitudes look
similar, two normalized per-cell ratios were computed at `block_size=4`:

- **A**: `|b_j| / sqrt((C_post)_jj)` — bias relative to claimed posterior
  uncertainty;
- **B**: `sqrt((V_post)_jj) / sqrt((C_post)_jj)` — repeated-noise
  variability relative to claimed posterior uncertainty (this is exactly
  `r_j` from the section above).

| ratio | boundary (n=300) | interior (n=100) | boundary/interior |
|---|---|---|---|
| A | mean 1.936, median 2.054, min 0.021, max 5.038 | mean 1.922, median 2.009, min 0.232, max 4.567 | 1.007 |
| B | mean 0.275, median 0.260, min 0.161, max 0.456 | mean 0.269, median 0.267, min 0.154, max 0.434 | 1.024 |

Boundary and interior are indistinguishable on both ratios at this block
size. Ray density, posterior variance, and bias/noise normalized by
uncertainty were all checked, and **none explains the observed coverage
difference at `block_size=4`** — the effect may arise from higher-order
spatial structure not captured by these diagnostics.

### Extending the ray-density and posterior-variance checks to larger block sizes

The same two confounds (ray density, `diag(C_post)`) were re-examined,
purely by recomputing the boundary/interior grouping at `block_size=8` and
`10` against the *same* `A`/`C_post` (neither depends on `block_size`; only
which cells count as "boundary" does) — no re-inversion and no new
Experiment III baseline was involved:

| block_size | boundary ray count | interior ray count | boundary post. variance | interior post. variance |
|---|---|---|---|---|
| 4 | 8.83 | 8.92 | 0.00946 | 0.00922 |
| 8 | 8.85 | 8.85 | 0.00947 | 0.00934 |
| 10 | 7.81 | 9.44 | 0.01012 | 0.00899 |

`block_size=8` shows no meaningful difference, consistent with
`block_size=4`. `block_size=10` shows a real difference reappearing:
boundary ray count is ~17% lower and boundary posterior variance ~13%
higher than interior — right at the block size where the coverage reversal
was previously observed to be strongest.

A specific caveat applies to this `block_size=10` finding: at that size (a
2x2 checkerboard on the 20x20 grid), the "boundary" cell classification may
structurally overlap with the domain's actual outer edge, which
independently has lower ray coverage under the source-left/receiver-right
acquisition geometry. This means the `block_size=10` result may reflect a
confound between "checkerboard boundary" and "domain edge" rather than a
checkerboard-specific effect — this was **not investigated further**.

### Final status

The mechanism behind the boundary/interior coverage reversal is
**unresolved at `block_size=4`** (the frozen baseline): ray density,
posterior variance, and both bias/noise normalized ratios were checked and
none showed a meaningful boundary/interior difference there. A candidate
confound (overlap between checkerboard-boundary and domain-edge cell
classification) was **identified but not confirmed** at `block_size=10`.
This investigation was **deliberately stopped** at this point — no
eigenmode analysis, no additional block sizes, no alternative acquisition
geometries were pursued. This is treated as a valid, deliberate stopping
point for this project, not an unfinished result.

## Out of scope

This is deliberately a finite-dimensional linear-Gaussian problem with an exact
closed-form posterior. The following are excluded by design, not by oversight:
iterative optimization, MCMC, neural networks or neural-operator surrogates,
curved-ray or nonlinear eikonal solvers, and external tomography frameworks
(e.g. SimPEG).

The project's scope is also limited to Experiments I-III: a systematic
acquisition-geometry sweep, a controlled prior-misspecification experiment,
and a controlled noise-misspecification experiment are not part of this
project. The fixed-truth comparison in Experiment III (see above) already
demonstrates the central phenomenon those experiments would also
illustrate — a well-defined posterior failing to describe physical
uncertainty under model mismatch — via a mismatched truth rather than a
mismatched prior or noise model.
