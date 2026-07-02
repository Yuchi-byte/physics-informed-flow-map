# FWI problem exploration

Empirical studies of *why* classical full-waveform inversion fails on the OpenFWI
setting used by `experiments/0004_inversion` — non-uniqueness, cycle skipping, and
whether "properly done" classical FWI closes the gap. Everything runs on the same
held-out target (seed-0 validation map, global index 6044, FlatVel-A) with the
differentiable Deepwave forward operator (`physics_informed_flow_map.physics.forward`):
70×70 grid at 10 m, 5 surface shots, 70 surface receivers, 15 Hz Ricker, 1 s at 1 ms.

Run any script from the repo root, e.g.
`uv run python FWI_problem_exploration/cycle_skipping_landscape.py`.

## Scripts

| script | figure(s) | question |
|---|---|---|
| `viz_flatvel_a.py` | `flatvel_a_viz.png` | what the FlatVel-A data looks like |
| `viz_seismic_displays.py` | `seismic_displays.png`, `seismic_spectra.png` | standard seismic displays & spectra |
| `classical_fwi_inversion.py` | `classical_fwi_forward.png`, `classical_fwi_nonuniqueness.png` | forward solve + non-uniqueness: 5 random starts → 5 different maps, all fitting the same `d_obs` |
| `cycle_skipping_landscape.py` | `cycle_skipping_landscape.png` | the L2 misfit landscape along `v = α·v_true`, full band vs low-passed, aggregate vs single traces |
| `cycle_skipping_escape.py` | `cycle_skipping_escape.png` | can multiscale / envelope misfits rescue FWI from a cycle-skipped start? |
| `cycle_skipping_barrier.py` | `cycle_skipping_barrier.png` | how far is a *trapped* model from truth; is there a real barrier between them; the "midpoint" test |
| `classical_fwi_regularised.py` | `classical_fwi_regularised.png` | does best-practice regularisation (multiscale + TV continuation + box constraints) close the gap? |

All FWI runs share the same protocol unless noted: Adam (lr 2e-2) on the velocity map in
normalised [-1, 1] space, 500 iterations, box constraints, starting model =
`0.75 × smooth(v_true)` (a realistic "poor background model", solidly cycle-skipped at
far offsets).

## Findings

**1. The aggregate L2 landscape is unimodal-but-saturating; the classic skip wells live
at the single-trace level** (`cycle_skipping_landscape.png`). Along `v = α·v_true` the
summed misfit over all 350 traces shows no spurious local minimum — it saturates into a
plateau beyond ~±20% velocity error because different traces skip at different α and
their oscillations average out. Individual far-offset traces keep the textbook wells,
e.g. at α ≈ 0.81 and 1.24, exactly where their arrival-time lag crosses ±1 wavelet
period. **A one-cycle skip corresponds to a ~20% (500–700 m/s) background velocity
error.** Low-passing to 3 Hz both widens the global valley and flattens the wells — the
multiscale rationale in one plot.

**2. Data fit says almost nothing about model quality here**
(`cycle_skipping_barrier.png`). The plain-L2 model "trapped" after 500 iterations fits
`d_obs` to a relative misfit of 1.7e-4 (traces visually identical to the observed data)
while being **526 m/s (17.8%) wrong on average, up to ~2,200 m/s locally**, with the
entire deep half replaced by high-frequency artifact soup. This is the quantitative
answer to "how different are the velocity maps behind `d_obs` and a well-fitting
`d_pred`": the ambiguity radius of a near-perfect fit spans a large fraction of the
velocity range.

**3. There is a genuine barrier between the trapped model and the truth — and relaxed
metrics remove it** (`cycle_skipping_barrier.png`, middle panel). Along the straight
segment `v(t) = (1-t)·v_trapped + t·v_true`, the raw L2 misfit rises **20×** before
falling to the global minimum: the two models sit in separate basins. The envelope
misfit lowers the barrier ~5×; the 3 Hz low-passed misfit is monotone downhill toward
the truth. Two corollaries:

- *The solution set is non-convex in model space*: the midpoint velocity
  `(v_true + v_trapped)/2` fits the data ~20× **worse** than either endpoint. Averaging
  two data-consistent models yields a data-inconsistent one, so "settling for a
  compromise map that works for both `d_obs` and `d_pred`" fails in the L2 geometry.
  The forward-map image is a curved manifold — the L2 midpoint of two cycle-shifted
  datasets (two half-amplitude wavelets) is not `simulate(v)` for any `v`. The
  *principled* version of ambiguity tolerance is changing the metric so a cycle shift
  is a small distance: envelope, optimal transport (Engquist & Froese; Métivier),
  adaptive waveform inversion (Warner & Guasch), or frequency continuation.
- *Trapped models betray themselves at low frequency*: the trapped model's 3 Hz misfit
  stays elevated even when its full-band misfit is tiny — a cheap "is this sample
  trapped?" diagnostic, e.g. for screening posterior samples.

**4. On this acquisition, the textbook fixes do not rescue the inversion**
(`cycle_skipping_escape.png`). From the same skipped start and budget, plain L2,
multiscale (3→6→full Hz) and envelope-first all reach ~1e-4 misfit at the **same wrong
MAE ≈ 0.35** (start: 0.449). Plain L2's loss curve descends smoothly — the optimiser is
never visibly "stuck"; it converges happily into the null space. The dominant pathology
of this surface-only, direct-wave-dominated, 700 m toy geometry is **non-uniqueness**,
not classic cycle skipping: even after the metric is convexified, vast families of
data-consistent wrong models remain, and each trace gets fitted on the wrong cycle
independently (which is how the artifact models of
`classical_fwi_nonuniqueness.png` arise).

**5. Best-practice regularisation cleans the image but does not close the gap**
(`classical_fwi_regularised.png`). Adding TV regularisation with weight continuation on
top of multiscale (the standard blocky-medium toolkit; same start, optimiser, and
budget) visibly suppresses the artifact soup — the recovered maps are much smoother —
but the error to truth does not improve: plain L2 MAE 526 m/s, multiscale 535 m/s,
multiscale+TV 548 m/s, all at ~1e-4 data misfit. The laterally averaged depth profile
shows why: all three variants track the truth well down to ~200 m and are then
essentially blind — below ~300 m the truth climbs from 2,600 to 3,800 m/s while every
reconstruction stays near ~2,300 m/s. With 15 Hz surface sources, surface receivers,
zero long-offset coverage and a 1 s record, the deep model is simply not constrained by
the data, and no misfit or hand-crafted regulariser can conjure that information.
"Doing FWI properly" fixes the *artifact* problem, not the *information* problem.
Selection among data-consistent models has to come from *prior information*; a TV prior
is a weak stand-in for the learned generative priors of experiments 0001/0002.

## Implication for the project

These experiments are the empirical motivation for the Camp A framing
(`docs/research/research-plan.html`): on OpenFWI-style problems the misfit *cannot*
choose among data-consistent models — by construction they all fit `d_obs`. Stability
must come from a prior over velocity maps, and the honest output is a **posterior
distribution** (the ambiguity made explicit) rather than a single compromise map. The
trapped models produced here are exactly the geologically absurd solutions a learned
prior assigns near-zero probability.

Caveats: single target (val map 6044), single toy geometry, scaled-truth starts. On
field-scale surveys with long offsets and diving waves, classic cycle skipping (where
multiscale genuinely rescues the inversion) plays a much larger role than it does here.
