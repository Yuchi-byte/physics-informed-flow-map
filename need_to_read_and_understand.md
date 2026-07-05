
1. The 4 rows are NOT the mc_samples — and yes, visualizing the mc draws would be worth it

The grid's 4 row-pairs are the n_samples=4 posterior trajectories (one per noise seed). mc_samples=4 is something else entirely: at every step, for each trajectory, mfm_g pushes 4 fresh noises through the one-step conditional map v(0,1,ε | t_cond=t, x_cond=x_t) to Monte-Carlo-estimate the reward-tilted drift (the IWAE estimator). Those 16 draws per step are consumed internally and currently never visualized — the "prediction" row shows the single Tweedie estimate instead.

Visualizing them is genuinely a good idea, and not just for pretty pictures: the journal's top-ranked hypothesis for why mfm_g trails flow_tilt is that the one-step posterior p(x1|x_t) is inaccurate or insensitive to x_cond — and a per-checkpoint grid of the mc draws (say for sample 0: rows = 4 draws, cols = t) is exactly the "posterior-fidelity check" the 2026-07-02 pickup entry lists as next-action (a). The steering helper already returns the draws (x1_samples, shape [B, mc, C, H, W]), so it's a small extension of the same hook. Say the word and I'll add it.

2. How the final image relates to the 4 seeds

There is no single "final inverted image" — the method's output is the set of 4 maps. Each seed is an independent draw from the (guidance-tilted) posterior p(v | d_obs): same observed data, same prior, different starting noise, 4 full guided trajectories, 4 velocity maps that all fit the data. What you see downstream: the inversion.png figure shows sample 0 as a representative draw (arbitrary, not selected), while the metrics (mae_mean etc.) are averages across all 4. The disagreement between the 4 final columns in the trajectory grid — mostly the middle layers — is the posterior uncertainty, i.e. exactly the part of the model the seismic data doesn't pin down. In a deployment you'd either pick the min-misfit sample (ground-truth-free) or report the posterior mean ± spread; the journal found that this selection step, not prior quality, is the current bottleneck — which is why we keep all 4 rather than collapsing them.

3. Is FMRG unstable? Yes — in a specific, diagnosable way

It's not that any config perturbation breaks it; it's that guidance strength has a cliff: g=0.3 works (MAE 0.137), g=1.0 breaks (0.676), and g=1.0 without gradient normalization diverges outright (0.884 — guidance increases the data error, because the raw seismic gradient is enormous). Two structural reasons: FMRG's time weight wt=(1-t)·t_next concentrates guidance late, so a too-large step arrives when the map is nearly formed and wrecks it; and its inner gradient lives in x1-space without the manifold-projecting chain rule flow_tilt gets, so overshoot goes off-manifold with nothing pulling it back. flow_tilt is far more forgiving (worked at g=1 across every prior today). The practical bug on our side: fmrg_e.yaml shipped with the proven-divergent defaults (g=1.0, normalize=false), so any un-overridden fmrg run produced garbage. Fixed — defaults are now the swept-best g=0.3/normalize=true (da3a78d).

## flow matching and mfm training method overview 


Correcting the mental model

Your version: "FM is self-distillation from the linear interpolant → flow map trained on top via consistency → MFM trained with the flow map as teacher." Three corrections:

1. FM has no teacher at all. It's plain supervised regression: sample noise x0, data x1, time s; the input is the interpolant Is = (1−s)·x0 + s·x1 and the target is its known derivative x1 − x0. The interpolant isn't a teacher network — it's the label. ("Self-distillation" does exist in this ecosystem, but it refers to the flow map's off-diagonal term in the from-scratch mf mode, where the model's own diagonal bootstraps its off-diagonal.)
2. There is no chain of three trainings. The paper (and our repo) goes FM → MFM in one hop. Flow-map-ness (off-diagonal) and meta-ness (conditioning) are trained jointly inside the same run, on the same network, at every gradient step.
3. Yes — 0001→0002 IS GLASS distillation. Your 0002 run used distillation_type=esd_teacher with 0001 as the teacher; extract_posterior_velocity (losses.py:103) is the GLASS formula. This mirrors upstream's recommended recipe exactly (their config_train.yaml: init_from_dmf: true, distillation_type: esd_teacher).

Step-by-step: what actually happens

Run 1 — experiment 0001 (train the teacher). Every gradient step:
1. Sample batch of clean velocity maps x1, noise x0, times s ~ U(0,1).
2. Build Is = (1−s)·x0 + s·x1.
3. Regress v(s, s, Is, t_cond=0, x_cond=noise) onto x1 − x0. (Both time slots get s; the conditioning inputs are fed null values and their pathways are zero-init/gated — dead.)
4. That's it. Off-diagonal disabled (warmup parked at 10¹²). After 100 epochs → step_99_ema.pt.

Run 2 — experiment 0002 (esd_teacher). Load 0001's checkpoint twice: once as the frozen teacher, once as the student's initialization (warm-start), then copy x_embedder → x_cond_embedder to wake up the conditioning pathway. Every gradient step, per sample in the batch:

1. Draw the conditioning state. Flip a coin: with prob uncond_prob=0.1 set t_cond = 0 (unconditional); else t_cond ~ U(0,1)^power. Draw fresh noise ε_c and build x_cond = (1−t_cond)·ε_c + t_cond·x1. Crucially, x_cond is built from the same x1 — that shared endpoint is the only reason a posterior is learnable.
2. Diagonal term (our repo: from data, since data_fm=True). Fresh x0, s ~ U(0,1), Is as before; regress v(s, s, Is, t_cond, x_cond) onto x1 − x0. Subtle but important: because the network also sees x_cond (which leaks information about x1), the regression's conditional mean is the posterior velocity E[x1−x0 | Is, x_cond] — so even this "plain FM" term is training the meta-conditioning.
3. Off-diagonal term (the flow-map part). Sample a pair s < u (see schedules below). Then:
  - GLASS step: compute t*  and x* = w₁·x_cond + w₂·Is (a linear "sufficient statistic" merging the current state and the conditioning state — losses.py:121-129), call the frozen teacher once, on its diagonal: v_teacher(t*, t*, x*), and analytically rescale the output. Result: vss = the conditional instantaneous velocity at (s, Is) given (t_cond, x_cond) — extracted from a teacher that was never trained conditionally. That's the whole magic of GLASS: linear interpolant + Gaussian noise ⇒ the posterior drift is a closed-form function of the marginal drift.
  - Tangent target: forward-mode JVP through the student: differentiate v(s, u, Is, …) in direction (ds=1, du=0, dx=vss), giving target = vss + (u−s)·jvp (losses.py:471-482). This enforces the self-consistency identity "a big jump must agree with composing its own instantaneous motion."
  - Regress the student's v(s, u, Is, t_cond, x_cond) onto that (stop-gradded) target.
4. Total = adaptive-weighted diagonal + off-diagonal → one optimizer step → EMA update.

The schedules (sample_s_u, sample_t_cond at losses.py:6-57) are the "staging" you were intuiting — but it's a curriculum inside one run, not separate runs: phase 1 (num_warmup_steps) trains diagonal only; phase 2 anneals the jump size u−s open linearly until anneal_end_step; phase 3 = all jumps. Your run had warmup=0 (fine for teacher mode — the warm-start already IS a trained diagonal) and jump annealing over the first 20k steps.

Your specific decision questions

data_fm vs distill_fm (diagonal from data vs teacher). My recommendation: keep data_fm=True (current behavior), and don't sweat it. Reasoning: upstream chose teacher-diagonal because their teacher (DMF-XL, heavily trained ImageNet model) is far better than anything they could re-learn from data in a short distillation run, and fixed targets are lower-variance. Your situation is different: 0001 and 0002 train on the same OpenFWI data for comparable budgets, so your teacher has no quality edge — data targets let the student surpass it rather than inherit its errors. The theoretical downside (diagonal-from-data is noisier) is handled by the adaptive weighting. If you ever distill from a much stronger, longer-trained 0001, flip to distill_fm=true then. Worth one sentence in the journal, not a code change now.

t_cond_power (1.0 vs upstream's 2.0). Power 2 concentrates training on small t_cond — the nearly-unconditional posteriors, which are the hardest (huge posterior variance) and, notably, the regime where inference-time steering starts (early in the trajectory, t small). Since your whole use case is steering from early/mid trajectory states, I'd switch to 2.0 for the teacher recipe to match the paper — it front-loads capacity exactly where MFM-G queries the model. Cheap change, aligned with the validated recipe.

uncond_prob — what and why. It's mfm's t_cond_0_rate: the probability a training sample gets t_cond = 0 exactly. Yes, the paper's code uses it, same value 0.1. It's needed because t_cond=0 is not just another value — it's the unconditional slice you use for prior sampling (both your ODE sampler and few-step sampler call with t_cond=0). If you sampled t_cond continuously, the event t_cond = 0 would have probability zero and the exact null slice would only be learned by extrapolation. The atom at 0 guarantees ~10% of all training explicitly maintains the unconditional prior. It's structurally the same idea as CFG's label-dropout: always keep the unconditional model alive inside the conditional one.

The three-way experiment — yes, and here's the design

Your comparison is well-posed, one nuance: FM can traverse time, just only by many small Euler steps (it knows only instantaneous velocity); the flow map amortizes the integral so one network call jumps s→u. The clean triple:

┌──────────────────────┬──────────────┬─────────────────────┬───────────────────────────────────────────────────────────────────┐
│      experiment      │ off-diagonal │    conditioning     │                        samplers to report                         │
├──────────────────────┼──────────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 0001 FM              │ ✗            │ ✗                   │ ODE-200 only (few-step would be garbage — untrained off-diagonal) │
├──────────────────────┼──────────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 0002-fm-uncond (new) │ ✓            │ ✗ (uncond_prob=1.0) │ ODE-200 + few-step-4                                              │
├──────────────────────┼──────────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 0002 MFM             │ ✓            │ ✓ (uncond_prob=0.1) │ ODE-200 + few-step-4 + posterior panel                            │
└──────────────────────┴──────────────┴─────────────────────┴───────────────────────────────────────────────────────────────────┘

The new middle column is one Hydra config (uncond_prob=1.0, teacher mode otherwise identical). Bonus, free comparison: evaluate the existing MFM checkpoint at t_cond=0 against the dedicated unconditional flow map — that measures what the 90% conditional training cost the unconditional slice.

CFG, and MFM-G vs MFM-GF

Why no CFG: classifier-free guidance sharpens samples toward a class/text label by extrapolating between a conditional and unconditional model — it presupposes labels. Your OpenFWI prior is unconditional (num_classes=0); there is no label to guide toward. The conditioning you actually care about — seismic data d — enters through inference-time tilting (reward r(v) = −‖F(v)−d‖²/2σ²), which is a different mechanism entirely (Doob h-transform on the drift, not label guidance). Upstream's model_guidance machinery (amortized CFG-scale embedding) exists to chase ImageNet FID; it has nothing to attach to here.

Why MFM-G, not MFM-GF: both estimate the steering drift ∇V_t from posterior draws, differing in what they need:
- MFM-GF (gradient-free): reward-weighted average of draws, Σᵢ x̂₁⁽ⁱ⁾ e^{rᵢ}/Σᵢ e^{rᵢ}. Only needs reward evaluations — its niche is non-differentiable rewards. Price: self-normalized importance weighting degenerates in high dimensions (one draw dominates), needs many samples.
- MFM-G (gradient-based/IWAE): ∇ₓ log (1/N) Σᵢ e^{r(Φ(εᵢ; t, x))} — differentiates through both the reward and the one-step posterior map. Needs ∇r.

Your reward is differentiable — F is Deepwave, a differentiable wave solver — so you're squarely in MFM-G's territory, and the paper shows MFM-G dominates GF/DPS/Best-of-N even at N=1. GF is documented in your prior-work.html as the fallback for a future non-differentiable misfit. Using GF when you have gradients would be throwing away information.