- (highest priority) Handoff state 2026-07-08 (written by the 0001-pod session before pod
  closure; plan Tasks 5-7 in docs/superpowers/plans/2026-07-06-full-openfwi-priors.md):

  DONE — Task 5: 0001 definitive prior trained + reviewed + journaled (e80d348).
  Run runs/0001_flow_matching/openfwi_2026-07-07T11-19-11Z, wandb gmgx7psw. Per-family
  energy distances all at the perfect-model floor (real-mixture control ±0.28; the ~14.3
  for CurveVel_B/FlatVel_B is the intrinsic floor for those families, not a defect).
  Teacher checkpoint for 0002:
  runs/0001_flow_matching/openfwi_2026-07-07T11-19-11Z/checkpoints/step_89_ema.pt
  wandb artifacts slimmed to finals-only (1.07 GB); harness now uploads finals-only with
  train_state stripped (e80d348) — never re-upload best pairs, quota is 5 GB.

  DONE — Task 6: 0003 diffusion definitive trained + reviewed + journaled (2026-07-08).
  Run runs/0003_diffusion/openfwi_2026-07-07T23-26-16Z, wandb 984t0883; artifacts slimmed
  to finals-only 1.07 GB (EMA v28 + weights-only raw v29). Energy vs floor: A/Style
  families at floor, fault families +0.25..+0.49 (mild under-fit at 60 epochs, accepted),
  B-Vel ~0.7 below their ~14 intrinsic floors. The 5090 pod can be closed.

  IN FLIGHT — Task 7: 0002 definitive LAUNCHED 2026-07-08 on the PRO 6000 96 GB:
  runs/0002_flow_map/openfwi_mf_2026-07-08T10-23-48Z, wandb l4l4ecfp, ETA ~66 h
  (~2026-07-11 morning). MEASURED on 96 GB (probes in
  runs/0002_flow_map/_probes/0002_full_teacher_b{256,192,128}_bf16_96g_probe.log,
  script run_96g_probe.sh): bs256 AND bs192 both OOM at step 0 (>94.7 / >94.9 GB —
  spec §6's "bs256 fits on 96 GB" was wrong; teacher-regime activations ≈ 0.65 GB/map,
  so bs128 is the largest viable batch on ANY current card). Definitive run: bs128 /
  lr 1.4e-4 / bf16, 86.9 GB peak, 1.25 it/s = 160 maps/s, 90 epochs = 297,360 steps.
  Then Tasks 8-9 per the plan (Task 8's CPU-side selection/manifest can start while 0002
  trains; legacy 6044 provenance pinned in data/inversion_bench/legacy_6044_provenance.json).






- (fix) also visualise the predicted x at every denoising time step (for when the trajectory is to be visualised) together with the noisy xt for both 0001 and 0002 just like what claude did to 0003. 

- (fix) add quantitative validation loss for off-diagonal velocities in 0002 flow map.  Consider using: self-consistency against the fine ODE because i think this is essentially quite easy to implement now? 
- (understand) Experiment 0003 diffusion. num_train_timesteps = 1000, and num_sample_steps = 200. How come those values are different? The structure of the configs for 0003 is not very consistent with that of 0001 and 0002. The number of parameters are also not present in the config -- should it have the same number of parameters to flow map? 

- (understand) There's ODE and SDE methods for the unconditional drfit calculation for sampling (maybe also for training?). What is used for mfm as specified in their paper? And which one is used for flow maps? 


- (low priority) (check and fix) Experiment 0001 records valuation loss too. But the section orders in wandb for 0001 is differnet to 0002. Reorder the sections so 0001 matches 0002. 
- (fix) in experiment 0004, also visualise the mc_samples for the mfm method. 
- (high priority) try different methods to improve: 

Frequency annealing matched to generative time — my top pick for cycle skipping. Classical multiscale FWI (Bunks et al., already cited in your research plan) fits low frequencies first because they're immune to cycle skipping. In guided sampling there's a beautiful structural alignment nobody has exploited cleanly: early reverse steps have rough x̂₀ estimates anyway, so guide against low-passed d_obs early and ramp the cutoff with sampler time t. It's nearly free (a frozen bank of filtered d_obs + a t-dependent misfit — drops straight into the new misfit infrastructure), attacks the same failure mode as OT from an orthogonal angle, and "misfit + schedule design for generative FWI" (L2 / OT / envelope × constant / annealed) makes a coherent paper section on its own. This is literally your research plan's A2 second bullet.

Source subsampling / encoding per guidance step — the compute-time lever. Wave solves dominate your budget (steps × mc_samples × n_samples). Classical FWI's answer is stochastic: use one random shot (or a random-polarity supershot, Krebs et al. 2009) per gradient step instead of all 5–10. In guided sampling the sampler's own stochasticity should average the shot noise over 200 steps — plausibly a 5× solve reduction at negligible quality cost. Trivial to implement (a shot_subsample knob on seismic_forward), and "stochastic source encoding for diffusion-guided FWI" appears unclaimed as a systematic study.

Noise + matched-σ posterior calibration, then twisted SMC — the thesis-critical one. Your journal already flags this as the unblocking experiment: under the noiseless inverse crime, no method can demonstrate posterior quality because σ is unprincipled. Adding noise with the likelihood σ matched to it makes calibration (coverage, CRPS — your evaluator already computes these) meaningful, and that's the setting where MFM-G's less-biased posterior should finally separate from Tweedie baselines. The natural method upgrade there is twisted SMC / Feynman-Kac steering over the flow trajectories (research plan A1): proper importance weighting corrects the DPS bias that no single-trajectory method can, and calibrated-posterior-FWI is a strong, defensible contribution.

Graph-space OT as the principled member of the misfit family — transports in the (t, amplitude) plane, needs no positivity hack, and directly fixes the zero-mean saturation we just diagnosed. Heavier (a small assignment problem per trace), but it slots into the same MisfitFn interface, and a comparison of OT variants under generative guidance — including the finding that Peng's shift-normalization gets its long-range sensitivity from the weighting, not the transport — is itself publishable.

