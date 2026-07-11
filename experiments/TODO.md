- (highest priority) Handoff state 2026-07-11 (updated by the PRO-6000-pod session before
  pod closure; plan in docs/superpowers/plans/2026-07-06-full-openfwi-priors.md):

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

  TRAINED, REVIEW PENDING — Task 7: 0002 definitive FINISHED 2026-07-11 on the PRO 6000:
  runs/0002_flow_map/openfwi_mf_2026-07-08T10-23-48Z, wandb l4l4ecfp. 90/90 epochs @
  bs128 / lr 1.4e-4 / bf16 (86.3 GB, 1.25 it/s; bs256 AND bs192 OOM on 96 GB — teacher
  regime needs ~0.65 GB/map of activations, bs128 is the ceiling on any current card).
  Final val/loss 206.3; checkpoints step_89{,_ema}.pt on the volume; wandb artifacts came
  out slim automatically via the new harness (openfwi-flowmap{,-ema}:v29, 1.08 GB;
  account ~3.2/5 GB). The 2026-07-11 inversion sweep already used step_89_ema (journal:
  distillation parity with the teacher, OT wins 10/10).
  STILL TO DO for Task 7 sign-off:
  (a) the formal training review + "0002_flow_map/openfwi_full DEFINITIVE" journal entry:
      per-family val losses from the run's metrics.jsonl, per-family energy distance
      (in the run's wandb summary under val/energy/<family>) vs the real-mixture floor
      control — floors + method in the 0003 journal entry (2026-07-08); NOTE ca17d47's
      correction: control seed noise is ~±0.5, deviations under that are "at floor";
      final sample grid vs val_reference.png.
  (b) quality-vs-NFE on the full 10-family priors — checklist item under plan Task 7
      (experiments/0005_analysis/prior_quality_vs_nfe.py is FlatVel_A-specific today;
      full priors are dit_b and 0003-full is the DiT denoiser, not the UNet).

  NEXT — Task 8 seismic extraction + Task 9 gates (any cheap GPU pod works; ≥16 vCPU /
  ≥64 GB RAM, ~50 GB free volume space for the transient download):
  seismic: transiently download ONLY the data/seis files containing the 201 manifest rows
  (~30-40 GB), extract seismic/<id>.npy (5x1000x70 each, ~280 MB total) into
  data/inversion_bench/seismic/ (git-ignored by design), delete transients, add
  manifest<->seismic consistency test. Then Task 9: prior-zoo doc, Gate 1 (each EMA
  artifact reloads from wandb + samples 64), Gate 2 (energy within factor of baseline),
  Gate 3 (flow_tilt + mfm_g on flatvel_a_legacy_6044 reproduces journal numbers — NOTE
  the manifest records in_current_val=false for it: the full priors SAW this map in
  training; it's a continuity check, not an unseen-target claim), Gate 4 (benchmark
  artifact upload — ASK Lyra before any wandb upload, quota 5 GB, ~3.2 used), then bulk
  deletion (keep velocity maps per spec §9.5).

  Pod notes for the next session: repo + runs live on the /workspace network volume (this
  working copy carries over; no re-clone). Run `uv run wandb login` first if wandb calls
  fail (fresh pods aren't logged in). Git history was rewritten 2026-07-08 to drop the
  benchmark data blobs (force-pushed) — any OLD clone elsewhere needs
  `git fetch && git reset --hard origin/main`, never `git pull`.







- (understand) Experiment 0003 diffusion. num_train_timesteps = 1000, and num_sample_steps = 200. How come those values are different? The structure of the configs for 0003 is not very consistent with that of 0001 and 0002. The number of parameters are also not present in the config -- should it have the same number of parameters to flow map? 

- (understand) There's ODE and SDE methods for the unconditional drfit calculation for sampling (maybe also for training?). What is used for mfm as specified in their paper? And which one is used for flow maps? 


- (low priority) (check and fix) Experiment 0001 records valuation loss too. But the section orders in wandb for 0001 is differnet to 0002. Reorder the sections so 0001 matches 0002. 
- (fix) in experiment 0004, also visualise the mc_samples for the mfm method. 
- (high priority) try different methods to improve: 

Frequency annealing matched to generative time — my top pick for cycle skipping. Classical multiscale FWI (Bunks et al., already cited in your research plan) fits low frequencies first because they're immune to cycle skipping. In guided sampling there's a beautiful structural alignment nobody has exploited cleanly: early reverse steps have rough x̂₀ estimates anyway, so guide against low-passed d_obs early and ramp the cutoff with sampler time t. It's nearly free (a frozen bank of filtered d_obs + a t-dependent misfit — drops straight into the new misfit infrastructure), attacks the same failure mode as OT from an orthogonal angle, and "misfit + schedule design for generative FWI" (L2 / OT / envelope × constant / annealed) makes a coherent paper section on its own. This is literally your research plan's A2 second bullet.

Source subsampling / encoding per guidance step — the compute-time lever. Wave solves dominate your budget (steps × mc_samples × n_samples). Classical FWI's answer is stochastic: use one random shot (or a random-polarity supershot, Krebs et al. 2009) per gradient step instead of all 5–10. In guided sampling the sampler's own stochasticity should average the shot noise over 200 steps — plausibly a 5× solve reduction at negligible quality cost. Trivial to implement (a shot_subsample knob on seismic_forward), and "stochastic source encoding for diffusion-guided FWI" appears unclaimed as a systematic study.

Noise + matched-σ posterior calibration, then twisted SMC — the thesis-critical one. Your journal already flags this as the unblocking experiment: under the noiseless inverse crime, no method can demonstrate posterior quality because σ is unprincipled. Adding noise with the likelihood σ matched to it makes calibration (coverage, CRPS — your evaluator already computes these) meaningful, and that's the setting where MFM-G's less-biased posterior should finally separate from Tweedie baselines. The natural method upgrade there is twisted SMC / Feynman-Kac steering over the flow trajectories (research plan A1): proper importance weighting corrects the DPS bias that no single-trajectory method can, and calibrated-posterior-FWI is a strong, defensible contribution.

Graph-space OT as the principled member of the misfit family — transports in the (t, amplitude) plane, needs no positivity hack, and directly fixes the zero-mean saturation we just diagnosed. Heavier (a small assignment problem per trace), but it slots into the same MisfitFn interface, and a comparison of OT variants under generative guidance — including the finding that Peng's shift-normalization gets its long-range sensitivity from the weighting, not the transport — is itself publishable.

