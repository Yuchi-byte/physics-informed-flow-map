- (highest priority) prompt for claude next session: Continue the full-OpenFWI prior training program. Read docs/superpowers/plans/2026-07-06-full-openfwi-priors.md (checkboxes show progress) and docs/superpowers/specs/2026-07-06-full-openfwi-priors-design.md first.
▎
▎ State: Tasks 0–4 are done and committed (through bd4ca2f) — all 470k velocity maps for the 10 families are verified under data/openfwi (8.8 GB, velocity-only), the loader has the per-family split/hflip/fingerprint, bf16 + dit_b presets are in, per-family eval is wired and smoke-tested in all three experiments, and the openfwi_full yamls are finalized and compose-validated.
▎
▎ Next is Task 5–7 (the training runs). I am running this on: [FILL IN: which GPU/pod you got, e.g. "a new RTX PRO 6000 96GB pod" / "this same PRO 4500"].
▎ 1. If this is a new pod, run uv sync and re-run the throughput calibration from plan Task 4 (the yaml comments have the 2026-07-07 PRO 4500 numbers to compare against: dit_b bf16 bs128 = 297 maps/s, 24.3 GB; bs256 OOMs on 32 GB — on a 32 GB card override training.batch_size=128 training.lr=1.4e-4).
▎ 2. Launch 0001: uv run python experiments/0001_flow_matching/run.py experiment=openfwi_full and 0003 in parallel if I have a second pod. After 0001 finishes and passes review (per-family val losses + sample grids), launch 0002 with experiment=openfwi_full training=teacher training.teacher_ckpt=<0001 final EMA checkpoint>.
▎ 3. Review each finished run per plan Tasks 5–7 (per-family metrics, journal entry, upload final EMA + final raw as wandb artifacts — nothing more, my wandb quota is 5 GB).
▎ Then proceed to Task 8 (201-target inversion benchmark) and Task 9 (verification gates, prior zoo, deletion) per the plan. Legacy target 6044 provenance is already pinned in data/inversion_bench/legacy_6044_provenance.json.



- (highest priority) Pick up the 0002 flow-map work from yesterday (see memory: 0002-batch128-bf16-handoff).

1. Check the overnight 0002 openfwi batch-128 run (wandb r3majqyl, dir
   /workspace/runs/0002_flow_map/openfwi_mf_2026-07-07T00-37-28Z). If it completed
   100 epochs, summarize final val/loss and the new off-diagonal metrics
   (val/fewstep_ode_gap*, val/jump_consistency*) and how they evolved. If it was
   killed again (like the epoch-80 SIGKILL the night before), report the last epoch
   and what survived.

2. bf16 probe, if it didn't already run overnight (look for a bf16 probe log in
   /workspace/runs/0002_flow_map/_probes/): run
   WANDB_MODE=offline uv run python experiments/0002_flow_map/run.py experiment=openfwi \
     training.batch_size=128 training.n_epochs=1 training.flow_map_warmup_steps=0 \
     training.warmup_steps=0 training.eval_every_epochs=0 training.precision=bf16
   Compare it/s, peak GPU memory, and the steps.jsonl loss trajectory against the
   fp32 reference (_probes/0002_b128_probe.log: 2.85 it/s, 28.3 GB peak). If parity
   holds, also probe whether batch 256 fits under bf16 (it OOMs in fp32).

3. Before launching any new long run: implement resume support (optimizer state +
   epoch in checkpoints, a training.resume_from flag for 0001/0002) and switch to
   setsid launches, then decide the bf16 (maybe batch-256) relaunch with me.






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

