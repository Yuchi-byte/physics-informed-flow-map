- (fix) also visualise the predicted x at every denoising time step (for when the trajectory is to be visualised) together with the noisy xt for both 0001 and 0002 just like what claude did to 0003. 
- (check and fix) The experiment results from 0004 doesn't seem to be recording the correct configuration parameters, or we didn't run the correct experiments. Eg. for the experiment 'inversion-diffusion-dps', the drift estimator should be dps, but in wandb's config it's recorded to be iwae. Actually, even for flow-tilt, the drift estimator is recorded as iwae whilst it's in reality dps as well. 
- (high priority) (Understand and fix) Flow matching, flow map and meta flow map should be three different models, every one being dependent on the previous one. And they should have more and more parameters. But I am seeing that we only have two models, and they have the same number of parameters (8,035,600) which shouldn't be the case. Inspect the current implementation, and devise a plan for fixing those issues. And I thought  flow map will have much less sampling steps (currently 200 for both of them) than flow matching, though 0002 has an additional 'few_step' variable, but not sure how it interacts with sampler_steps. Also 0002 actually has the longest time metric compared to the other two, which is counter-intuitive, so this makes me doubt whether it's been impelmented correctly. 
- (understand) The experiment 0002 result shows that validation loss increases before decreasing with epoch. But how can validatino loss be so low when we just started the training loop? Let's think about how validation loss is calculated: it's the l2 error between the sampled final image with xxxx??? wait i think it's different to both the training loss (difference in model's predicted flow and target flow) and the reward loss (difference between predicted velocity's seismic data and the labelled and actual seismic data) 

- (understand and check) Experiment 0002 has 2000 warmup_steps -- are those too many? I don't understand why it's necessary to keep the uncond_prob variable -- is the mfm paper using this variable too? What's the difference between sample_few_steps and samples? Also the sample_few_steps  don't yet have their seeds fixed. make sure to fix this. 
- (understand) Experiment 0003 diffusion. num_train_timesteps = 1000, and num_sample_steps = 200. How come those values are different? The structure of the configs for 0003 is not very consistent with that of 0001 and 0002. The number of parameters are also not present in the config -- should it have the same number of parameters to flow map? 

- (understand) There's ODE and SDE methods for the unconditional drfit calculation for sampling (maybe also for training?). What is used for mfm as specified in their paper? And which one is used for flow maps? 


- (fix) In misfit calculation, rather than using the true d_obs, the code uses simulate (v_true) to get d_obs. Why not just use the true d_obs directly? 
- (low priority) (check and fix) Experiment 0001 records valuation loss too. But the section orders in wandb for 0001 is differnet to 0002. Reorder the sections so 0001 matches 0002. 
- (high priority) Pick up commit 035dbe7 "feat(0004): guidance misfit knob — Peng et al. OT potential vs L2" (git pull first). It adds method.misfit=l2|ot to 0004_inversion: `ot` is the Peng et al. OT data-consistency potential (physics_informed_flow_map/physics/misfit.py, OTMisfit), threaded as an optional misfit_fn through all guidance samplers. misfit=l2 is the default and leaves every existing path byte-identical. Read the JOURNAL.md entry "0004_inversion/ot-misfit" for context and known caveats before running anything. Verify in this order, cheapest first:

1. Full unit tests (the Mac could only run tests/test_misfit.py standalone — 9 passed):
   uv run pytest packages/physics-informed-flow-map/tests/ -q
   Confirm no regressions in test_tilt / test_flowmap_steer / test_diffusion_dps /
   test_red_diffeq / test_inversion_eval.

2. Smoke inversion, both misfits, same prior/ckpt/target (use the usual esd_teacher
   flow_map checkpoint and small steps so it's minutes not hours):
   uv run python experiments/0004_inversion/run.py prior=flow_map method=flow_tilt \
       ckpt=<esd_teacher ckpt> steps=20 n_samples=2
   uv run python experiments/0004_inversion/run.py prior=flow_map method=flow_tilt \
       ckpt=<same ckpt> steps=20 n_samples=2 method.misfit=ot
   Check: (a) both run to completion; (b) the ot run's name gets the "-ot" suffix and its
   summary contains inv/guidance_misfit_guided < inv/guidance_misfit_unguided (guidance
   actually descends J); (c) the l2 run's metrics are unchanged vs a pre-change run at the
   same settings/seed (it must be byte-identical); (d) J values are O(0.01-1), not O(1e6).

3. Only if 1-2 pass, the first real comparison (single map 6044, 200 steps, n_samples=4,
   matched solves): flow_tilt and mfm_g, each with method.misfit=l2 vs ot. For mfm_g note
   sigma was tuned for L2's huge scale (σ=100); the OT potential is O(1), so σ needs a
   quick sweep (try ~0.03/0.1/0.3) — for flow_tilt normalize_grad=true absorbs the scale
   change so guidance_strength should transfer. Expectations are calibrated in the journal
   entry: under the noiseless inverse crime both misfits can reach near-zero, so the
   interesting readout is trajectory stability and MAE/SSIM at matched budget, not final
   misfit. Log findings to experiments/JOURNAL.md.

One thing worth flagging for step 3: your existing TODO item about d_obs coming from simulate(v_true) rather than stored OpenFWI data is related — if you ever switch to the dataset's own d_obs, that breaks the inverse crime slightly (different solver/acquisition than our guidance operator), which woT comparison more meaningful for free.