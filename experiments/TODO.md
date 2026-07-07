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

