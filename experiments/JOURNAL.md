# Journal

One line per concluded finding. Newest last. Each line cites the framework
and run, and the headline metric.

Format: `NNNN_slug/variant — headline metric (evidence)`

<!-- e.g. 0001_mnist_pipeline/default — final loss 0.21 (200 steps, runs/0001_mnist_pipeline/2026-06-24T...Z) -->
## model tuning 
### mfm 
- 0004_inversion/mfm_g mc_samples 4→20 (single held-out map 6044, esd_teacher flow-map ckpt, σ1000/renorm=F/SDE, 200 steps) — metrics IDENTICAL to mc=4 (MAE 0.2160 vs 0.2157, SSIM 0.356 vs 0.356, misfit_ratio 0.735 vs 0.724) at 5× the solves (16k vs 3.2k). mc=64 is infeasible on 32 GB (mc=20 already peaks 24.7 GiB — the IWAE grad holds all n_samples×mc wave-solve graphs at once). CONFIRMS the n=32 conclusion from the MC side: at σ1000 the steering signal is so gentle that MC-estimate quality is irrelevant — draw count is NOT the lever. mfm_g still far behind the simple Tweedie flow_tilt on the same target+ckpt (flow_tilt MAE 0.127/SSIM 0.635/misfit_ratio 0.072 @ 800 solves vs mfm_g 0.216/0.356/0.73 @ 16k). (runs/0004_inversion/flow_map_mfm_g_2026-07-02T22-03-01Z)
- 0004_inversion/mfm_g sigma/renorm sweep on the same single map 6044 (mc=4, 200 steps) — σ100/renorm=F is the best mfm_g config: MAE 0.195/SSIM 0.431/misfit_ratio 0.077 (vs σ1000's 0.216/0.356/0.73; σ300 in between at 0.193/0.373/0.14; renorm=T at σ1000 explodes MAE to 0.459 exactly as the earlier eval-scale tuning found). KEY MECHANISM FINDING: at σ100 mfm_g fits the data as hard as flow_tilt (misfit_ratio 0.077 vs 0.072) yet still loses badly on model error (MAE 0.195 vs 0.127, SSIM 0.431 vs 0.635) — the gap is NOT steering strength but the DIRECTION of the likelihood gradient: the Tweedie chain rule (through the network Jacobian at x_t) stays manifold-aligned, while the IWAE gradient through fresh-noise one-step posterior draws finds off-manifold ways to cut the misfit. PAPER CHECK (arXiv 2601.14430 F.3.2/F.3.6): the paper's ImageNet practice is ODE (250 Euler steps) + steering-drift RESCALE to ||base_drift|| with λ=1 (Eq. 136) — i.e. renorm=true IS the paper's practical setting (the mfm_g.yaml comment claimed otherwise); their GMM inverse problem uses the TRUE observation noise as sigma, reinforcing that our noiseless-inverse-crime sigma is unprincipled (runs/0004_inversion/flow_map_mfm_g_2026-07-02T{22-47-38,22-52-34,22-57-39}Z)
 sigma gives the steering strength?????
- PICKUP 2026-07-02 STATE: mfm_g best config now σ100/renorm=F/mc=4 (yaml default updated); at that setting it matches flow_tilt's data fit (misfit_ratio 0.077 vs 0.072) but loses on model error (MAE 0.195 vs 0.127, SSIM 0.431 vs 0.635) on single map 6044, esd_teacher ckpt. Draw count (mc 4→20) and estimator family (gf per paper) are ELIMINATED as levers; renorm=T eliminated (over-fits). EDUCATED GUESSES, ranked: The IWAE gradient is manifold-unaligned for hard data-consistency rewards: Tweedie's chain rule through the network Jacobian acts as a manifold projector; the posterior-draw gradient has no such projection, so equal misfit ≠ equal model error under noiseless non-uniqueness. 
- 0004_inversion/fmrg_e first run — guidance sweep {0.3,1.0,3.0} + normalize_grad={T,F} on single map 6044, esd_teacher ckpt, 200 steps, n_samples=4 (800 solves). BEST: normalize=T, g=0.3: MAE 0.137/SSIM 0.578/misfit_ratio 0.013. FMRG-E does NOT beat flow_tilt at matched solve budget (n_opt=1). SHARP CLIFF: g=0.3 → good (MAE 0.137), g=1.0 → broken (MAE 0.676), g=3.0 → worst (MAE 0.899). KEY MECHANISM: FMRG-E at g=0.3 drives misfit_ratio to 0.013 (5× lower than flow_tilt's 0.072) yet has WORSE model error (MAE 0.137 vs 0.127) — fits data harder but goes more off-manifold. Two reasons: (1) wt=(1-t)*t_next decays to zero at t≈0, suppressing guidance during the early trajectory steps where rough velocity structure forms (flow_tilt applies constant guidance from step 1); (2) inner gradient is in x1-space and NOT manifold-aligned — FMRG-E's n_opt=1 single step in x1-space differs from flow_tilt's chain-rule gradient in xt-space which implicitly projects onto the trajectory tangent. BOTTOM LINE: FMRG-E (n_opt=1) is a DPS variant with FMRG time-weighting, but the wt schedule that's optimal by control theory is suboptimal for the FWI recovery objective under the inverse crime. (runs/0004_inversion/flow_map_fmrg_e_2026-07-03T00-08-01Z/{0,1,2}; 5n5nfb0z)



## Inversion results
- 0004_inversion/L2-vs-OT across ALL prior families (map 6044, 200 steps, n=4, 800 solves each, defaults per method yaml; wandb qdhdzmo3/0sqtvodl/g3mxl5p5/z7a57sxz + the flow_map pair below) — the OT edge GENERALIZES. PATTERN: in all four pairs OT fits the *L2 eval misfit* less or comparably tightly yet recovers a better model — consistent with the amplitude-weighted potential acting as a residual preconditioner that redirects (not strengthens) the guidance gradient; the weaker/more fragile the method's own gradient direction (dps, mfm_g), the bigger the OT gain. ┌───────────────────────────┬─────────┬───────────────┬───────────────┬─────────────────┬────────────────┐
│      prior · method       │ misfit  │      MAE      │     SSIM      │ L2 misfit ratio │     wandb      │
├───────────────────────────┼─────────┼───────────────┼───────────────┼─────────────────┼────────────────┤
│ flow_matching · flow_tilt │ l2      │ 0.128         │ 0.634         │ 0.097           │ qdhdzmo3       │
├───────────────────────────┼─────────┼───────────────┼───────────────┼─────────────────┼────────────────┤
│ flow_matching · flow_tilt │ ot      │ 0.107         │ 0.714         │ 0.040           │ 0sqtvodl       │
├───────────────────────────┼─────────┼───────────────┼───────────────┼─────────────────┼────────────────┤
│ flow_map · flow_tilt      │ l2 / ot │ 0.126 / 0.109 │ 0.637 / 0.682 │                 │ (this morning) │
├───────────────────────────┼─────────┼───────────────┼───────────────┼─────────────────┼────────────────┤
│ flow_map · mfm_g σ0.03    │ l2 / ot │ 0.195 / 0.128 │ 0.431 / 0.518 │                 │ sv73iql9       │
├───────────────────────────┼─────────┼───────────────┼───────────────┼─────────────────┼────────────────┤
│ diffusion · dps           │ l2      │ 0.172         │ 0.491         │ 0.020           │ g3mxl5p5       │
├───────────────────────────┼─────────┼───────────────┼───────────────┼─────────────────┼────────────────┤
│ diffusion · dps           │ ot      │ 0.124         │ 0.629         │ 0.006           │ z7a57sxz       │
└───────────────────────────┴─────────┴───────────────┴───────────────┴─────────────────┴────────────────┘


- 0003_diffusion/openfwi_full DEFINITIVE (2026-07-08) — the full-OpenFWI diffusion prior. Run runs/0003_diffusion/openfwi_2026-07-07T23-26-16Z, wandb 984t0883. Epoch-59 grid shows every family archetype (flat/curved layers, fault offsets, both Style textures), crisp, no collapse. 
┌──────────────┬────────┬──────────┬───────────┐
│    family    │ gen ED │ floor ED │ gen−floor │
├──────────────┼────────┼──────────┼───────────┤
│ CurveFault_A │ 1.687  │ 1.734    │ −0.047    │
├──────────────┼────────┼──────────┼───────────┤
│ CurveFault_B │ 2.189  │ 2.114    │ +0.075    │
├──────────────┼────────┼──────────┼───────────┤
│ CurveVel_A   │ 1.570  │ 1.329    │ +0.241    │
├──────────────┼────────┼──────────┼───────────┤
│ CurveVel_B   │ 13.290 │ 13.570   │ −0.280    │
├──────────────┼────────┼──────────┼───────────┤
│ FlatFault_A  │ 2.094  │ 2.235    │ −0.141    │
├──────────────┼────────┼──────────┼───────────┤
│ FlatFault_B  │ 2.721  │ 2.741    │ −0.020    │
├──────────────┼────────┼──────────┼───────────┤
│ FlatVel_A    │ 2.167  │ 1.898    │ +0.269    │
├──────────────┼────────┼──────────┼───────────┤
│ FlatVel_B    │ 13.349 │ 13.621   │ −0.272    │
├──────────────┼────────┼──────────┼───────────┤
│ Style_A      │ 3.643  │ 3.330    │ +0.313    │
├──────────────┼────────┼──────────┼───────────┤
│ Style_B      │ 3.843  │ 3.870    │ −0.027    │
└──────────────┴────────┴──────────┴───────────┘


- 0004_inversion. Trained on full openFWI dataset. 
┌───────────────────────────┬────────┬────────┐
│          config           │  MAE   │  SSIM  │
├───────────────────────────┼────────┼────────┤
│ fm · flow_tilt · ot       │ 0.2518 │ 0.4053 │
├───────────────────────────┼────────┼────────┤
│ flow_map · flow_tilt · ot │ 0.2546 │ 0.4004 │
├───────────────────────────┼────────┼────────┤
│ fm · flow_tilt · l2       │ 0.2844 │ 0.3380 │
├───────────────────────────┼────────┼────────┤
│ flow_map · flow_tilt · l2 │ 0.2853 │ 0.3359 │
├───────────────────────────┼────────┼────────┤
│ diffusion · dps · l2      │ 0.2864 │ 0.3494 │
├───────────────────────────┼────────┼────────┤
│ diffusion · dps · ot      │ 0.2960 │ 0.3839 │
└───────────────────────────┴────────┴────────┘

# Prior Training 
## On FlatVel_A dataset  
* flow matching x 1
* flow map x 1 
* Diffusion x 1

## On full OpenFWI 
* flow matching x 1 
* flow map x 1 
* Diffusion: DIT. 100 epochs. 

## Analysis 

# Inversion on FlatVel_A prior 


# Inversion on full openFWI prior




# Hypotheses to test
## Disease diagnosis: Does the current methods suffer from non-uniqueness or cycle-skipping? 
* Non-uniqueness problem can be solved by the priors, and cycle-skipping are solved by a misfit better than l2. And current diffusion+L2 steering suffers more from cycle-skipping than than non-uniqueness. 
    * Method: Use trained priors and inversion methods (diffusion+Tweedie+L2) to visualise 20 velocity maps that give the same d_obs. 
        * Seed benchmark datasets: 1 per family (10 in total). For each data, put their seismic data through the diffusion+Tweedie+L2 framwork 20 times (equivalent to starting from 20 differnet initial models). 
        * Record metrics along the inversion trajectory: visualise the noisy image and its tweedie prediction, 
        * result: examine the seismic data of each final inversion and determine whether each image is cycle-skipped or fell into the global minimum.  Are they drastically different to each other? Are they broadly similar and only differ on the finer gradient? Or even large-scale features are mis-matching? If the inversion failed, what is the reason? 
        * Caveat: whether images are non-unique or cycle-skipped is difficult to tell. If at all possible, need to look at the d_obs (probably its evolution through denoising time) in order to tell. 
    * cycle-skipped images are hard to construct. The solution of the PDS is not a linear superposition of the solutions for each trace due to the spatial gradient term. So can't easily time-shift the whole image, but different traces are shifted in different directions. 
        * cycle-skipping isn't caused by observational error. It's a fault of the l2 misfit design, which caused a lot of local minimum. 
* flow-map is better at solving the non-uniqueness problem than a diffuision prior. 



## Priors 
* Different priors have differing generative/regularisation powers on more complex velocity maps. Eg. mfm might outperform diffusion for the Style Family but underperforming for FlatVelA. 
    * methodology: for differnt families, test how mfm's inversion and training metric compares with diffusion and flow matching. 
    * why it matters: field data and velocity map are complex due to small scale features like oil reservoir, pipes, intrusions, etc (need to fact check). So the ability to handle/generate such complex images should be prioritised. 
    * Methodology: Seed test dataset: 5-10 from each family. Visualise trajectory to see whether the process arrives at the wrong image too early or too late -- this can inform the underlying mechanism for incorrect inversion. 


## Cycle skipping optimisation
* In cycle skipping, the traces are shifted independently.
    * To test it, look at the d_obs of cycle skipped velocity maps. Plot the difference between that and d_obs. 
    * why? This gives us information on how to construct cycle-skipped d_obs to invert. 

* Training an auxiliary neural network to learn how different the velocity maps corresponding to two d_obs is a better alternative than using the naive L2 or OT misfit. 
    * Method: 
    * implementation details: need to think about how to train. 
    * why? Adhering to PDE means that F(velocity) should be as close to d_obs as possible. But at early stages of the transport, they won't be aligned at all. We need a misfit function to tell us which direction in the velocity map space gives the steepest descent in order to steer the trajectory. But L2 has too many local minima, and OT (WHAT'S WRONG WITH OT????)??because of cycle skipping and non-uniqueness, learning 

* Spectral information are susseptable to cycle skipping. Cycle-skipped d_obs will have similar spectral graphs because the frequency and wavelength aren't affected by time-shift. Spectral graphs are still susceptible to non-uniqueness though because d_obs and its FFT is a one-to-one mapping, so every spectral graph is also mapped to many velocity maps.  
    * method: Find 10 velocity maps that give cycle-skipped d_obs. And then compare the FFT of those cycle-skipped d_obs, and look at whether the spectral images have a high variability. 




## Inversion engineering optimisation
* d_obs data is attenuated quite a lot (the amplitude of a trace decreases with time -- is this built into the equation, or it's a field phenomena?). Turning it to log scale for the inversion would make the inversion more accurate. But this also requires the log to be taken for the whole equation. 
## Extension: new framwork and problem
* elastic FWI as opposed to acoustic FWI gives better estimates of the parameter models. Even just the velocity parameter. And it's more useful to the field. 
    * method: literature review. Ask field geologists whether elastic FWI is an untrodden frontier. 
    * why? Appling flow map to elastic FWI may have more practical value than acoustic FWI. 

## Experiments 2026-07-18

check mfm renorm=true works better (the change is already made to the configs). 
run on curvefault_b_17, and Use L2 and OT, use mfm prior, drfit estimate = dps (and then try iwae). visualise the monte-carlo samples for iwae. Overall use n_samples =10 adn visualise their trajectory. All the codes are already in place in 0004 inversion, just make sure you turn on the correct switches. Also vary steps: 10, 200, 400. 
turn off wandb. Everything should produce a folder in runs with the visualisation as we normally do. Make sure the evolution of d_obs (in log scale) is visualised too (code already exists). 
Then in this chat, give me a summary statistics of all the runs so that i can compare. 