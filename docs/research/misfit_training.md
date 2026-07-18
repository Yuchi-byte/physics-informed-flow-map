# Misfit function training 


## Motivation 
The L2 and OT misfits guides the FWI gradient descent in the model space. This steering is combined with the drift from the generative prior -- both of them together inverts the seismic data. 

The L2 and OT misfits are not perfect and don't produce the best inverted velocity map as can be seen from the inversion results. 

The choice of the misfit function is crucial to the inversion process. From previous inversion experiemnts, using a different misfit funciton results in a different inverted velocity map, and most of the time OT generates better inversion than L2. This is the case for both the flow-matching prior and the diffusion prior. 

## Goal
To learn a general/blackbox misfit function that boosts the performance of the overall inversion. 

## Key requirement 
The misfit function, denoted by J, takes in two d_obs (seismic detector amplitudes) -- J(d1, d2). J is a scalar that tells us how much the corresponding velocity maps, v1 and v2, differ. d1 and d2 can be any arbitrary seismic traces as long as that (d1, v1) and (d2, v2) are solutions to the same acoustic wave equation (i.e. share the same source). And that J(d1, d1) = 0 and J(d1, d2) = J(d2, d1). 

Practically, one of the input seismic traces, say d1, should be the observed d that the inversion targets. This means that v1 is the true velocity map that is unknown. v2 would be one of the velocity maps along the FWI/inversion trajectory. The gradient of the misfit function at v2 informs how to get to the next velocity state.


Such learned misfit function will hopefully smooth out the misfit landscape that reduces the number of local minima and the gradient of the misfit will be more accurate in pointing to the global minimum. The neural network learns how different v1 and v2 is from the form of d1 and d2, as opposed to the simple L2 misfit where the granularities are lost. 

One assumption implied by J(d1, d2=d1)=0 is that we are not concerned with the non-uniqueness issue at all. The ill-posedness of the inverse problem means that there can be two very different velocity models that have the same d, and in principle the misfit between them should be bigger than zero. So I'm assuming that as long as d1 and d2 are the same, v1 and v2 are also the same. This assumption is backed by the experimental observation: when the velocity map is far from the true model, its seismic trace is also very off. The velocity maps with a more accurate seismic trace also tend to be closer to the true map.  


## Training Design 
Get pairs of (d1, d2) and the corresponding (v1, v2). Pass (d1, d2) into the neural network that parameterises J, and then the output of J should learn to be the L2 misfit between (v1, v2). 

## Wiring into the current architecture 

The trained misfit function should be wired into 0004 inversion as an alternative to the current L2 and OT misfit function.
## Resolved design decisions
The buildable version of this note lives in
`docs/superpowers/specs/2026-07-18-learned-misfit-design.md`. Key choices made there:

- **Form (Siamese Euclidean metric).** `J(d1,d2) = ‖φ(d1) − φ(d2)‖²` with a shared encoder φ.
  This is *why* `J(d,d)=0`, symmetry, `J≥0` and the triangle inequality hold — they are
  structural, not trained. It also matches the goal of "smoothing the misfit landscape": a
  squared Euclidean distance in the latent space is exactly a well-behaved, single-basin metric.
- **Target = MSE of the normalized `[-1,1]` velocity maps.** MSE is itself a squared-Euclidean
  distance, so it is the geometrically matched label for the Siamese form (report RMSE = √J).
  SSIM was considered but is not a metric (no triangle inequality) and only fits an expressive
  non-metric head, so it was dropped for the MVP.
- **Training data = an offline pair bank** of `(v, d=simulate(v))` covering not just real maps
  but the smooth/blurry maps an inversion actually visits (Gaussian blur, convex blends of real
  maps). The forward solves are cached once so training samples pairs cheaply.
- **Inference:** `φ(d_obs)` is frozen once, so `J` drops into `0004`'s existing `MisfitFn`
  interface as a third option alongside `l2` and `ot`.

## Literature review 
https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.1011825/full

## Criticism
- **Train/inference distribution gap (the main risk).** `J`'s gradient is only trustworthy on
  data it was trained near. Early in an inversion the state is smooth and far from any real map;
  if the bank does not cover that regime, guidance is out-of-distribution exactly where escaping
  cycle-skipping matters most. The augmentations are an approximation of the real trajectory-state
  distribution; the honest fallback is to harvest actual intermediate states from `0004` runs.
- **The wave adjoint is still in the loop.** The steering gradient is
  `∇_v J = (∂F/∂v)ᵀ ∇_d2 J`, so `J` only changes the *data-space* direction `∇_d2 J`, not the
  adjoint. The bet is that this direction — which approximates `∇_d2 ‖v_true − v(d2)‖²` — is a
  better-conditioned target than L2's `2(F(v) − d_obs)`. That is plausible but unproven; it is
  what the CurveFault_B experiment tests.
- **Non-uniqueness is assumed away.** `J(d,d)=0 ⇒ v1=v2` denies genuine FWI ambiguity. Accepted
  on the empirical observation that closer data ⇒ closer models, but it caps how "correct" `J`
  can be where the forward map is many-to-one.
- **Euclidean-embedding capacity.** If the shared-encoder squared distance cannot fit the MSE
  target well, the geometry is wrong and a more expressive (non-metric) head is needed — trading
  the landscape guarantee for capacity.
- **Single family / single geometry.** Trained and tested on CurveFault_B with a fixed
  acquisition; nothing yet shows it transfers.