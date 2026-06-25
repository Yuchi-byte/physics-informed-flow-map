# Journal

One verdict line per concluded finding. Newest last. Each line cites the
framework and run, the verdict, and the headline number.

Format: `NNNN_slug/variant — verdict: headline (evidence)`

<!-- e.g. 0001_mnist_pipeline/default — pass: final loss 0.21 < 0.30 gate (200 steps, runs/0001_mnist_pipeline/2026-06-24T...Z) -->

- 0001_flow_matching/gaussians — pass: energy distance 0.009576 < 0.5 gate; 8 modes recovered (runs/0001_flow_matching/2026-06-24T204300Z)
- 0001_flow_matching/mnist — pass: final FM loss 158.02 < 240 gate; samples digit-like (runs/0001_flow_matching/2026-06-24T205632Z)
- 0001_flow_matching/gaussians — pass: energy distance 0.0104 < 0.5 gate (~2× the 0.0050 noise floor); epoch loop, 100 epochs / 200 sampler steps (runs/0001_flow_matching/2026-06-25T00-45-58Z)
- 0001_flow_matching/mnist — pass: final FM loss ~107 < 240 gate; clean digits across all 10 classes; DiT 256/6/8, 100 epochs, 200 sampler steps (runs/0001_flow_matching/2026-06-25T00-53-06Z)
