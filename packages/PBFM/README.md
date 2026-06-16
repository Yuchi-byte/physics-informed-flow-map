## [ICLR 2026] Official implementation of PBFM - Physics-Based Flow Matching

<div align="center">
    
[![arXiv](https://img.shields.io/badge/arXiv-2506.08604-b31b1b.svg)](https://arxiv.org/abs/2506.08604)
[![Datasets on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-Huggingface-yellow)](https://huggingface.co/datasets/thuerey-group/PBFM)
</div>


***Authors:*** [Giacomo Baldan](https://baldang.github.io), [Qiang Liu](https://qiauil.github.io/), [Alberto Guardone](https://www.aero.polimi.it/en/staff/alberto.guardone), [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/)

---
<div align="center">
<img src="pbfm.webp" alt="PBFM architecture" width="85%"/>
</div>

***Key contributions:***
* _Physics-Based Flow Matching (PBFM)_: Proposes a novel framework for integrating physical constraints into flow matching objectives, leveraging conflict-free gradient updates to minimize PDE and algebraic residuals simultaneously without manual weight tuning.
* _Mitigation of Jensen’s Gap via Unrolling_: Demonstrates that unrolled training trajectories effectively bridge the gap between training objectives and inference-time performance, yielding superior physical consistency without increasing the computational overhead of the final sampler.
* _Analysis of additional Gaussian Noise_: Provides a theoretical and empirical analysis of the role of Gaussian noise in constrained flow matching, demonstrating how the choice of noise floor ($\sigma_{\min}​$) affects the trade-off between distributional accuracy and the precision of physical constraints.
* _Stochastic vs. Deterministic Sampling Analysis_: Provides a formal analysis of the physics-vs-distribution trade-off, establishing the advantages of stochastic sampling and Gaussian noise injection for maintaining distributional fidelity under rigid physical priors.
* _Seamless Integration_: Offers a straightforward implementation strategy that can be integrated into existing flow matching pipelines, consistently improving both distributional accuracy and physical validity across multiple generative tasks.

## Paper

***Abstract:*** Physics-constrained generative modeling aims to produce high-dimensional samples that are both physically consistent and distributionally accurate, a task that remains challenging due to often conflicting optimization objectives. Recent advances in flow matching and diffusion models have enabled efficient generative modeling, but integrating physical constraints often degrades generative fidelity or requires costly inference-time corrections. Our work is the first to recognize the trade-off between distributional and physical accuracy. Based on the insight of inherently conflicting objectives, we introduce <em>Physics-Based Flow Matching</em> (PBFM) a method that enforces physical constraints at training time using conflict-free gradient updates and unrolling to mitigate Jensen's gap. Our approach avoids manual loss balancing and enables simultaneous optimization of generative and physical objectives. As a consequence, physics constraints do not impede inference performance. We benchmark our method across three representative PDE benchmarks. PBFM achieves a Pareto-optimal trade-off, competitive inference speed, and generalizes to a wide range of physics-constrained generative tasks, providing a practical tool for scientific machine learning.

***Cite as:*** 

```latex
@inproceedings{pbfm2026,
    title={Physics vs Distributions: Pareto Optimal Flow Matching with Physics Constraints},
    author={Giacomo Baldan and Qiang Liu and Alberto Guardone and Nils Thuerey},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=tAf1KI3d4X}
}
```


## Installation
Install the required Python packages using pip:
```
pip install torch h5py torchfsm conflictfree einops timm findiff rotary_embedding_torch
```

## Training
Training requires at least one GPU and uses PyTorch's Distributed Data Parallel (DDP). To train the model on a single GPU, run:

```
torchrun --nnodes=1 --nproc_per_node=1 train_ddp.py
```

## Reproducing sampling results
Pretrained model checkpoints for each test case are available in the `logs/PBFM` folder. To generate samples using the pretrained PBFM model, run:

```
python sample.py --version PBFM
```

## Test cases
See the [reference paper](https://arxiv.org/abs/2506.08604) for more details:
- Darcy flow
- Kolmogorov flow
- Dynamic stall

Kolmogorov flow and dynamic stall datasets are available from [Hugging Face](https://huggingface.co/datasets/thuerey-group/PBFM). For the Darcy flow dataset, see [PIDM](https://doi.org/10.3929/ethz-b-000674074).

```
PBFM
├── darcy_flow
│   ├── train
│   │   ├── K_data.csv
│   │   └── p_data.csv
│   └── valid
│       ├── K_data.csv
│       └── p_data.csv
├── dynamic_stall
│   ├── dynamic_stall_test.h5
│   └── dynamic_stall_train.h5
└── kolmogorov_flow
    ├── kolmogorov_test.h5
    └── kolmogorov_train.h5
```
