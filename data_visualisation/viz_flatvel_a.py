"""Visualise one FlatVel_A sample: all 5 seismic source gathers + velocity map.

Run from the repo root:
    uv run python data_visualisation/viz_flatvel_a.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = "data/openfwi/FlatVel_A"
SAMPLE_IDX = 0  # which sample inside data1.npy
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

seismic = np.load(f"{DATA_DIR}/data/data1.npy")[SAMPLE_IDX]   # (5, 1000, 70)
velocity = np.load(f"{DATA_DIR}/model/model1.npy")[SAMPLE_IDX, 0]  # (70, 70)

n_sources = seismic.shape[0]  # 5

# clip seismic for display (same symmetric limit for all panels)
vabs = np.percentile(np.abs(seismic), 99)

fig = plt.figure(figsize=(18, 7))
# 5 seismic panels + 1 velocity panel, with a wider velocity panel
gs = gridspec.GridSpec(1, n_sources + 1, figure=fig,
                       width_ratios=[1] * n_sources + [1.3],
                       wspace=0.08)

for s in range(n_sources):
    ax = fig.add_subplot(gs[0, s])
    gather = seismic[s]  # (1000, 70)
    im = ax.imshow(gather, aspect="auto", cmap="RdBu_r",
                   vmin=-vabs, vmax=vabs,
                   extent=[0, 70, 1000, 0])
    ax.set_title(f"Source {s+1}", fontsize=11)
    ax.set_xlabel("Receiver", fontsize=9)
    if s == 0:
        ax.set_ylabel("Time step", fontsize=9)
    else:
        ax.set_yticklabels([])

# shared colorbar for seismic
cax_s = fig.add_axes([0.02, 0.12, 0.008, 0.74])
sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                            norm=plt.Normalize(vmin=-vabs, vmax=vabs))
sm.set_array([])
cb = fig.colorbar(sm, cax=cax_s)
cb.set_label("Amplitude", fontsize=9)

# velocity map
ax_v = fig.add_subplot(gs[0, n_sources])
im_v = ax_v.imshow(velocity, aspect="auto", cmap="jet",
                   vmin=velocity.min(), vmax=velocity.max(),
                   extent=[0, 70, 70, 0])
ax_v.set_title("Velocity map", fontsize=11)
ax_v.set_xlabel("X (grid cells)", fontsize=9)
ax_v.set_ylabel("Depth (grid cells)", fontsize=9)
cb_v = fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
cb_v.set_label("Velocity (m/s)", fontsize=9)

fig.suptitle(
    f"FlatVel_A  — file data1/model1, sample index {SAMPLE_IDX}\n"
    f"Seismic: {seismic.shape} (sources × time × receivers)   "
    f"Velocity: {velocity.shape}",
    fontsize=12, y=1.01
)

out = os.path.join(OUT_DIR, "flatvel_a_viz.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved → {out}")
