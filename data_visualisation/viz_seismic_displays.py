"""The three standard seismic displays for one FlatVel_A shot gather.

Takes a single source's record (1000 time steps x 70 receivers) and shows it as:
  1. variable-density image (amplitude as colour)
  2. wiggle + variable-area plot (the classic seismic display)
  3. single-trace oscillations (amplitude vs time for a few receivers)

Run from the repo root:
    uv run python data_visualisation/viz_seismic_displays.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data/openfwi/FlatVel_A"
SAMPLE_IDX = 0    # sample inside data1.npy
SOURCE_IDX = 2    # which of the 5 shots (0..4); 2 = centre shot
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

seismic = np.load(f"{DATA_DIR}/data/data1.npy")[SAMPLE_IDX]  # (5, 1000, 70)
gather = seismic[SOURCE_IDX]                                 # (1000, 70): time x receiver
nt, n_rec = gather.shape
t = np.arange(nt)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
vabs = np.percentile(np.abs(gather), 99)

# --- 1. variable-density image -------------------------------------------------
ax = axes[0]
ax.imshow(gather, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs,
          extent=[0, n_rec, nt, 0])
ax.set_title("1. Variable-density image\n(amplitude = colour)", fontsize=11)
ax.set_xlabel("Receiver")
ax.set_ylabel("Time step")

# --- 2. wiggle + variable-area -------------------------------------------------
# Each receiver trace is drawn as a vertical wiggle: x = receiver + scaled amplitude.
# Positive lobes are filled black — the classic seismic "variable-area wiggle".
ax = axes[1]
scale = 2.5 / vabs           # amplitude -> receiver-spacing units
step = 2                     # draw every 2nd trace so it isn't too dense
for r in range(0, n_rec, step):
    tr = gather[:, r] * scale
    x = r + tr
    ax.plot(x, t, color="k", linewidth=0.4)
    ax.fill_betweenx(t, r, x, where=(x > r), color="k", linewidth=0)
ax.set_ylim(nt, 0)
ax.set_xlim(-2, n_rec + 2)
ax.set_title("2. Wiggle + variable-area\n(classic seismic display)", fontsize=11)
ax.set_xlabel("Receiver")
ax.set_ylabel("Time step")

# --- 3. single-trace oscillations ---------------------------------------------
ax = axes[2]
picks = [10, 35, 60]         # near, middle, far receivers
offsets = np.linspace(0, 2 * (len(picks) - 1), len(picks))
for off, r in zip(offsets, picks):
    ax.plot(t, gather[:, r] / vabs + off, linewidth=0.8, label=f"receiver {r}")
    ax.axhline(off, color="grey", linewidth=0.4, alpha=0.5)
ax.set_yticks(offsets)
ax.set_yticklabels([f"rec {r}" for r in picks])
ax.set_title("3. Single-trace oscillations\n(amplitude vs time)", fontsize=11)
ax.set_xlabel("Time step")
ax.set_ylabel("Receiver (offset for clarity)")

fig.suptitle(
    f"FlatVel_A · source {SOURCE_IDX + 1} of 5 · one shot gather (1000 time steps x 70 receivers)",
    fontsize=12, y=1.02,
)
fig.tight_layout()
out = os.path.join(OUT_DIR, "seismic_displays.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved → {out}")
