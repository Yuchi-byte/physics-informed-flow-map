"""Standard seismic displays for one FlatVel_A shot gather, in space-time and in the spectrum.

Writes two figures. seismic_displays.png shows the record (1000 time steps x 70 receivers) as:
  1. variable-density image (amplitude as colour)
  2. wiggle + variable-area plot (the classic seismic display)
  3. single-trace oscillations (amplitude vs time for a few receivers)

seismic_spectra.png shows the same data in the frequency domain:
  A. amplitude spectrum (|FFT| over time, avg over receivers) — the source wavelet band
  B. the same in dB — the usable bandwidth
  C. f-k spectrum (2D FFT over time x receiver) — apparent velocities / moveout in f-k space

Run from the repo root:
    uv run python FWI_problem_exploration/viz_seismic_displays.py
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


# =============================================================================
# Spectral displays of the same shot gather.
# OpenFWI acquisition: dt = 1 ms (time step), dx = 10 m (receiver spacing).
# =============================================================================
DT = 1e-3   # seconds per time sample
DX = 10.0   # metres between receivers

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# --- A. amplitude spectrum: |FFT over time|, averaged across receivers --------
# One line per source, so you can see all 5 shots share the source wavelet's band.
freqs = np.fft.rfftfreq(nt, d=DT)  # Hz, 0..500 (Nyquist)
ax = axes2[0]
for s in range(seismic.shape[0]):
    spec = np.abs(np.fft.rfft(seismic[s], axis=0)).mean(axis=1)  # avg over receivers
    ax.plot(freqs, spec, linewidth=1.0, label=f"source {s + 1}")
ax.axvline(15.0, color="grey", linestyle="--", linewidth=0.8, label="15 Hz (Ricker peak)")
ax.set_xlim(0, 80)
ax.set_title("A. Amplitude spectrum\n(|FFT| over time, avg over receivers)", fontsize=11)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.legend(fontsize=7)

# --- B. same, in decibels — reveals the usable bandwidth ----------------------
ax = axes2[1]
spec_c = np.abs(np.fft.rfft(gather, axis=0)).mean(axis=1)
spec_db = 20 * np.log10(spec_c / spec_c.max() + 1e-12)
ax.plot(freqs, spec_db, color="C3", linewidth=1.2)
ax.axhline(-40, color="grey", linestyle=":", linewidth=0.8, label="-40 dB")
ax.set_xlim(0, 120)
ax.set_ylim(-80, 5)
ax.set_title(f"B. Amplitude spectrum in dB\n(source {SOURCE_IDX + 1}, normalised)", fontsize=11)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB re max)")
ax.legend(fontsize=8)

# --- C. f-k spectrum: 2D |FFT| over (time, receiver) --------------------------
# The classic frequency-wavenumber panel: energy dipping in f-k picks out apparent
# velocities / moveout directions. Full 2D FFT, shifted so zero is centred.
fk = np.fft.fftshift(np.abs(np.fft.fft2(gather)))
f_axis = np.fft.fftshift(np.fft.fftfreq(nt, d=DT))       # Hz
k_axis = np.fft.fftshift(np.fft.fftfreq(n_rec, d=DX))    # 1/m
ax = axes2[2]
im = ax.imshow(
    20 * np.log10(fk / fk.max() + 1e-12),
    aspect="auto", cmap="magma", vmin=-60, vmax=0, origin="lower",
    extent=[k_axis[0], k_axis[-1], f_axis[0], f_axis[-1]],
)
ax.set_ylim(0, 80)  # positive frequencies, source band
ax.set_title("C. f-k spectrum\n(2D FFT over time x receiver)", fontsize=11)
ax.set_xlabel("Wavenumber (1/m)")
ax.set_ylabel("Frequency (Hz)")
fig2.colorbar(im, ax=ax, fraction=0.046, label="dB re max")

fig2.suptitle(
    f"FlatVel_A · spectral view of the observed seismic (dt = 1 ms, dx = 10 m, 15 Hz Ricker source)",
    fontsize=12, y=1.02,
)
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "seismic_spectra.png")
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved → {out2}")
