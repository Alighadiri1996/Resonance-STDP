"""
Supplementary Fig. S1B: resonance frequency of a layer network versus its inhibitory
synaptic time constant (mean +- SD over trials).

Motivation
----------
Calibrates how the inhibitory time constant of layer 2 detunes its resonance frequency
in Supplementary Figs. S2 and S3.

What it measures
----------------
Reads complete_resonance_metadata.npz written by S1B_resonance_vs_tau_in.py and plots
the mean and SD of the per-trial resonance frequencies for tau_in = 6 to 12 ms.

Usage
-----
    python S1B_plot.py                                           # published data (101 trials)
    python S1B_plot.py --data complete_resonance_metadata.npz    # after re-running the sweep

The .npz stores the per-trial frequencies as an object array, hence allow_pickle=True.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--data', default='data/complete_resonance_metadata.npz', help='input .npz')
parser.add_argument('--out', default='FigS1B_resonance_vs_tau_in.png', help='output figure')
args = parser.parse_args()

# -----------------------------
# Load Data
# -----------------------------
data = np.load(args.data, allow_pickle=True)
tau_range = data['tau_range']
raw_freqs = data['raw_frequencies']
means = data['means']
stds = data['stds']

# -----------------------------
# Global Styling for Publication
# -----------------------------
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.figsize': [11, 7],
    'font.family': 'sans-serif'
})

fig, ax = plt.subplots()

# Mean +- SD over trials
ax.errorbar(tau_range, means, yerr=stds, fmt='o-',
            color='#003366', ecolor='#CC0000', elinewidth=2,
            capsize=8, markersize=10, linewidth=3, label="Mean ± SD")

ax.set_xlabel(r"$\tau_{syn,in}$ (ms)")
ax.set_ylabel("Resonance Frequency (Hz)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='white', framealpha=1)
ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig(args.out, dpi=300, bbox_inches='tight')

for t, r, m, s in zip(tau_range, raw_freqs, means, stds):
    print(f"tau_in = {t:4.1f} ms: {m:6.2f} +- {s:5.2f} Hz (n = {len(r)})")
print(f"Saved {args.out}")
