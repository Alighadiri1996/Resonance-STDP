"""
Supplementary Fig. S3A: percentage of potentiated FF and FB connections versus the
inhibitory time constant of layer 2.

Motivation
----------
Quantifies how many FF and FB synapses survive learning when the resonance frequency
of layer 2 is detuned.

What it measures
----------------
For sim1.npz ... sim7.npz (tau_in,2 = 6..12 ms), the percentage of FF and FB synapses
whose final weight is >= 0.15 nS, relative to all initial synapses (4900 per direction).

Usage
-----
    python plot_S3A_stabilized_connections.py
    python plot_S3A_stabilized_connections.py --dir results
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--dir', default='.', help='folder with sim1.npz ... sim7.npz')
parser.add_argument('--out', default='stabilization_percentage_vs_tau.png', help='output figure')
args = parser.parse_args()

# -----------------------------
# Global Plot Styling (rcParam)
# -----------------------------
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.figsize': [8, 5],
    'savefig.dpi': 300,
    'lines.linewidth': 2,
    'font.family': 'sans-serif'
})

# -----------------------------
# Configuration
# -----------------------------
sim_files = [f"sim{i}.npz" for i in range(1, 8)]
tau_values = [6, 7, 8, 9, 10, 11, 12]
threshold = 0.15

ff_percentages = []
fb_percentages = []

for filename in sim_files:
    data = np.load(os.path.join(args.dir, filename))

    w_FF_initial = data['w_FF_initial']
    w_FF_final = data['w_FF_final']
    w_FB_initial = data['w_FB_initial']
    w_FB_final = data['w_FB_final']

    # (Number of stabilized / Total initial connections) * 100
    ff_pct = (np.sum(w_FF_final >= threshold) / len(w_FF_initial)) * 100
    fb_pct = (np.sum(w_FB_final >= threshold) / len(w_FB_initial)) * 100

    ff_percentages.append(ff_pct)
    fb_percentages.append(fb_pct)

    print(f"tau={tau_values[len(ff_percentages)-1]}: FF={ff_pct:.1f}%, FB={fb_pct:.1f}%")

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(tau_values, ff_percentages, 'o-', color='blue', label='Feedforward (FF)', linewidth=2)
plt.plot(tau_values, fb_percentages, 'o-', color='red', label='Feedback (FB)', linewidth=2)

plt.xlabel(r"$\tau_{inh,2}$ (ms)")
plt.ylabel("Stabilized Connections (%)")
plt.ylim(-5, 105)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(args.out, dpi=300, bbox_inches='tight')
print(f"Plot saved as {args.out}")
