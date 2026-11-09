"""
Supplementary Fig. S3B: weight-averaged FF and FB delays and their sum versus the
inhibitory time constant of layer 2.

Motivation
----------
Tests whether the total FF + FB loop delay selected by STDP follows the resonance
period of the detuned bi-layer network.

What it measures
----------------
For sim1.npz ... sim7.npz (tau_in,2 = 6..12 ms), the average axonal delay of FF and of
FB synapses weighted by their final weight, sum(w * d) / sum(w), and the sum FF + FB.
As in the published panel, the average runs over all synapses (weight threshold 0 nS):
depressed synapses end near 0 nS and contribute little, but synapses with intermediate
final weights are included. Set --threshold 0.15 to restrict the average to the
synapses counted in Fig. S3A.

Usage
-----
    python plot_S3B_weighted_delays.py
    python plot_S3B_weighted_delays.py --threshold 0.15
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--dir', default='.', help='folder with sim1.npz ... sim7.npz')
parser.add_argument('--threshold', type=float, default=0.0,
                    help='minimal final weight (nS) included in the average; default 0 (published)')
parser.add_argument('--out', default='weighted_delay_vs_tau.png', help='output figure')
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
weight_threshold = args.threshold


def get_weighted_avg(weights, delays, thresh):
    """Average delay weighted by final weight, over synapses with weight >= thresh."""
    mask = weights >= thresh
    if np.sum(weights[mask]) > 0:
        return np.sum(weights[mask] * delays[mask]) / np.sum(weights[mask])
    return np.nan  # no synapse above threshold


ff_weighted_delays = []
fb_weighted_delays = []

for filename in sim_files:
    data = np.load(os.path.join(args.dir, filename))

    # Delays do not change; final weights shift the weighted average
    w_FF = data['w_FF_final']
    d_FF = data['D_FF_initial']
    w_FB = data['w_FB_final']
    d_FB = data['D_FB_initial']

    ff_avg = get_weighted_avg(w_FF, d_FF, weight_threshold)
    fb_avg = get_weighted_avg(w_FB, d_FB, weight_threshold)

    ff_weighted_delays.append(ff_avg)
    fb_weighted_delays.append(fb_avg)

    print(f"tau={tau_values[len(ff_weighted_delays)-1]}: FF_avg={ff_avg:.2f}ms, FB_avg={fb_avg:.2f}ms")

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(tau_values, ff_weighted_delays, 'o-', color='blue', label='Average FF delays', linewidth=2)
plt.plot(tau_values, fb_weighted_delays, 'o-', color='red', label='Average FB delays', linewidth=2)

# Total loop delay
total_loop = np.array(ff_weighted_delays) + np.array(fb_weighted_delays)
plt.plot(tau_values, total_loop, '--', color='black', label='Total (FF+FB)', alpha=0.7)

plt.xlabel(r"$\tau_{inh,2}$ (ms)")
plt.ylabel("Delay (ms)")
plt.legend(loc='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(args.out, dpi=300, bbox_inches='tight')
print(f"Plot saved as {args.out}")
