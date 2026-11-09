"""
Supplementary Fig. S5B: weight-averaged delays of potentiated FF and FB connections and
their sum versus the initial inter-layer connection probability.

Motivation
----------
Tests whether the total FF + FB loop delay selected by STDP stays matched to the
resonance period when the inter-layer connectivity is sparser.

What it measures
----------------
For sim8.npz ... sim11.npz (p = 0.25, 0.5, 0.75, 1), the average axonal delay weighted
by final weight, sum(w * d) / sum(w), over synapses with final weight >= threshold, with
thresholds 0.15 nS * (4, 2, 1.33, 1) = 0.6, 0.3, 0.2, 0.15 nS. The script prints how many
synapses pass each threshold: in the published runs the FB averages for p = 0.25, 0.5
and 0.75 rest on 30, 15 and 12 synapses (725 at p = 1).

Usage
-----
    python plot_S5B_weighted_delays.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--dir', default='.', help='folder with sim8.npz ... sim11.npz')
parser.add_argument('--base-threshold', type=float, default=0.15,
                    help='threshold at p = 1 in nS, scaled like the weights; default 0.15 (published)')
parser.add_argument('--out', default='weighted_delay_vs_density.png', help='output figure')
args = parser.parse_args()

# -----------------------------
# Global Plot Styling (rcParam)
# -----------------------------
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 20,
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
sim_files = ["sim8.npz", "sim9.npz", "sim10.npz", "sim11.npz"]
conn_percentages = [25, 50, 75, 100]

# Thresholds scaled with the connection density to filter out weak synapses
thresholds = [s * args.base_threshold for s in (4, 2, 1.33, 1)]


def get_weighted_avg(weights, delays, thresh):
    """Average delay weighted by final weight, over synapses with weight >= thresh."""
    mask = weights >= thresh
    if np.sum(weights[mask]) > 0:
        return np.sum(weights[mask] * delays[mask]) / np.sum(weights[mask])
    return np.nan


ff_weighted_delays = []
fb_weighted_delays = []

for i, filename in enumerate(sim_files):
    data = np.load(os.path.join(args.dir, filename))
    current_threshold = thresholds[i]

    # Final weights and (unchanging) axonal delays
    w_FF = data['w_FF_final']
    d_FF = data['D_FF_initial']
    w_FB = data['w_FB_final']
    d_FB = data['D_FB_initial']

    ff_avg = get_weighted_avg(w_FF, d_FF, current_threshold)
    fb_avg = get_weighted_avg(w_FB, d_FB, current_threshold)

    ff_weighted_delays.append(ff_avg)
    fb_weighted_delays.append(fb_avg)

    print(f"Density {conn_percentages[i]}% (Thresh={current_threshold:.3f}): "
          f"FF_avg={ff_avg:.2f}ms (n={np.sum(w_FF >= current_threshold)}), "
          f"FB_avg={fb_avg:.2f}ms (n={np.sum(w_FB >= current_threshold)})")

# -----------------------------
# Plotting
# -----------------------------
plt.figure()

plt.plot(conn_percentages, ff_weighted_delays, 'o-', color='blue', label='Average FF delays')
plt.plot(conn_percentages, fb_weighted_delays, 'o-', color='red', label='Average FB delays')

# Total loop delay (FF + FB)
total_loop = np.array(ff_weighted_delays) + np.array(fb_weighted_delays)
plt.plot(conn_percentages, total_loop, '--', color='black', label='Total (FF+FB)', alpha=0.7)

plt.xlabel("Projection Connection Percentage (%)")
plt.ylabel("Delay (ms)")
plt.xticks(conn_percentages)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(args.out, dpi=300, bbox_inches='tight')
print(f"Plot saved as {args.out}")
