"""
Supplementary Fig. S5A: percentage of potentiated FF and FB connections versus the
initial inter-layer connection probability.

Motivation
----------
Compares how many synapses survive learning when the projection neurons start from
25 %, 50 %, 75 % or 100 % connectivity, with weights rescaled by 1 / p.

What it measures
----------------
For sim8.npz ... sim11.npz (p = 0.25, 0.5, 0.75, 1), the percentage of FF and FB
synapses whose final weight is >= threshold, relative to all initial synapses.
The thresholds used for the published panel equal the scaled initial mean weight,
0.07 nS * (4, 2, 1.33, 1) = 0.28, 0.14, 0.093, 0.07 nS. (Fig. S5B uses
0.15 nS * (4, 2, 1.33, 1) = 0.6, 0.3, 0.2, 0.15 nS.) Pass --base-threshold 0.15 for
the Fig. S5B thresholds.

Usage
-----
    python plot_S5A_stabilized_connections.py
    python plot_S5A_stabilized_connections.py --base-threshold 0.15

Change from the script used for the paper (v2/plot3.py): the legend now reads
"Feedforward (FF)" and "Feedback (FB)"; the published panel shows "Average FF/FB delays".
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--dir', default='.', help='folder with sim8.npz ... sim11.npz')
parser.add_argument('--base-threshold', type=float, default=0.07,
                    help='threshold at p = 1 in nS, scaled like the weights; default 0.07 (published)')
parser.add_argument('--out', default='stabilization_vs_density_dynamic_thresh.png', help='output figure')
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

# Thresholds scaled with the connection density (weight scale 4, 2, 1.33, 1)
thresholds = [s * args.base_threshold for s in (4, 2, 1.33, 1)]

ff_percentages = []
fb_percentages = []

for i, filename in enumerate(sim_files):
    data = np.load(os.path.join(args.dir, filename))
    current_threshold = thresholds[i]

    w_FF_initial = data['w_FF_initial']
    w_FF_final = data['w_FF_final']
    w_FB_initial = data['w_FB_initial']
    w_FB_final = data['w_FB_final']

    # Percentages relative to all initial synapses of this file
    ff_pct = (np.sum(w_FF_final >= current_threshold) / len(w_FF_initial)) * 100
    fb_pct = (np.sum(w_FB_final >= current_threshold) / len(w_FB_initial)) * 100

    ff_percentages.append(ff_pct)
    fb_percentages.append(fb_pct)

    print(f"Density {conn_percentages[i]}% (Thresh={current_threshold:.3f}): FF={ff_pct:.1f}%, FB={fb_pct:.1f}%")

# -----------------------------
# Plotting
# -----------------------------
plt.figure()

plt.plot(conn_percentages, ff_percentages, 'o-', color='blue', label='Feedforward (FF)')
plt.plot(conn_percentages, fb_percentages, 'o-', color='red', label='Feedback (FB)')

plt.xlabel("Projection Connection Percentage (%)")
plt.ylabel("Stabilized Connections (%)")
plt.ylim(-5, 105)
plt.xticks(conn_percentages)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(args.out, dpi=300, bbox_inches='tight')
print(f"Plot saved as {args.out}")
