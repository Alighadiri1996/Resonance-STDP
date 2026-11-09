"""
Supplementary Fig. S4: initial and final FF / FB weights and delay distributions for
each initial inter-layer connection probability.

Motivation
----------
Shows whether delay selection persists when the projection neurons start from sparser
inter-layer connectivity (25 %, 50 %, 75 %) instead of all-to-all (100 %).

What it measures
----------------
For sim8.npz (p = 0.25), sim9.npz (0.5), sim10.npz (0.75) written by
sim_bilayer_stdp_sparse.py and sim11.npz (p = 1, the tau_in,2 = 8 ms all-to-all run of
Figs. S2/S3), writes the same four figures per file as plot_S2.py to plots/. The
published figure shows p = 0.25, 0.5, 0.75 (upper rows from fig3_scatter, lower rows
from fig1_delays).

The "Final" delay distributions use D_*_final saved by the simulation, i.e. synapses with
final weight > 0.15 nS for every p, although initial weights and Wmax were scaled by
4, 2 and 1.33. At p = 0.25 every FB synapse starts above 0.15 nS (initial mean 0.28 nS),
so the "Final" FB distribution there also contains synapses that were depressed but
stayed above 0.15 nS. plot_S5B_weighted_delays.py uses scaled thresholds instead.

Usage
-----
    python plot_S4.py               # sim8.npz ... sim11.npz in the current folder
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--dir', default='.', help='folder with sim8.npz ... sim11.npz')
parser.add_argument('--outdir', default='plots', help='output folder')
args = parser.parse_args()

# -----------------------------
# Global Styling for Publication
# -----------------------------
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'font.family': 'sans-serif'
})
# -----------------------------
# Configuration
# -----------------------------
sim_files = [f"sim{i}.npz" for i in range(8, 12)]
output_dir = args.outdir
# Weight scale of each file (used for the axis limits)
ylims = [4, 2, 1.33, 1]

os.makedirs(output_dir, exist_ok=True)


def process_sim(data_path, prefix, current_ylim):
    print(f"Loading {data_path} with ylim={current_ylim:.2f}...")
    data = np.load(data_path)

    # Extract data
    sim_arr = data['sim_arr']
    D_FF, d_FF = data['D_FF_initial'], data['D_FF_final']
    w_FF_init, w_FF_final = data['w_FF_initial'], data['w_FF_final']
    weight_arr_FF = data['weight_trace_FF']

    D_FB, d_FB = data['D_FB_initial'], data['D_FB_final']
    w_FB_init, w_FB_final = data['w_FB_initial'], data['w_FB_final']
    weight_arr_FB = data['weight_trace_FB']

    # --- Figure 1: Delay Distributions ---
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    sns.kdeplot(D_FF, color='m', fill=True, label='Initial', bw_adjust=1.)
    sns.kdeplot(d_FF, color='g', fill=True, label='Final', bw_adjust=1.)
    plt.legend(frameon=False); plt.yticks([0, 0.25]); plt.xticks([]); plt.ylim(0, .3); plt.xlim(0, 50); plt.ylabel('FF Density')

    plt.subplot(2, 1, 2)
    sns.kdeplot(D_FB, color='blue', fill=True, label='Initial', bw_adjust=1.5)
    sns.kdeplot(d_FB, color='red', fill=True, label='Final', bw_adjust=1.5)
    plt.legend(frameon=False); plt.yticks([0, 0.25]); plt.xticks([10, 20, 30, 40]); plt.ylim(0, .3); plt.xlim(0, 50); plt.ylabel('FB Density'); plt.xlabel('Delay (ms)')
    plt.savefig(f"{output_dir}/{prefix}_fig1_delays.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 2: Weight Histograms ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.hist(w_FF_final, bins=300, color='gray'); plt.xlabel('Final FF weight (nS)')
    plt.subplot(1, 2, 2); plt.hist(w_FB_final, bins=300, color='gray'); plt.xlabel('Final FB weight (nS)')
    plt.savefig(f"{output_dir}/{prefix}_fig2_weights.png", dpi=200)
    plt.close()

    # --- Figure 3: Scatter (axis limits depend on the weight scale) ---
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(D_FF, w_FF_init, '.', color='m', markersize=4, label='FF Initial', alpha=0.5)
    plt.plot(D_FF, w_FF_final, '.', color='g', markersize=3, label='FF Final')
    plt.ylabel(r'W$_{FF}$ (nS)'); plt.xlim(-10, 50)
    plt.ylim(-0.025,)
    plt.yticks([0, 0.25])
    if current_ylim == 4:
        plt.yticks([0, 1.0])
        plt.ylim(-0.025, 1.1)
    plt.xticks([]); plt.legend(frameon=False, markerscale=8.0)

    plt.subplot(2, 1, 2)
    plt.plot(D_FB, w_FB_init, '.', color='b', markersize=4, label='FB Initial', alpha=0.5)
    plt.plot(D_FB, w_FB_final, '.', color='r', markersize=3, label='FB Final')
    plt.ylabel(r'W$_{FB}$ (nS)'); plt.xlabel('Delay (ms)'); plt.xlim(0, 50)
    plt.ylim(-0.025, )
    plt.yticks([0, 0.25])
    if current_ylim == 4:
        plt.yticks([0, 1.0])
        plt.ylim(-0.025, 1.1)
    if current_ylim == 1.33:
        plt.yticks([0, 0.25])
        plt.ylim(-0.025, 0.35)
    plt.xticks([10, 20, 30, 40]); plt.legend(frameon=False, markerscale=8.0)

    plt.savefig(f"{output_dir}/{prefix}_fig3_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 4: Traces ---
    plt.figure(figsize=(10, 5))
    n_samples = 50
    np.random.seed(42)
    selected_FF = np.random.choice(weight_arr_FF.shape[1], n_samples, replace=False)
    selected_FB = np.random.choice(weight_arr_FB.shape[1], n_samples, replace=False)
    cmap = plt.cm.jet

    plt.subplot(1, 2, 1)
    norm_FF = plt.Normalize(vmin=D_FF.min(), vmax=D_FF.max())
    for i in selected_FF:
        plt.plot(sim_arr, weight_arr_FF[:, i], color=cmap(norm_FF(D_FF[i])), lw=1)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_FF, cmap=cmap), ax=plt.gca()).set_label('FF delay (ms)')
    plt.xlabel('Time (ms)'); plt.ylabel('FF weight (nS)')
    plt.ylim(0, current_ylim)

    plt.subplot(1, 2, 2)
    norm_FB = plt.Normalize(vmin=D_FB.min(), vmax=D_FB.max())
    for i in selected_FB:
        plt.plot(sim_arr, weight_arr_FB[:, i], color=cmap(norm_FB(D_FB[i])), lw=1)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_FB, cmap=cmap), ax=plt.gca()).set_label('FB delay (ms)')
    plt.xlabel('Time (ms)'); plt.ylabel('FB weight (nS)')
    plt.ylim(0, current_ylim)

    plt.savefig(f"{output_dir}/{prefix}_fig4_traces.png", dpi=200)
    plt.close()


for i, f in enumerate(sim_files):
    path = os.path.join(args.dir, f)
    if os.path.exists(path):
        process_sim(path, f.replace(".npz", ""), ylims[i])
    else:
        print(f"Skipping {path}, file not found.")

print(f"All plots generated in the '{output_dir}/' folder.")
