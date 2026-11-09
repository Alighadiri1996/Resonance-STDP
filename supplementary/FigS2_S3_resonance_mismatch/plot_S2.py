"""
Supplementary Fig. S2: initial and final FF / FB weights and delay distributions for
each inhibitory time constant of layer 2.

Motivation
----------
Shows how delay selection of FF and FB connections changes when the resonance
frequency of layer 2 is detuned (tau_in,2 = 6 to 12 ms; layer 1 at 8 ms).

What it measures
----------------
For each sim<i>.npz (i = 1..7 for tau_in,2 = 6..12 ms) written by
sim_bilayer_stdp_tau_mismatch.py, writes four figures to plots/:
  sim<i>_fig1_delays.png   KDE of all initial delays and of the delays of synapses with
                           final weight > 0.15 nS (D_*_final), FF top, FB bottom
  sim<i>_fig2_weights.png  histograms of the final weights
  sim<i>_fig3_scatter.png  weight vs delay, initial and final, FF top, FB bottom
  sim<i>_fig4_traces.png   weight traces of 50 random synapses, coloured by delay
The published figure shows tau_in,2 = 6, 9 and 12 ms (sim1, sim4, sim7), upper panels
from fig3_scatter and lower panels from fig1_delays.

Usage
-----
    python plot_S2.py                 # sim1.npz ... sim7.npz in the current folder
    python plot_S2.py --dir results
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--dir', default='.', help='folder with sim1.npz ... sim7.npz')
parser.add_argument('--outdir', default='plots', help='output folder')
args = parser.parse_args()

# --- Global Font Settings ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.dpi': 300,
    'savefig.dpi': 300
})
# -----------------------------
# Configuration
# -----------------------------
sim_files = [f"sim{i}.npz" for i in range(1, 8)]
output_dir = args.outdir
os.makedirs(output_dir, exist_ok=True)


def process_sim(data_path, prefix):
    print(f"Loading {data_path}...")
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

    # --- Figure 3: Scatter ---
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(D_FF, w_FF_init, '.', color='m', markersize=4, label='FF Initial', alpha=0.5)
    plt.plot(D_FF, w_FF_final, '.', color='g', markersize=3, label='FF Final')
    plt.legend(frameon=False, markerscale=8.0)
    plt.ylabel(r'W$_{FF}$ (nS)'); plt.xlim(0, 50); plt.ylim(-0.05, 0.35); plt.xticks([10, 20, 30, 40], ["", "", "", ""])
    plt.yticks([0, 0.25])

    plt.subplot(2, 1, 2)
    plt.plot(D_FB, w_FB_init, '.', color='b', markersize=4, label='FB Initial', alpha=0.5)
    plt.plot(D_FB, w_FB_final, '.', color='r', markersize=3, label='FB Final')
    plt.ylabel(r'W$_{FB}$ (nS)'); plt.xlabel('Delay (ms)'); plt.xlim(0, 50); plt.ylim(-.05, 0.35); plt.xticks([10, 20, 30, 40]); plt.yticks([0, 0.25])
    plt.legend(frameon=False, markerscale=8.0)
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

    plt.subplot(1, 2, 2)
    norm_FB = plt.Normalize(vmin=D_FB.min(), vmax=D_FB.max())
    for i in selected_FB:
        plt.plot(sim_arr, weight_arr_FB[:, i], color=cmap(norm_FB(D_FB[i])), lw=1)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_FB, cmap=cmap), ax=plt.gca()).set_label('FB delay (ms)')
    plt.xlabel('Time (ms)'); plt.ylabel('FB weight (nS)')
    plt.savefig(f"{output_dir}/{prefix}_fig4_traces.png", dpi=300, bbox_inches='tight')
    plt.close()


for f in sim_files:
    path = os.path.join(args.dir, f)
    if os.path.exists(path):
        process_sim(path, f.replace(".npz", ""))
    else:
        print(f"Skipping {path}, file not found.")

print(f"All plots generated in the '{output_dir}/' folder.")
