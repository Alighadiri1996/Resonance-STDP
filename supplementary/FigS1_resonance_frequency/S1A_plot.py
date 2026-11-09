"""
Supplementary Fig. S1A: raster and excitatory population activity of a single layer
network around one pulse packet.

Motivation
----------
Shows the damped oscillation that a pulse packet evokes in an isolated layer network;
its period defines the network resonance frequency.

What it measures
----------------
Reads the spikes saved by S1A_single_network_response.py and plots (top) the raster of
excitatory (blue) and inhibitory (red) neurons with the pulse time, and (bottom) the
excitatory spike count in 3 ms bins, between 100 and 800 ms.

Usage
-----
    python S1A_plot.py                                   # published data
    python S1A_plot.py --data resonance_results.npz      # after re-running the simulation

Notes
-----
The double arrow labelled T_res is drawn at fixed positions (500 to 550 ms) as a visual
guide. The period computed by the simulation script is the interval between the first
two detected PSTH peaks after the pulse (551.5 and 597.5 ms in the published run,
46 ms, 21.7 Hz).
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--data', default='data/resonance_results.npz', help='input .npz')
parser.add_argument('--out', default='FigS1A_raster_resonance.png', help='output figure')
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

# 1. Load the data
data = np.load(args.data)

ts_ex = data['exc_spikes']
gids_ex = data['exc_ids']
ts_in = data['inh_spikes']
gids_in = data['inh_ids']

# Constants of the simulation (not stored in the .npz)
NE = 200
pulse_time = 500.0
simtime = 1000.0


# 2. Raster helper
def plot_raster_from_data(ts, gids, color, offset=0):
    if len(gids) > 0:
        normalized_gids = gids - min(gids) + offset
        plt.plot(ts, normalized_gids, 'o', color=color, markersize=2)


# 3. Two-panel figure
plt.figure(figsize=(10, 8))

# --- Top: raster ---
plt.subplot(2, 1, 1)
plot_raster_from_data(ts_ex, gids_ex, "blue")
plot_raster_from_data(ts_in, gids_in, "red", offset=NE)
plt.axvline(pulse_time, color='green', linestyle='--', lw=2, alpha=0.75, label='Pulse')
plt.ylabel("Neuron ID")
plt.xlim(100, 800)
plt.legend(['Exc', 'Inh', 'Pulse'], loc='upper right', frameon=True, markerscale=5)

# --- Bottom: excitatory spike count, 3 ms bins ---
plt.subplot(2, 1, 2)
t_start = 500.0
t_end = 550.0  # visual guide only, see Notes
y_pos = 15

bins = np.arange(0, simtime + 3, 3.0)
plt.hist(ts_ex, bins=bins, color='blue', alpha=0.7, label='Exc Count')

plt.annotate('',
             xy=(t_end, y_pos),
             xytext=(t_start, y_pos),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
plt.text((t_start + t_end)/2, y_pos + 1, r'T$_{res}$',
         ha='center', va='bottom', fontsize=18, fontweight='bold')

plt.axvline(pulse_time, color='green', linestyle='--', alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Spike Count")
plt.xlim(100, 800)
plt.legend(loc='upper right', frameon=True, markerscale=2)
plt.tight_layout()
plt.savefig(args.out, dpi=300, bbox_inches='tight')
print(f"Saved {args.out}")
