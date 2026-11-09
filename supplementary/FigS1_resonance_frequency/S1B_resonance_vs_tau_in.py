"""
Resonance frequency of a single layer network as a function of its inhibitory synaptic
time constant tau_in.

Simulation for Supplementary Fig. S1B of:

    Ghadiri A, Taghavi S, Rezaei H, Aertsen A, Kumar A, Valizadeh A (2026).
    Delay selection by spike-timing-dependent plasticity shapes efficient networks
    for signal transmission. J Neurosci 46(32). doi:10.1523/JNEUROSCI.0103-26.2026

Motivation
----------
Supplementary Figs. S2 and S3 detune layer 2 by changing its tau_in. This sweep
calibrates how much each tau_in shifts the resonance frequency of a layer network.

What it measures
----------------
For tau_in = 6, 7, ..., 12 ms and n trials each (101 in the paper), builds one layer
(200 E and 50 I iaf_cond_exp neurons, fixed in-degree 0.2, Poisson drive 6700 / 4000 Hz),
injects one pulse packet (80 spikes, sd 2 ms, weight 0.8 nS) into 70 excitatory neurons
at 500 ms and simulates 1000 ms. The resonance frequency of a trial is
1000 / (t2 - t1) from the first two peaks (distance >= 10 ms, prominence >= 1.2) of the
Savitzky-Golay smoothed (11, 3) 1 ms excitatory PSTH after 500 ms; trials with fewer
than two peaks are dropped (none were in the published run). Saves per-trial
frequencies, means and SDs to complete_resonance_metadata.npz (input of S1B_plot.py).

Usage
-----
    python S1B_resonance_vs_tau_in.py                    # 7 x 101 runs of 1 s
    python S1B_resonance_vs_tau_in.py --n-trials 3       # quick check

Notes
-----
Trial seeds are base_msd + trial + int(100 * tau_in), so for any n_trials the first
trials reproduce the published per-trial values. With 101 trials, neighbouring tau_in
values share one seed (trial 100 of tau_in and trial 0 of tau_in + 1 ms). As in the
published run, this network allows autapses and multapses (NEST 2 defaults) and its
pulse packet (80 spikes, 0.8 nS) is stronger than in the plasticity simulations
(50 spikes, 0.6 nS). Changes from the script used for the paper
(resonance_stability/resonance_frequency_vs_tau_syn_in.py): number of trials and output
name are options, and matplotlib is only imported for the optional quick-look figure.
"""
import argparse
import os

import numpy as np
import nest
from scipy.signal import find_peaks, savgol_filter

# PyNEST 2.20 still uses np.float_, which NumPy 2 removed.
if not hasattr(np, 'float_'):
    np.float_ = np.float64

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--n-trials', type=int, default=101, help='trials per tau_in; default 101')
parser.add_argument('--out', default='complete_resonance_metadata.npz', help='output .npz')
args = parser.parse_args()

# -----------------------
# Parameters & Setup
# -----------------------
base_msd = 4652
tau_in_range = np.arange(6.0, 12.50, 1.0)
n_trials = args.n_trials

# Storage for plotting
mean_freqs = []
std_freqs = []
all_raw_data = []
for t_in in tau_in_range:
    trial_results = []
    print("Testing Tau_in: {:.1f}ms...".format(t_in))

    for trial in range(n_trials):
        nest.ResetKernel()

        # Every trial gets its own seed (see Notes for the one overlap per tau_in pair)
        current_seed = base_msd + trial + int(t_in * 100)
        nest.SetKernelStatus({'grng_seed': current_seed, 'rng_seeds': [current_seed + 1]})
        np.random.seed(current_seed)

        # Build network
        exc_params = {'V_th': -54., 'V_reset': -70., 'tau_syn_ex': 3.0,
                      'tau_syn_in': t_in, 'g_L': 16.67, 'C_m': 250.0, "E_in": -80.0}

        pop1_ex = nest.Create("iaf_cond_exp", 200, params=exc_params)
        pop1_in = nest.Create("iaf_cond_exp", 50, params=exc_params)

        # Initialize V_m randomly
        nest.SetStatus(pop1_ex + pop1_in, [{'V_m': np.random.uniform(-75.0, -65.0)} for _ in range(250)])

        pulse = nest.Create('pulsepacket_generator', params={'activity': 80, 'sdev': 2.0, 'pulse_times': [500.0]})
        noise = nest.Create('poisson_generator', 2, [{"rate": 6700.0}, {"rate": 4000.0}])

        # Connections
        nest.Connect(noise[:1], pop1_ex, syn_spec={'weight': 0.25, 'delay': 0.1})
        nest.Connect(noise[1:], pop1_in, syn_spec={'weight': 0.4, 'delay': 0.1})
        nest.Connect(pulse, pop1_ex[:70], syn_spec={'weight': 0.8, 'delay': 0.1})

        nest.Connect(pop1_ex, pop1_ex, {'rule': 'fixed_indegree', 'indegree': 40}, {'weight': 0.33, 'delay': 1.0})
        nest.Connect(pop1_ex, pop1_in, {'rule': 'fixed_indegree', 'indegree': 40}, {'weight': 5.5, 'delay': 1.0})
        nest.Connect(pop1_in, pop1_ex, {'rule': 'fixed_indegree', 'indegree': 10}, {'weight': -6.2, 'delay': 2.0})
        nest.Connect(pop1_in, pop1_in, {'rule': 'fixed_indegree', 'indegree': 10}, {'weight': -15.0, 'delay': 2.0})

        sd_ex = nest.Create('spike_detector')
        nest.Connect(pop1_ex, sd_ex)

        nest.Simulate(1000.0)

        # --- Peak Analysis ---
        ev_ex = nest.GetStatus(sd_ex, "events")[0]['times']
        counts, bin_edges = np.histogram(ev_ex, bins=np.arange(0, 1001.0, 1.0))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = savgol_filter(counts, 11, 3)

        post_pulse = np.where(bin_centers > 500.0)[0]
        peaks, _ = find_peaks(smoothed[post_pulse], distance=10, prominence=1.2)
        peak_times = bin_centers[post_pulse][peaks]

        if len(peak_times) >= 2:
            freq = 1000.0 / (peak_times[1] - peak_times[0])
            trial_results.append(freq)

    # Statistics for this tau_in
    if trial_results:
        all_raw_data.append(trial_results)
        mean_freqs.append(np.mean(trial_results))
        std_freqs.append(np.std(trial_results))
    else:
        all_raw_data.append([])
        mean_freqs.append(np.nan)
        std_freqs.append(0)

# -----------------------
# Save all raw data (input of S1B_plot.py)
# -----------------------
raw = np.empty(len(all_raw_data), dtype=object)
raw[:] = [list(r) for r in all_raw_data]
np.savez_compressed(
    args.out,
    tau_range=tau_in_range,
    raw_frequencies=raw,
    means=mean_freqs,
    stds=std_freqs
)
print(f"Full dataset saved successfully to {args.out}")
print(f"File size: {os.path.getsize(args.out) / 1024:.2f} KB")
for t_in, m, s, r in zip(tau_in_range, mean_freqs, std_freqs, all_raw_data):
    print(f"tau_in = {t_in:4.1f} ms: {m:6.2f} +- {s:5.2f} Hz (n = {len(r)})")

# -----------------------
# Quick-look figure (the published panel comes from S1B_plot.py)
# -----------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed: skipping the quick-look figure")
else:
    plt.figure(figsize=(8, 5))
    for i, tau in enumerate(tau_in_range):
        plt.scatter([tau] * len(all_raw_data[i]), all_raw_data[i], color='gray', alpha=0.4,
                    label='Individual Trials' if i == 0 else None)
    plt.errorbar(tau_in_range, mean_freqs, yerr=std_freqs, fmt='o-',
                 capsize=5, color='darkblue', ecolor='red', label="Mean ± SD")
    plt.xlabel(r"Inhibitory Synaptic Time Constant $\tau_{syn,in}$ (ms)")
    plt.ylabel("Resonance Frequency (Hz)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("resonance_frequency_vs_tau_syn_in.png")
