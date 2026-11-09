"""
Response of a single E-I network (one layer, no inter-layer projections) to one pulse
packet, and its resonance frequency.

Simulation for Supplementary Fig. S1A of:

    Ghadiri A, Taghavi S, Rezaei H, Aertsen A, Kumar A, Valizadeh A (2026).
    Delay selection by spike-timing-dependent plasticity shapes efficient networks
    for signal transmission. J Neurosci 46(32). doi:10.1523/JNEUROSCI.0103-26.2026

Motivation
----------
In the asynchronous-irregular regime each layer network responds to a transient
synchronous input with damped oscillations. The period of these oscillations (the
resonance period) is the quantity that delay selection by STDP is matched to.

What it measures
----------------
One layer (200 excitatory and 50 inhibitory iaf_cond_exp neurons, tau_syn_in = 9 ms,
fixed in-degree 0.2, Poisson drive 6700 / 4000 Hz) receives one pulse packet (50 spikes,
sd 2 ms, weight 0.6 nS) on its 70 projection neurons at t = 500 ms; 1000 ms in total.
Detects peaks of the smoothed (Savitzky-Golay 11, 3) 1 ms excitatory PSTH after the
pulse and reports f_res = 1000 / (t2 - t1) from the first two detected peaks. Saves all
spikes and the peak times to resonance_results.npz, which S1A_plot.py turns into the
published panel; also writes a quick-look figure (raster_resonance.png) when
matplotlib is installed.

Usage
-----
    python S1A_single_network_response.py              # writes ./resonance_results.npz
    python S1A_single_network_response.py --outdir results

Notes
-----
The run is fully seeded (master seed 46521), so it reproduces data/resonance_results.npz
on NEST 2.20.2 with one thread. As in the published run, the recurrent connections of
this script allow autapses and multapses (NEST 2 defaults) and V_m starts uniform in
[-65, -55] mV, unlike the bi-layer simulations. Changes from the script used for the
paper (res_freq.py): output folder option, docstring, and matplotlib only imported for
the optional quick-look figure.
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
parser.add_argument('--outdir', default='.', help='folder for resonance_results.npz; default .')
args = parser.parse_args()

# -----------------------
# Seeds
# -----------------------
msd = 46521
nest.ResetKernel()
n_vp = nest.GetKernelStatus()['total_num_virtual_procs']
msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)

nest.SetKernelStatus({'grng_seed': msd + n_vp, 'rng_seeds': list(msdrange2)})
np.random.seed(msd)

# -----------------------
# Parameters
# -----------------------
NE, NI = 200, 50
NP = 70
simtime = 1000.0
pulse_time = 500.0

# Connectivity
epsilonEE, epsilonIE, epsilonEI, epsilonII = 0.2, 0.2, 0.2, 0.2
CEE, CIE, CEI, CII = int(epsilonEE*NE), int(epsilonIE*NI), int(epsilonEI*NE), int(epsilonII*NI)

# Params
exc_params = {'V_th': -54., 'V_reset': -70., 'tau_syn_ex': 3., 'tau_syn_in': 9.,
              'g_L': 16.67, 'C_m': 250.0, "E_in": -80.0}

# -----------------------
# Construction
# -----------------------
pop1_ex = nest.Create("iaf_cond_exp", NE, params=exc_params)
pop1_in = nest.Create("iaf_cond_exp", NI, params=exc_params)

# Random Initial Membrane Potential
nest.SetStatus(pop1_ex + pop1_in, [{'V_m': np.random.uniform(-65.0, -55.0)} for _ in range(NE+NI)])

# Noise & Pulse
pulse = nest.Create('pulsepacket_generator', params={'activity': 50, 'sdev': 2.0, 'pulse_times': [pulse_time]})
noise = nest.Create('poisson_generator', 2, [{"rate": 6700.0}, {"rate": 4000.0}])

# Connections internal to Pop 1
nest.Connect(noise[:1], pop1_ex, syn_spec={'weight': 0.25, 'delay': 0.1})
nest.Connect(noise[1:], pop1_in, syn_spec={'weight': 0.4, 'delay': 0.1})
nest.Connect(pulse, pop1_ex[:NP], syn_spec={'weight': 0.6, 'delay': 0.1})

nest.Connect(pop1_ex, pop1_ex, {'rule': 'fixed_indegree', 'indegree': CEE}, {'weight': 0.33, 'delay': 1.0})
nest.Connect(pop1_ex, pop1_in, {'rule': 'fixed_indegree', 'indegree': CEI}, {'weight': 5.5, 'delay': 1.0})
nest.Connect(pop1_in, pop1_ex, {'rule': 'fixed_indegree', 'indegree': CIE}, {'weight': -6.2, 'delay': 2.0})
nest.Connect(pop1_in, pop1_in, {'rule': 'fixed_indegree', 'indegree': CII}, {'weight': -15.0, 'delay': 2.0})

# --- Recording ---
sd_ex = nest.Create('spike_detector')
sd_in = nest.Create('spike_detector')
nest.Connect(pop1_ex, sd_ex)
nest.Connect(pop1_in, sd_in)

# -----------------------
# Simulation
# -----------------------
nest.Simulate(simtime)

ev_ex_all = nest.GetStatus(sd_ex, "events")[0]
ev_in_all = nest.GetStatus(sd_in, "events")[0]

# -----------------------
# Resonance frequency from the PSTH peaks after the pulse
# -----------------------
ev_ex = ev_ex_all['times']
bins = np.arange(0, simtime, 1.0)  # 1 ms bins
counts, bin_edges = np.histogram(ev_ex, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Smooth the PSTH to find clean peaks (window 11, polynomial order 3)
smoothed_counts = savgol_filter(counts, 11, 3)

# Peaks after the pulse time
post_pulse_idx = np.where(bin_centers > pulse_time)[0]
peaks, _ = find_peaks(smoothed_counts[post_pulse_idx], distance=10, prominence=2)
peak_times = bin_centers[post_pulse_idx][peaks]

freq_hz = np.nan
delta_t = np.nan
if len(peak_times) >= 2:
    t1 = peak_times[0]
    t2 = peak_times[1]
    delta_t = t2 - t1  # Period in ms
    freq_hz = 1000.0 / delta_t  # Frequency in Hz
    print("\n--- Resonance Analysis ---")
    print("First Peak:  {:.2f} ms".format(t1))
    print("Second Peak: {:.2f} ms".format(t2))
    print("Period (T):  {:.2f} ms".format(delta_t))
    print("Frequency:   {:.2f} Hz".format(freq_hz))
else:
    print("Could not find enough peaks to calculate resonance.")

# -----------------------
# Save (input of S1A_plot.py)
# -----------------------
os.makedirs(args.outdir, exist_ok=True)
filename = os.path.join(args.outdir, 'resonance_results.npz')
np.savez(filename,
         freq_hz=freq_hz,
         delta_t=delta_t,
         peak_times=peak_times,
         exc_spikes=ev_ex_all['times'],
         exc_ids=ev_ex_all['senders'],
         inh_spikes=ev_in_all['times'],
         inh_ids=ev_in_all['senders'])
print("\nData successfully saved to: {}".format(filename))

# -----------------------
# Quick-look figure (the published panel comes from S1A_plot.py)
# -----------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed: skipping raster_resonance.png")
else:
    def plot_raster(ev, color, offset=0):
        ts = ev['times']
        gids = ev['senders']
        if len(gids) > 0:
            # Zero the GIDs of this recording, then add the offset
            plt.plot(ts, gids - min(gids) + offset, '|', color=color, markersize=2)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plot_raster(ev_ex_all, "blue")
    plot_raster(ev_in_all, "red", offset=NE)
    plt.axvline(pulse_time, color='green', linestyle='--', label='Pulse', alpha=0.5)
    plt.ylabel("Neuron ID")
    plt.legend(['Exc', 'Inh', 'Pulse'])
    plt.xlim(100, 800)

    plt.subplot(2, 1, 2)
    plt.hist(ev_ex, bins=np.arange(0, simtime, 3.0), color='blue', alpha=0.7)
    plt.plot(bin_centers, smoothed_counts, color='red', label='Smoothed PSTH')
    plt.plot(peak_times, smoothed_counts[post_pulse_idx][peaks], "ro", label="Detected Peaks")
    plt.xlim(100, 800)
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "raster_resonance.png"))
