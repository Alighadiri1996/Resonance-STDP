from __future__ import division, print_function
import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

if not hasattr(np, 'float_'): np.float_ = np.float64

# -----------------------
# Parameters & Setup
# -----------------------
base_msd = 4652
tau_in_range = np.arange(6.0, 12.50, 1.0) 
n_trials = 51

# Storage for plotting
mean_freqs = []
std_freqs = []
all_raw_data=[]
for t_in in tau_in_range:
    trial_results = []
    print("Testing Tau_in: {:.1f}ms...".format(t_in))
    
    for trial in range(n_trials):
        nest.ResetKernel()
        
        # Ensure every trial has a unique seed
        current_seed = base_msd + trial + int(t_in * 100)
        nest.SetKernelStatus({'grng_seed': current_seed, 'rng_seeds': [current_seed + 1]})
        np.random.seed(current_seed)
        
        # Build network (same as before)
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
            
    # Calculate statistics for this Tau_in
    if trial_results:
        all_raw_data.append(trial_results)
        mean_freqs.append(np.mean(trial_results))
        std_freqs.append(np.std(trial_results))
    else:
        mean_freqs.append(np.nan)
        std_freqs.append(0)

# -----------------------
# Plotting with Error Bars
# -----------------------
plt.figure(figsize=(8, 5))
for i, tau in enumerate(tau_in_range):
    # Create an array of the current tau value, same length as trial_results
    x_values = [tau] * len(all_raw_data[i])
    y_values = all_raw_data[i]
    
    # Use alpha=0.3 to make the points semi-transparent
    if i == 0: # Add label only once for the legend
        plt.scatter(x_values, y_values, color='gray', alpha=0.4, label='Individual Trials')
    else:
        plt.scatter(x_values, y_values, color='gray', alpha=0.4)
plt.errorbar(tau_in_range, mean_freqs, yerr=std_freqs, fmt='o-', 
             capsize=5, color='darkblue', ecolor='red', label="Mean ± SD")

plt.xlabel(r"Inhibitory Synaptic Time Constant $\tau_{syn,in}$ (ms)")
plt.ylabel("Resonance Frequency (Hz)")
# plt.title("Resonance Frequency Stability (Average of 9 Trials)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig("resonance_frequency_vs_tau_syn_in.png")
plt.show()