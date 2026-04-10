from __future__ import division, print_function
import nest
import nest.raster_plot # Specific for legacy NEST plotting
import numpy as np
import matplotlib.pyplot as plt

# Fix for the NumPy 2.0 error in Docker
if not hasattr(np, 'float_'): np.float_ = np.float64

# -----------------------
# Seeds
# -----------------------
msd = 4652
nest.ResetKernel()
n_vp = nest.GetKernelStatus()['total_num_virtual_procs']
msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
# print("*************",msdrange2)

nest.SetKernelStatus({'grng_seed': msd + n_vp, 'rng_seeds': list(msdrange2)})
# nest.SetKernelStatus({'grng_seed': msd , 'rng_seeds': [msd]})
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
nest.SetStatus(pop1_ex + pop1_in, [{'V_m': np.random.uniform(-75.0, -65.0)} for _ in range(NE+NI)])

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

# -----------------------
# Plotting
# -----------------------
def plot_raster(sd, title, color, offset=0):
    ev = nest.GetStatus(sd, "events")[0]
    ts = ev['times']
    gids = ev['senders']
    
    if len(gids) > 0:
        # Subtract the smallest GID in this specific recording 
        # to "zero" the indices, then add your custom offset.
        normalized_gids = gids - min(gids) + offset
        plt.plot(ts, normalized_gids, '|', color=color, markersize=2)

plt.figure(figsize=(10, 8))

# Subplot 1: Raster
plt.subplot(2, 1, 1)
plot_raster(sd_ex, "Excitatory", "blue")
plot_raster(sd_in, "Inhibitory", "red", offset=NE)
plt.axvline(pulse_time, color='green', linestyle='--', label='Pulse')
# plt.title("Population 1 Raster Plot (Disconnected from Pop 2)")
plt.ylabel("Neuron ID")
plt.legend(['Exc', 'Inh', 'Pulse'])

# Subplot 2: PSTH (To see resonance frequency)
plt.subplot(2, 1, 2)
ev_ex = nest.GetStatus(sd_ex, "events")[0]['times']
bins = np.arange(0, simtime, 2.0)
plt.hist(ev_ex, bins=bins, color='blue', alpha=0.7)
# plt.title("Population Activity (PSTH)")
plt.xlabel("Time (ms)")
plt.ylabel("Spike Count")
#plt.xticks(np.arange(0,1001,40))
plt.tight_layout()
plt.savefig("raster_resonance.png")
plt.show()



from scipy.signal import find_peaks, savgol_filter

# 1. Get the histogram data (already in your script, but let's ensure variables)
ev_ex = nest.GetStatus(sd_ex, "events")[0]['times']
bins = np.arange(0, simtime, 1.0) # 1ms bins for better resolution
counts, bin_edges = np.histogram(ev_ex, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 2. Smooth the PSTH to find clean peaks
# Window size 11, polynomial order 3 (adjust if too blurry or too noisy)
smoothed_counts = savgol_filter(counts, 11, 3)

# 3. Find peaks after the pulse_time (500ms)
# We only look at indices where bin_centers > pulse_time
post_pulse_idx = np.where(bin_centers > pulse_time)[0]
peaks, _ = find_peaks(smoothed_counts[post_pulse_idx], distance=10, prominence=2)

# Map peak indices back to actual times
peak_times = bin_centers[post_pulse_idx][peaks]

if len(peak_times) >= 2:
    # The first peak is usually the direct response to the pulse
    # The second peak is the first "resonance" oscillation
    t1 = peak_times[0]
    t2 = peak_times[1]
    
    delta_t = t2 - t1  # Period in ms
    freq_hz = 1000.0 / delta_t  # Frequency in Hz
    
    print("\n--- Resonance Analysis ---")
    print("First Peak:  {:.2f} ms".format(t1))
    print("Second Peak: {:.2f} ms".format(t2))
    print("Period (T):  {:.2f} ms".format(delta_t))
    print("Frequency:   {:.2f} Hz".format(freq_hz))
    
    # Optional: Highlight them on your plot
    plt.plot(peak_times, smoothed_counts[post_pulse_idx][peaks], "ro", label="Detected Peaks")
    plt.legend()
else:
    print("Could not find enough peaks to calculate resonance.")