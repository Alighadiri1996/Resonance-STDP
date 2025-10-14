# ==============================================
# Two-layer spiking network with inter-layer STDP
# ==============================================

import nest
import numpy as np
import matplotlib.pyplot as plt
import pylab
import nest.raster_plot

# -----------------------------
# Simulation setup
# -----------------------------
nest.ResetKernel()
nest.SetKernelStatus({
    'print_time': True,
    'local_num_threads': 1,
    'overwrite_files': True
})

msd = 610
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd + n_vp)
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
nest.SetKernelStatus({'grng_seed': msd + n_vp, 'rng_seeds': msdrange2})

# -----------------------------
# Parameters
# -----------------------------
a = 50                # spikes per pulse packet
sdev = 2.0            # pulse width (ms)
weight = 0.6          # PP amplitude (mV)
jitter = 100          # ms
pulse_times = np.arange(1000., 490000., 500)
pulse_times += np.random.uniform(-jitter, jitter, len(pulse_times))

NE, NI = 200, 50
NP = 70
simtime = 500000.0  # total simulation time
sampling = 2000.0    # time step for discrete weight recording

in_delay_ex, in_delay_in = 1.0, 2.0
p_rate_ex, p_rate_in = 6700.0, 4000.0

# Synaptic weights
j_exc_exc = 0.33
j_exc_inh = 5.5
j_inh_exc = -6.2
j_inh_inh = -15.0

# Connection probabilities
epsilonEE = epsilonIE = epsilonEI = epsilonII = 0.2


CEE = int(epsilonEE * NE)
CIE = int(epsilonIE * NI)
CEI = int(epsilonEI * NE)
CII = int(epsilonII * NI)

g = 0.15    # Mean initial inter-layer connection strength
g_threshold = 0.2  # Threshold to determine the final delay distribution

# Inter-layer delay distribution
bet_delay = {'distribution': 'normal_clipped', 'mu': 10., 'sigma': 3.0, 'low': 0.1}

# -----------------------------
# Neuron parameters
# -----------------------------
exci_params = {
    'V_th': -54.,
    'V_reset': -70.,
    'tau_syn_ex': 3.0,
    'tau_syn_in': 8.0,
    'tau_minus': 19.3,
    'E_in': -80.0,
    'g_L': 16.67,
    'C_m': 250.0
}
inhi_params = exci_params.copy()

# -----------------------------
# STDP parameters
# -----------------------------
nest.SetDefaults('stdp_synapse', {
    'alpha': 0.9,
    'lambda': 0.005,
    'tau_plus': 15.9,
    'mu_plus': 0.0,
    'mu_minus': 0.0
})
STDP_Wmax = 0.3

# -----------------------------
# Create populations
# -----------------------------
Layer1_ex = nest.Create("iaf_cond_exp", NE, params=exci_params)
Layer1_in = nest.Create("iaf_cond_exp", NI, params=inhi_params)

Layer2_ex = nest.Create("iaf_cond_exp", NE, params=exci_params)
Layer2_in = nest.Create("iaf_cond_exp", NI, params=inhi_params)

# Dummy neurons for axonal delay modeling (FF)
FF_dummy = nest.Create("parrot_neuron", 4900)

all_neurons = Layer1_ex + Layer1_in + Layer2_ex + Layer2_in

# Initialize membrane potentials randomly
for gid, vp in [(ni['global_id'], ni['vp']) for ni in nest.GetStatus(all_neurons) if ni['local']]:
    nest.SetStatus([gid], {'V_m': pyrngs[vp].uniform(-75.0, -65.0)})

# -----------------------------
# Input and background noise
# -----------------------------
pp = nest.Create('pulsepacket_generator', params={'activity': a, 'sdev': sdev, 'pulse_times': pulse_times})
noise = nest.Create("poisson_generator", 2, [{"rate": p_rate_ex}, {"rate": p_rate_in}])

# -----------------------------
# Recurrent E-I connectivity
# -----------------------------
for pop_ex, pop_in in [(Layer1_ex, Layer1_in), (Layer2_ex, Layer2_in)]:
    nest.Connect(pop_ex, pop_ex, {'rule': 'fixed_indegree', 'indegree': CEE,'autapses': False,"multapses":False},
                 {'weight': j_exc_exc, 'delay': in_delay_ex})
    nest.Connect(pop_ex, pop_in, {'rule': 'fixed_indegree', 'indegree': CEI,'autapses': False,"multapses":False},
                 {'weight': j_exc_inh, 'delay': in_delay_ex})
    nest.Connect(pop_in, pop_ex, {'rule': 'fixed_indegree', 'indegree': CIE,'autapses': False,"multapses":False},
                 {'weight': j_inh_exc, 'delay': in_delay_in})
    nest.Connect(pop_in, pop_in, {'rule': 'fixed_indegree', 'indegree': CII,'autapses': False,"multapses":False},
                 {'weight': j_inh_inh, 'delay': in_delay_in})

# -----------------------------
# 
# -----------------------------
nest.CopyModel('static_synapse_hom_w', 'pp_excitatory', {'weight': weight, 'delay': 0.1})
nest.Connect(pp, Layer1_ex[:NP], {'rule': 'fixed_indegree', 'indegree': 1}, 'pp_excitatory')

wr = nest.Create('weight_recorder')
nest.CopyModel('stdp_synapse', 'excitatory_plastic', {'weight_recorder': wr[0]})

# Feedforward (FF) connections
for i in range(70):
    nest.Connect(Layer1_ex[i:i+1], FF_dummy[70*i:70*(i+1)],
                 {'rule': "all_to_all",'autapses': False,"multapses":False}, {'delay': bet_delay})
    nest.Connect(FF_dummy[70*i:70*(i+1)], Layer2_ex[:NP],
                 {'rule': 'one_to_one', 'autapses': False, 'multapses': False},
                 {"model": "excitatory_plastic", 'delay': 0.1,
                  "weight": {'distribution': 'normal', 'mu': g, 'sigma': 0.01},
                  "Wmax": {'distribution': 'normal', 'mu': STDP_Wmax, 'sigma': 0.01}})

# -----------------------------
# Noise inputs
# -----------------------------
nest.CopyModel('static_synapse_hom_w', 'NOISE_EXCI_SYN', {'weight': 0.25, 'delay': 0.1})
nest.CopyModel('static_synapse_hom_w', 'NOISE_INHI_SYN', {'weight': 0.4, 'delay': 0.1})

nest.Connect(noise[:1], Layer1_ex + Layer2_ex, syn_spec='NOISE_EXCI_SYN')
nest.Connect(noise[1:], Layer1_in + Layer2_in, syn_spec='NOISE_INHI_SYN')

# -----------------------------
# Spike detector
# -----------------------------
SD1 = nest.Create('spike_detector', params={"start": 0.0, "stop": simtime, "to_file": True, "label": "spikes"})
nest.Connect(all_neurons, SD1)

# -----------------------------
# Run simulation in discrete steps and record weights
# -----------------------------
sim_arr = np.arange(0., simtime, sampling)
conn_FF_L2 = nest.GetConnections(FF_dummy, Layer2_ex, synapse_model='excitatory_plastic')
n_connections = len(conn_FF_L2)

weight_arr = np.zeros((len(sim_arr), n_connections))

with nest.RunManager():
     for i, t in enumerate(sim_arr):
         nest.Run(sampling)
         weight_arr[i, :] = np.array(nest.GetStatus(conn_FF_L2, "weight"))

# -----------------------------
# Analysis of final weights and delays
# -----------------------------
conn_FF = nest.GetConnections(Layer1_ex, FF_dummy)
all_FF = np.array(nest.GetStatus(conn_FF, ["source", "target", "weight", "delay"]))

idx_sorted = np.argsort(all_FF[:, 1])
sorted_FF = all_FF[idx_sorted]
D_FF = sorted_FF[:, 3]

w_FF = weight_arr[-1, :]

active_idx = np.where(w_FF > g_threshold)[0]
d_FF = D_FF[active_idx % len(D_FF)]


# -----------------------------
# Plots
# -----------------------------
plt.figure(1)
plt.hist(D_FF, bins=300, label="initial delay", color='blue', alpha=0.6)
plt.hist(d_FF, bins=300, label="final delay (w>0.2)", color='red', alpha=0.6)
plt.legend()
plt.xlabel("Axonal delay (ms)")
plt.ylabel("Count")

plt.figure(2)
plt.hist(w_FF, bins=300, color='gray')
plt.xlabel("Final synaptic weights")
plt.ylabel("Count")

plt.figure(3)
plt.plot(D_FF, w_FF, '.k', markersize=2)
plt.xlabel("Initial delay (ms)")
plt.ylabel("Final weight")

# -----------------------------
# Figure 4: Evolution of selected FF connections colored by initial delay
# -----------------------------
plt.figure(4)
ax4 = plt.gca()

n_connections = len(D_FF)   # total FF connections
n_samples = 50              # number of random connections to plot
np.random.seed(42)
selected_indices = np.random.choice(n_connections, size=n_samples, replace=False)

# colormap normalization based on initial delays
norm = plt.Normalize(vmin=D_FF.min(), vmax=D_FF.max())
cmap = plt.cm.jet

# plot each selected connection
for i in selected_indices:
    color = cmap(norm(D_FF[i]))           # color based on initial delay
    ax4.plot(sim_arr, weight_arr[:, i], color=color, lw=1, label=str(D_FF[i]))

# add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cbar = plt.colorbar(sm, ax=ax4)
cbar.set_label("Initial axonal delay (ms)")

plt.xlabel("Time (ms)")
plt.ylabel("Weight")
plt.title("Evolution of selected FF connections")



# -----------------------------
# Print statistics
# -----------------------------
print("Initial delays: mean={0:.2f}, std={1:.2f}".format(np.mean(D_FF), np.std(D_FF)))
print("Final weights: mean={0:.2f}".format(np.mean(w_FF)))
print("Selected (w>0.2) final delays: mean={0:.2f}, std={1:.2f}".format(np.mean(d_FF), np.std(d_FF)))

# Raster plot
nest.raster_plot.from_device(SD1, hist=False)
plt.show()
