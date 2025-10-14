# ==============================================
# Two-layer spiking network with FF (static) and FB (plastic STDP)
# ==============================================

import nest
import numpy as np
import matplotlib.pyplot as plt
import nest.raster_plot

# -----------------------------
# Simulation setup
# -----------------------------
nest.ResetKernel()
nest.SetKernelStatus({'print_time': True, 'local_num_threads': 1})

msd = 1234
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd + n_vp)
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
nest.SetKernelStatus({'grng_seed': msd + n_vp, 'rng_seeds': msdrange2})

# -----------------------------
# Parameters
# -----------------------------
a = 50
sdev = 2.0
weight = 0.6
jitter = 200
pulse_times = np.arange(5000., 1995000., 1000)
pulse_times += np.random.uniform(-jitter, jitter, len(pulse_times))

NE, NI = 200, 50
NP = 70
simtime = 2000000.0
sampling = 2000.0

p_rate_ex, p_rate_in = 6700.0, 4000.0

j_exc_exc = 0.33
j_exc_inh = 5.5
j_inh_exc = -6.2
j_inh_inh = -15.0

epsilonEE = epsilonIE = epsilonEI = epsilonII = 0.2
epsilonP = 0.5
CEE = int(epsilonEE * NE)
CIE = int(epsilonIE * NI)
CEI = int(epsilonEI * NE)
CII = int(epsilonII * NI)
CEP   = int(epsilonP*NP)

g_FF, g_FB = 0.2 , 0.11   # Mean initial inter-layer connection strength
g_threshold = 0.2  # Threshold to determine the final delay distribution

# Delays
FF_delay = {'distribution': 'normal_clipped', 'mu': 14., 'sigma': 1.0, 'low': 0.1}
FB_delay = {'distribution': 'normal_clipped', 'mu': 25., 'sigma': 8.0, 'low': 0.1}

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
STDP_Wmax = 0.4

# -----------------------------
# Create populations
# -----------------------------
Layer1_ex = nest.Create("iaf_cond_exp", NE, params=exci_params)
Layer1_in = nest.Create("iaf_cond_exp", NI, params=inhi_params)

Layer2_ex = nest.Create("iaf_cond_exp", NE, params=exci_params)
Layer2_in = nest.Create("iaf_cond_exp", NI, params=inhi_params)

# Dummy neurons for FB pathway
FBdummy = nest.Create("parrot_neuron", 4900)

all_neurons = Layer1_ex + Layer1_in + Layer2_ex + Layer2_in

# -----------------------------
# Initialize membrane potentials
# -----------------------------
for gid, vp in [(ni['global_id'], ni['vp']) for ni in nest.GetStatus(all_neurons) if ni['local']]:
    nest.SetStatus([gid], {'V_m': pyrngs[vp].uniform(-75.0, -65.0)})

# -----------------------------
# Input and noise
# -----------------------------
pp = nest.Create('pulsepacket_generator', params={'activity': a, 'sdev': sdev, 'pulse_times': pulse_times})
noise = nest.Create("poisson_generator", 2, [{"rate": p_rate_ex}, {"rate": p_rate_in}])

# -----------------------------
# Recurrent connectivity
# -----------------------------
for pop_ex, pop_in in [(Layer1_ex, Layer1_in), (Layer2_ex, Layer2_in)]:
    nest.Connect(pop_ex, pop_ex, {'rule': 'fixed_indegree', 'indegree': CEE,'autapses': False,"multapses":False}, {'weight': j_exc_exc, 'delay': 1.})
    nest.Connect(pop_ex, pop_in, {'rule': 'fixed_indegree', 'indegree': CEI,'autapses': False,"multapses":False}, {'weight': j_exc_inh, 'delay': 1.})
    nest.Connect(pop_in, pop_ex, {'rule': 'fixed_indegree', 'indegree': CIE,'autapses': False,"multapses":False}, {'weight': j_inh_exc, 'delay': 2.})
    nest.Connect(pop_in, pop_in, {'rule': 'fixed_indegree', 'indegree': CII,'autapses': False,"multapses":False}, {'weight': j_inh_inh, 'delay': 2.})

nest.CopyModel('static_synapse_hom_w', 'pp_excitatory', {'weight': weight, 'delay': 0.1})
nest.Connect(pp, Layer1_ex[:NP], {'rule': 'fixed_indegree', 'indegree': 1}, 'pp_excitatory')
# -----------------------------
# Feedforward (FF) - static
# -----------------------------
nest.Connect(Layer1_ex[:NP], Layer2_ex[:NP],
             {'rule': 'fixed_indegree', 'indegree': CEP,'autapses': False,"multapses":False},
             {'delay': FF_delay, 'weight': {'distribution': 'normal', 'mu':g_FF, 'sigma': 0.01}})

# -----------------------------
# Feedback (FB) - plastic
# -----------------------------

nest.CopyModel('stdp_synapse', 'excitatory_plastic')

for i in range(70):
    nest.Connect(Layer2_ex[i:i+1], FBdummy[70*i:70*(i+1)],
                 {'rule': "all_to_all",'autapses': False,"multapses":False}, {'delay': FB_delay})
    nest.Connect(FBdummy[70*i:70*(i+1)], Layer1_ex[:NP],
                 {'rule': 'one_to_one', 'autapses': False, 'multapses': False},
                 {"model": "excitatory_plastic", 'delay': 0.1,
                  "weight": {'distribution': 'normal', 'mu': g_FB, 'sigma': 0.01},
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
SD = nest.Create('spike_detector', params={"start": 0.0, "stop": simtime})
nest.Connect(all_neurons, SD)

# -----------------------------
# Run simulation & record FB weights
# -----------------------------
sim_arr = np.arange(0., simtime, sampling)
conn_FB = nest.GetConnections(FBdummy, Layer1_ex[:NP], synapse_model='excitatory_plastic')
n_conn = len(conn_FB)
weight_arr_FB = np.zeros((len(sim_arr), n_conn))

with nest.RunManager():
    for i, t in enumerate(sim_arr):
        nest.Run(sampling)
        weight_arr_FB[i, :] = np.array(nest.GetStatus(conn_FB, "weight"))

# -----------------------------
# Analysis
# -----------------------------
conn_FB = nest.GetConnections(Layer2_ex, FBdummy)
all_FB = np.array(nest.GetStatus(conn_FB, ["source", "target", "weight", "delay"]))

idx_sorted = np.argsort(all_FB[:, 1])
sorted_FB = all_FB[idx_sorted]
D_FB = sorted_FB[:, 3]

w_FB = weight_arr_FB[-1, :]

active_idx = np.where(w_FB > g_threshold)[0]
d_FB = D_FB[active_idx % len(D_FB)]

#np.savetxt("FB_initial_delay.txt", D_FB)
#np.savetxt("FB_final_delay.txt", d_FB)
#np.savetxt("FB_final_weight.txt", w_FB)

# -----------------------------
# Plots
# -----------------------------
plt.figure(1)
plt.hist(D_FB, bins=300, label="initial delay", color='blue', alpha=0.6)
plt.hist(d_FB, bins=300, label="final delay (w>0.2)", color='red', alpha=0.6)
plt.legend()
plt.xlabel("Axonal delay (ms)")
plt.ylabel("Count")

plt.figure(2)
plt.hist(w_FB, bins=300, color='gray')
plt.xlabel("Final synaptic weights")
plt.ylabel("Count")

plt.figure(3)
plt.plot(D_FB, w_FB, '.k', markersize=2)
plt.xlabel("Initial delay (ms)")
plt.ylabel("Final weight")

plt.figure(4)
ax4 = plt.gca()
n_samples = 50
np.random.seed(42)
selected_indices = np.random.choice(n_conn, size=n_samples, replace=False)

norm = plt.Normalize(vmin=D_FB.min(), vmax=D_FB.max())
cmap = plt.cm.jet

for i in selected_indices:
    color = cmap(norm(D_FB[i]))
    ax4.plot(sim_arr, weight_arr_FB[:, i], color=color, lw=1)

#  fix for Python 2.7
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array(D_FB)
cbar = plt.colorbar(sm, ax=ax4)
cbar.set_label("Initial axonal delay (ms)")

plt.xlabel("Time (ms)")
plt.ylabel("Weight")
plt.title("Evolution of selected FB connections")


#nest.raster_plot.from_device(SD, hist=False)
plt.show()
