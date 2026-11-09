# ==============================================
# Two-layer spiking network with plastic FF & FB (STDP) (AI-state)
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
pulse_times = np.arange(5000., 2495000., 1000)
pulse_times += np.random.uniform(-jitter, jitter, len(pulse_times))

NE, NI = 200, 50
NP = 70
simtime = 2500000.0
sampling = 2000.0

p_rate_ex, p_rate_in = 6700.0, 4000.0

j_exc_exc = 0.33
j_exc_inh = 5.5
j_inh_exc = -6.2
j_inh_inh = -15.0

epsilonEE = epsilonIE = epsilonEI = epsilonII = 0.2

CEE = int(epsilonEE * NE)
CIE = int(epsilonIE * NI)
CEI = int(epsilonEI * NE)
CII = int(epsilonII * NI)

g = 0.07    # Mean initial inter-layer connection strength
g_threshold = 0.15  # Threshold to determine the final delay distribution

# Delays
FF_delay = {'distribution': 'normal_clipped', 'mu': 16., 'sigma': 2.0, 'low': 0.1}
FB_delay = {'distribution': 'normal_clipped', 'mu': 20., 'sigma': 8.0, 'low': 0.1}

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
STDP_Wmax = 0.22

# -----------------------------
# Create populations
# -----------------------------
Layer1_ex = nest.Create("iaf_cond_exp", NE, params=exci_params)
Layer1_in = nest.Create("iaf_cond_exp", NI, params=inhi_params)
Layer2_ex = nest.Create("iaf_cond_exp", NE, params=exci_params)
Layer2_in = nest.Create("iaf_cond_exp", NI, params=inhi_params)

FFdummy = nest.Create("parrot_neuron", 4900)
FBdummy = nest.Create("parrot_neuron", 4900)

all_neurons = Layer1_ex + Layer1_in + Layer2_ex + Layer2_in

# -----------------------------
# Initialize membrane potentials
# -----------------------------
for ni in nest.GetStatus(all_neurons):
    if ni['local']:
        nest.SetStatus([ni['global_id']], {'V_m': pyrngs[ni['vp']].uniform(-75.0, -65.0)})

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
# Plastic Feedforward (FF)
# -----------------------------
nest.CopyModel('stdp_synapse', 'excitatory-plastic')
for i in range(70):
    nest.Connect(Layer1_ex[i:i+1], FFdummy[70*i:70*(i+1)], {'rule': "all_to_all",'autapses': False,"multapses":False}, {'delay': FF_delay})
    nest.Connect(FFdummy[70*i:70*(i+1)], Layer2_ex[:NP],
                 {'rule': 'one_to_one', 'autapses': False, 'multapses': False},
                 {"model": "excitatory-plastic", 'delay': 0.1,
                  "weight": {'distribution': 'normal_clipped', 'mu': g, 'sigma': 0.03,"low":0.01},
                  "Wmax": {'distribution': 'normal', 'mu': STDP_Wmax, 'sigma': 0.03}})

# -----------------------------
# Plastic Feedback (FB)
# -----------------------------

for i in range(70):
    nest.Connect(Layer2_ex[i:i+1], FBdummy[70*i:70*(i+1)], {'rule': "all_to_all",'autapses': False,"multapses":False}, {'delay': FB_delay})
    nest.Connect(FBdummy[70*i:70*(i+1)], Layer1_ex[:NP],
                 {'rule': 'one_to_one', 'autapses': False, 'multapses': False},
                 {"model": "excitatory-plastic", 'delay': 0.1,
                  "weight": {'distribution': 'normal_clipped', 'mu': g, 'sigma': 0.03,"low":0.01},
                  "Wmax": {'distribution': 'normal', 'mu': STDP_Wmax, 'sigma': 0.03,}})

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
# Run simulation in steps
# -----------------------------
sim_arr = np.arange(0., simtime, sampling)

conn_FF = nest.GetConnections(FFdummy, Layer2_ex[:NP], synapse_model='excitatory-plastic')
conn_FB = nest.GetConnections(FBdummy, Layer1_ex[:NP], synapse_model='excitatory-plastic')

n_conn_FF = len(conn_FF)
n_conn_FB = len(conn_FB)

weight_arr_FF = np.zeros((len(sim_arr), n_conn_FF))
weight_arr_FB = np.zeros((len(sim_arr), n_conn_FB))

with nest.RunManager():
    for i, t in enumerate(sim_arr):
        nest.Run(sampling)
        weight_arr_FF[i, :] = np.array(nest.GetStatus(conn_FF, "weight"))
        weight_arr_FB[i, :] = np.array(nest.GetStatus(conn_FB, "weight"))

# -----------------------------
# Analysis (sorted delays)
# -----------------------------
# FF
conn_FF = nest.GetConnections(Layer1_ex, FFdummy)
all_FF = np.array(nest.GetStatus(conn_FF, ["source", "target", "weight", "delay"]))
idx_sorted_FF = np.argsort(all_FF[:, 1])
sorted_FF = all_FF[idx_sorted_FF]
D_FF = sorted_FF[:, 3]
w_FF = weight_arr_FF[-1, :]
active_idx_FF = np.where(w_FF > g_threshold)[0]
d_FF = D_FF[active_idx_FF % len(D_FF)]

# FB
conn_FB = nest.GetConnections(Layer2_ex, FBdummy)
all_FB = np.array(nest.GetStatus(conn_FB, ["source", "target", "weight", "delay"]))
idx_sorted_FB = np.argsort(all_FB[:, 1])
sorted_FB = all_FB[idx_sorted_FB]
D_FB = sorted_FB[:, 3]
w_FB = weight_arr_FB[-1, :]
active_idx_FB = np.where(w_FB > g_threshold)[0]
d_FB = D_FB[active_idx_FB % len(D_FB)]

# -----------------------------
# Plots
# -----------------------------
plt.figure(1, figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(D_FF, bins=300, color='blue', alpha=0.6, label='Initial')
plt.hist(d_FF, bins=300, color='red', alpha=0.6, label='Final (w>0.15)')
plt.legend(); plt.xlabel('FF delay (ms)'); plt.ylabel('Count')

plt.subplot(1,2,2)
plt.hist(D_FB, bins=300, color='blue', alpha=0.6, label='Initial')
plt.hist(d_FB, bins=300, color='red', alpha=0.6, label='Final (w>0.15)')
plt.legend(); plt.xlabel('FB delay (ms)'); plt.ylabel('Count')

plt.figure(2, figsize=(10,5))
plt.subplot(1,2,1); plt.hist(w_FF, bins=300, color='gray'); plt.xlabel('Final FF weight (nS)'); plt.ylabel('Count')
plt.subplot(1,2,2); plt.hist(w_FB, bins=300, color='gray'); plt.xlabel('Final FB weight (nS)'); plt.ylabel('Count')

plt.figure(3, figsize=(10,5))
plt.subplot(1,2,1); plt.plot(D_FF, w_FF, '.k', markersize=2); plt.xlabel('FF delay (ms)'); plt.ylabel('FF weight (nS)')
plt.subplot(1,2,2); plt.plot(D_FB, w_FB, '.k', markersize=2); plt.xlabel('FB delay (ms)'); plt.ylabel('FB weight (nS)')

plt.figure(4, figsize=(10,5))
n_samples = 50
np.random.seed(42)
selected_FF = np.random.choice(len(D_FF), n_samples, replace=False)
selected_FB = np.random.choice(len(D_FB), n_samples, replace=False)
norm_FF = plt.Normalize(vmin=D_FF.min(), vmax=D_FF.max())
norm_FB = plt.Normalize(vmin=D_FB.min(), vmax=D_FB.max())
cmap = plt.cm.jet

plt.subplot(1,2,1)
for i in selected_FF:
    color = cmap(norm_FF(D_FF[i]))
    plt.plot(sim_arr, weight_arr_FF[:, i], color=color, lw=1)
sm1 = plt.cm.ScalarMappable(norm=norm_FF, cmap=cmap)
sm1.set_array(D_FF)
cbar1 = plt.colorbar(sm1)
cbar1.set_label('FF delay (ms)')
plt.xlabel('Time (ms)'); plt.ylabel('FF weight (nS)')

plt.subplot(1,2,2)
for i in selected_FB:
    color = cmap(norm_FB(D_FB[i]))
    plt.plot(sim_arr, weight_arr_FB[:, i], color=color, lw=1)
sm2 = plt.cm.ScalarMappable(norm=norm_FB, cmap=cmap)
sm2.set_array(D_FB)
cbar2 = plt.colorbar(sm2)
cbar2.set_label('FB delay (ms)')
plt.xlabel('Time (ms)'); plt.ylabel('FB weight (nS)')

plt.tight_layout()
plt.show()
