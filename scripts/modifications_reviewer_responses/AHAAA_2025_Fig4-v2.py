# ==============================================
# Two-layer spiking network with plastic FF & FB (STDP) 
# instead of all to all projection neurons connections, here we consider a connection probability for both FF and FB projections.
# ==============================================

import nest
import numpy as np
import matplotlib.pyplot as plt
import nest.raster_plot
from time import time
# -----------------------------
# Simulation setup
# -----------------------------
start_time = time()
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
simtime = 1000000.0
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
exci_params_l1 = {
    'V_th': -54.,
    'V_reset': -70.,
    'tau_syn_ex': 3.0,
    'tau_syn_in': 8.0,
    'tau_minus': 19.3,
    'E_in': -80.0,
    'g_L': 16.67,
    'C_m': 250.0
}
inhi_params_l1 = exci_params_l1.copy()

exci_params_l2 = {
    'V_th': -54.,
    'V_reset': -70.,
    'tau_syn_ex': 3.0,
    'tau_syn_in': 8.0,
    'tau_minus': 19.3,
    'E_in': -80.0,
    'g_L': 16.67,
    'C_m': 250.0
}
inhi_params_l2 = exci_params_l2.copy()

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
Layer1_ex = nest.Create("iaf_cond_exp", NE, params=exci_params_l1)
Layer1_in = nest.Create("iaf_cond_exp", NI, params=inhi_params_l1)
Layer2_ex = nest.Create("iaf_cond_exp", NE, params=exci_params_l2)
Layer2_in = nest.Create("iaf_cond_exp", NI, params=inhi_params_l2)

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
import numpy as np

# -----------------------------
# Plastic Feedforward (FF) - True 50% Sparsity with Unique Delays
# -----------------------------
nest.CopyModel('stdp_synapse', 'excitatory-plastic')

# Pre-define our parameters
N_src = 70
N_tgt = 70
prob = 0.25
k_connections = int(N_tgt * prob)  # 35 targets per source neuron

# We use a running counter to pick a unique parrot relay for EVERY synapse
parrot_idx = 0 

for i in range(N_src):
    # 1. For source 'i', randomly pick 35 target indices out of the 70 available
    # replace=False guarantees no multapses (no duplicate connections)
    chosen_targets = np.random.choice(np.arange(N_tgt), size=k_connections, replace=False)
    
    for j in chosen_targets:
        # 2. Generate a unique delay for this specific synapse
        # We use rejection sampling to prevent negative delays
        delay = -1.0
        while delay < 0.1:
            delay = np.random.normal(loc=16.0, scale=2.0)
            
        # 3. Source connects to a UNIQUE Parrot with the unique delay
        nest.Connect(Layer1_ex[i:i+1], FFdummy[parrot_idx : parrot_idx+1], 
                     syn_spec={'delay': delay})
        
        # 4. That unique Parrot connects to the chosen Layer 2 target with Plastic Weight
        # Note: We scale the weight 'g' and 'STDP_Wmax' by 2.0 to compensate for 50% sparsity!
        nest.Connect(FFdummy[parrot_idx : parrot_idx+1], Layer2_ex[j:j+1],
                     syn_spec={"model": "excitatory-plastic", 
                               "delay": 0.1,
                               "weight": {'distribution': 'normal_clipped', 'mu': g * 4.0, 'sigma': 0.03, "low": 0.01},
                               "Wmax": {'distribution': 'normal', 'mu': STDP_Wmax * 4.0, 'sigma': 0.03}})
        
        # Move to the next parrot neuron in our pool of 4900
        parrot_idx += 1

# -----------------------------
# Plastic Feedback (FB) - True 50% Sparsity with Unique Delays
# -----------------------------
# We continue using parrot_idx from where the FF loop left off 
# (Ensure your FBdummy pool is large enough, e.g., 4900 total)
parrot_idx_fb = 0 

# Parameters for FB (matching your FF logic)
N_src_fb = 70  # Layer 2 neurons
N_tgt_fb = 70  # Layer 1 neurons
prob = 0.25
k_connections_fb = int(N_tgt_fb * prob)  # 35 targets per source neuron


for i in range(N_src_fb):
    # 1. For each L2 neuron 'i', randomly pick 35 L1 targets
    chosen_targets_fb = np.random.choice(np.arange(N_tgt_fb), size=k_connections_fb, replace=False)
    
    for j in chosen_targets_fb:
        # 2. Generate a unique delay for this FB synapse (Mean=20, Sigma=8)
        delay_fb = -1.0
        while delay_fb < 0.1:
            delay_fb = np.random.normal(loc=20.0, scale=8.0)
            
        # 3. Source (L2) connects to a UNIQUE FB-Parrot with the unique delay
        nest.Connect(Layer2_ex[i:i+1], FBdummy[parrot_idx_fb : parrot_idx_fb+1], 
                     syn_spec={'delay': delay_fb})
        
        # 4. FB-Parrot connects to the L1 target with Plastic Weight
        # Corrected syn_spec to avoid the DictError (Wmax at top level)
        nest.Connect(FBdummy[parrot_idx_fb : parrot_idx_fb+1], Layer1_ex[j:j+1],
                     syn_spec={
                         "model": "excitatory-plastic", 
                         "delay": 0.1,
                         "weight": {'distribution': 'normal_clipped', 
                                    'mu': g * 4.0, 'sigma': 0.03, "low": 0.01},
                         "Wmax": STDP_Wmax * 4.0
                     })
        
        parrot_idx_fb += 1
# --- Capture Initial Weights ---
conn_FF_init = nest.GetConnections(FFdummy, Layer2_ex[:NP], synapse_model='excitatory-plastic')
conn_FB_init = nest.GetConnections(FBdummy, Layer1_ex[:NP], synapse_model='excitatory-plastic')
w_FF_initial = np.array(nest.GetStatus(conn_FF_init, "weight"))
w_FB_initial = np.array(nest.GetStatus(conn_FB_init, "weight"))

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

runtime = time() - start_time
print(f"*********  Total runtime: {runtime:.2f} seconds")

# -----------------------------
# Data Save
# -----------------------------
save_filename = "sim8.npz"  # Change this to sim2.npz, sim3.npz, etc., for each run

np.savez(save_filename, 
         # Time axis
         sim_arr=sim_arr, 
         
         # Feedforward (FF) Data
         D_FF_initial=D_FF,              # The full pool of sorted initial delays
         D_FF_final=d_FF,                # The delays of synapses that passed the threshold
         w_FF_initial=w_FF_initial,      # Weights at t=0
         w_FF_final=w_FF,                # Weights at t=simtime
         weight_trace_FF=weight_arr_FF,  # Full 2D array [time, synapse]
         
         # Feedback (FB) Data
         D_FB_initial=D_FB,              # The full pool of sorted initial delays
         D_FB_final=d_FB,                # The delays of synapses that passed the threshold
         w_FB_initial=w_FB_initial,      # Weights at t=0
         w_FB_final=w_FB,                # Weights at t=simtime
         weight_trace_FB=weight_arr_FB,  # Full 2D array [time, synapse]
         
         # Metadata
         g_threshold=g_threshold
)

print(f"--- Data saved successfully to {save_filename} ---")

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
plt.subplot(1,2,1) 
plt.plot(D_FF, w_FF_initial, '.', color='m', markersize=4, label='Initial', alpha=0.5)
plt.plot(D_FF, w_FF, '.',color='g', markersize=2); plt.xlabel('FF delay (ms)'); plt.ylabel('FF weight (nS)')

plt.subplot(1,2,2)
plt.plot(D_FB, w_FB_initial, '.', color='b', markersize=4, label='Initial', alpha=0.5)
plt.plot(D_FB, w_FB, '.',color='r', markersize=2); plt.xlabel('FB delay (ms)'); plt.ylabel('FB weight (nS)')

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
# cbar1 = plt.colorbar(sm1)
cbar1 = plt.colorbar(sm1, ax=plt.gca()) #add by saeed
cbar1.set_label('FF delay (ms)')
plt.xlabel('Time (ms)'); plt.ylabel('FF weight (nS)')

plt.subplot(1,2,2)
for i in selected_FB:
    color = cmap(norm_FB(D_FB[i]))
    plt.plot(sim_arr, weight_arr_FB[:, i], color=color, lw=1)
sm2 = plt.cm.ScalarMappable(norm=norm_FB, cmap=cmap)
sm2.set_array(D_FB)
# cbar2 = plt.colorbar(sm2)
cbar2 = plt.colorbar(sm2, ax=plt.gca()) # add by saeed
cbar2.set_label('FB delay (ms)')
plt.xlabel('Time (ms)'); plt.ylabel('FB weight (nS)')

plt.tight_layout()
plt.show()
