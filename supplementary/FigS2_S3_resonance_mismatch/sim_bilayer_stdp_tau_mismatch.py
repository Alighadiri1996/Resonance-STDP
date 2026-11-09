"""
Bi-layer E-I network with plastic FF and FB connections (STDP), with an adjustable
inhibitory synaptic time constant in layer 2.

Used for Supplementary Figs. S2 and S3 (tau_in,2 = 6 to 12 ms) and for the 100 %
connectivity point of Supplementary Figs. S4 and S5 (tau_in,2 = 8 ms) of:

    Ghadiri A, Taghavi S, Rezaei H, Aertsen A, Kumar A, Valizadeh A (2026).
    Delay selection by spike-timing-dependent plasticity shapes efficient networks
    for signal transmission. J Neurosci 46(32). doi:10.1523/JNEUROSCI.0103-26.2026

Motivation
----------
The main-text simulations use two layers with the same intrinsic resonance frequency.
Changing the inhibitory synaptic time constant of layer 2 (tau_in,2) shifts its
resonance frequency (Fig. S1B). This script tests whether delay selection by STDP
survives a resonance mismatch between the two layers.

What it measures
----------------
Same network and protocol as scripts/AHAAA_2025_Fig4.py: plastic FF and FB
connections, all-to-all between the 70 projection neurons of each layer (4900 FF and
4900 FB synapses, one parrot relay per synapse carrying the axonal delay), FF delays
N(16, 2) ms, FB delays N(20, 8) ms, initial weights N(0.07, 0.03) nS, Wmax N(0.22, 0.03) nS,
pulse packets (50 spikes, sd 2 ms) every 1000 +- 200 ms from 5 s to 2495 s, 2500 s in
total. Only tau_syn_in of layer 2 changes (--tau-in-l2); layer 1 stays at 8 ms.
Saves, per synapse: axonal delay, initial weight, final weight and the weight trace
sampled every 2 s, plus the delays of the synapses whose final weight exceeds 0.15 nS.

Usage
-----
    python sim_bilayer_stdp_tau_mismatch.py --tau-in-l2 9 --msd 1234 --out sim4.npz
    python sim_bilayer_stdp_tau_mismatch.py --tau-in-l2 8 --save-spikes spikes_tau8.npz
    python sim_bilayer_stdp_tau_mismatch.py --simtime 6000 --out smoke.npz   # quick test

run_S2_S3.sh lists the exact runs behind the published panels. Requires NEST 2.20
(nest/nest-simulator:2.20.2) and must run with one thread: the alignment of delays and
weights in the analysis relies on the connection order of a single-threaded kernel.

Changes from the scripts used for the paper (v1/sim1..sim7/sim.py)
-------------------------------------------------------------------
tau_in,2, master seed, simulated time and output names are command-line options
(the seven published copies differed only in these). NumPy's global RNG, which draws
the pulse-time jitter, is now seeded from --msd; the published runs left it unseeded,
so a re-run reproduces the statistics, not the exact realization. Quick-look figures
are optional (--plot). The network, protocol and analysis are unchanged.
"""
import argparse
from time import time

import numpy as np
import nest

# PyNEST 2.20 still uses np.float_, which NumPy 2 removed.
if not hasattr(np, 'float_'):
    np.float_ = np.float64

parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument('--tau-in-l2', type=float, default=8.0,
                    help='tau_syn_in of layer 2 in ms (layer 1 is fixed at 8 ms); default 8')
parser.add_argument('--msd', type=int, default=1234,
                    help='master seed for NEST, the V_m initialization and NumPy; default 1234')
parser.add_argument('--simtime', type=float, default=2500000.0,
                    help='simulated time in ms (>= 2000); default 2500000 (2500 s)')
parser.add_argument('--out', default=None,
                    help='output .npz; default sim_tau<tau>_msd<msd>.npz')
parser.add_argument('--save-spikes', default=None,
                    help='also save all spikes (times, senders) to this .npz')
parser.add_argument('--plot', action='store_true', help='show quick-look figures at the end')
args = parser.parse_args()
save_filename = args.out or f"sim_tau{args.tau_in_l2:g}_msd{args.msd}.npz"

# -----------------------------
# Simulation setup
# -----------------------------
start_time = time()
nest.ResetKernel()
nest.SetKernelStatus({'print_time': True, 'local_num_threads': 1})

msd = args.msd
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd + n_vp)
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
nest.SetKernelStatus({'grng_seed': msd + n_vp, 'rng_seeds': msdrange2})
# Not in the published runs (see docstring): seed NumPy's global RNG with a value
# not used by the NEST or V_m streams above.
np.random.seed(msd + 2 * n_vp + 1)

# -----------------------------
# Parameters
# -----------------------------
a = 50          # spikes per pulse packet
sdev = 2.0      # temporal spread of a pulse packet (ms)
weight = 0.6    # pulse-packet synapse weight (nS)
jitter = 200
pulse_times = np.arange(5000., 2495000., 1000)
pulse_times += np.random.uniform(-jitter, jitter, len(pulse_times))

NE, NI = 200, 50
NP = 70
simtime = args.simtime
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
    'tau_syn_in': args.tau_in_l2,
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
    nest.Connect(pop_ex, pop_ex, {'rule': 'fixed_indegree', 'indegree': CEE, 'autapses': False, "multapses": False}, {'weight': j_exc_exc, 'delay': 1.})
    nest.Connect(pop_ex, pop_in, {'rule': 'fixed_indegree', 'indegree': CEI, 'autapses': False, "multapses": False}, {'weight': j_exc_inh, 'delay': 1.})
    nest.Connect(pop_in, pop_ex, {'rule': 'fixed_indegree', 'indegree': CIE, 'autapses': False, "multapses": False}, {'weight': j_inh_exc, 'delay': 2.})
    nest.Connect(pop_in, pop_in, {'rule': 'fixed_indegree', 'indegree': CII, 'autapses': False, "multapses": False}, {'weight': j_inh_inh, 'delay': 2.})

# The same pulse packet (a spikes) reaches all NP projection neurons of layer 1.
nest.CopyModel('static_synapse_hom_w', 'pp_excitatory', {'weight': weight, 'delay': 0.1})
nest.Connect(pp, Layer1_ex[:NP], {'rule': 'fixed_indegree', 'indegree': 1}, 'pp_excitatory')

# -----------------------------
# Plastic Feedforward (FF)
# Source i -> 70 parrots (axonal delay) -> 70 targets (plastic synapse, 0.1 ms
# dendritic delay). Parrot 70*i + j carries the synapse from source i to target j.
# -----------------------------
nest.CopyModel('stdp_synapse', 'excitatory-plastic')
for i in range(70):
    nest.Connect(Layer1_ex[i:i+1], FFdummy[70*i:70*(i+1)], {'rule': "all_to_all", 'autapses': False, "multapses": False}, {'delay': FF_delay})
    nest.Connect(FFdummy[70*i:70*(i+1)], Layer2_ex[:NP],
                 {'rule': 'one_to_one', 'autapses': False, 'multapses': False},
                 {"model": "excitatory-plastic", 'delay': 0.1,
                  "weight": {'distribution': 'normal_clipped', 'mu': g, 'sigma': 0.03, "low": 0.01},
                  "Wmax": {'distribution': 'normal', 'mu': STDP_Wmax, 'sigma': 0.03}})

# -----------------------------
# Plastic Feedback (FB)
# -----------------------------
for i in range(70):
    nest.Connect(Layer2_ex[i:i+1], FBdummy[70*i:70*(i+1)], {'rule': "all_to_all", 'autapses': False, "multapses": False}, {'delay': FB_delay})
    nest.Connect(FBdummy[70*i:70*(i+1)], Layer1_ex[:NP],
                 {'rule': 'one_to_one', 'autapses': False, 'multapses': False},
                 {"model": "excitatory-plastic", 'delay': 0.1,
                  "weight": {'distribution': 'normal_clipped', 'mu': g, 'sigma': 0.03, "low": 0.01},
                  "Wmax": {'distribution': 'normal', 'mu': STDP_Wmax, 'sigma': 0.03}})

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
# Column k of weight_arr_* is the plastic synapse of parrot k; sorting the
# source -> parrot connections by parrot id aligns each delay with its weight.
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
print(f"Total runtime: {runtime:.2f} seconds")

# -----------------------------
# Data save for post-processing (plot_S2.py, plot_S3A_*.py, plot_S3B_*.py)
# -----------------------------
np.savez(save_filename,
         # Time axis
         sim_arr=sim_arr,

         # Feedforward (FF) Data
         D_FF_initial=D_FF,              # Axonal delay of every FF synapse
         D_FF_final=d_FF,                # Delays of FF synapses with final weight > g_threshold
         w_FF_initial=w_FF_initial,      # Initial Weights
         w_FF_final=w_FF,                # Final Weights at end of sim
         weight_trace_FF=weight_arr_FF,  # Full weight evolution over time [sample, synapse]

         # Feedback (FB) Data
         D_FB_initial=D_FB,              # Axonal delay of every FB synapse
         D_FB_final=d_FB,                # Delays of FB synapses with final weight > g_threshold
         w_FB_initial=w_FB_initial,      # Initial Weights
         w_FB_final=w_FB,                # Final Weights at end of sim
         weight_trace_FB=weight_arr_FB,  # Full weight evolution over time [sample, synapse]

         # Metadata for thresholding
         g_threshold=g_threshold
         )
print(f"--- All data saved successfully to {save_filename} ---")

if args.save_spikes:
    events = nest.GetStatus(SD, "events")[0]
    np.savez_compressed(args.save_spikes,
                        ts=events['times'],
                        gids=events['senders'],
                        simtime=simtime)
    print(f"--- Spike data saved to {args.save_spikes} ---")

# -----------------------------
# Quick-look plots (optional; the published panels come from plot_S2.py)
# -----------------------------
if args.plot:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(1, figsize=(10, 5))
    plt.subplot(2, 1, 1)
    sns.kdeplot(D_FF, color='m', fill=True, label='Initial', bw_adjust=1.)
    sns.kdeplot(d_FF, color='g', fill=True, label='Final', bw_adjust=1.)
    plt.legend(frameon=False)
    plt.yticks([0, 0.25]); plt.xticks([])
    plt.ylim(0, .3); plt.xlim(0, 50)

    plt.subplot(2, 1, 2)
    sns.kdeplot(D_FB, color='blue', fill=True, label='Initial', bw_adjust=1.5)
    sns.kdeplot(d_FB, color='red', fill=True, label='Final', bw_adjust=1.5)
    plt.yticks([0, 0.25]); plt.xticks([10, 20, 30, 40])
    plt.ylim(0, .3); plt.xlim(0, 50)
    plt.legend(frameon=False); plt.xlabel('Delay (ms)'); plt.ylabel('Density')
    plt.tight_layout()

    plt.figure(2, figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.hist(w_FF, bins=300, color='gray'); plt.xlabel('Final FF weight (nS)'); plt.ylabel('Count')
    plt.subplot(1, 2, 2); plt.hist(w_FB, bins=300, color='gray'); plt.xlabel('Final FB weight (nS)'); plt.ylabel('Count')

    plt.figure(3, figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(D_FF, w_FF_initial, '.', color='m', markersize=4, label='FF Initial', alpha=0.5)
    plt.plot(D_FF, w_FF, '.', color='g', markersize=2, label='FF Final')
    plt.xlabel(''); plt.ylabel(r'W$_{FF}$ (nS)')
    plt.xlim(0, 50); plt.ylim(0, 0.3)
    plt.xticks([]); plt.yticks([0, 0.25])
    plt.legend(frameon=False)

    plt.subplot(2, 1, 2)
    plt.plot(D_FB, w_FB_initial, '.', color='b', markersize=4, label='FB Initial', alpha=0.5)
    plt.plot(D_FB, w_FB, '.', color='r', markersize=2, label='FB Final')
    plt.xlabel(''); plt.ylabel(r'W$_{FB}$ (nS)')
    plt.xlim(0, 50); plt.ylim(0, 0.3)
    plt.xticks([10, 20, 30, 40]); plt.yticks([0, 0.25])
    plt.legend(frameon=False)

    plt.figure(4, figsize=(10, 5))
    n_samples = 50
    np.random.seed(42)
    selected_FF = np.random.choice(len(D_FF), n_samples, replace=False)
    selected_FB = np.random.choice(len(D_FB), n_samples, replace=False)
    norm_FF = plt.Normalize(vmin=D_FF.min(), vmax=D_FF.max())
    norm_FB = plt.Normalize(vmin=D_FB.min(), vmax=D_FB.max())
    cmap = plt.cm.jet

    plt.subplot(1, 2, 1)
    for i in selected_FF:
        plt.plot(sim_arr, weight_arr_FF[:, i], color=cmap(norm_FF(D_FF[i])), lw=1)
    sm1 = plt.cm.ScalarMappable(norm=norm_FF, cmap=cmap)
    sm1.set_array(D_FF)
    plt.colorbar(sm1, ax=plt.gca()).set_label('FF delay (ms)')
    plt.xlabel('Time (ms)'); plt.ylabel('FF weight (nS)')

    plt.subplot(1, 2, 2)
    for i in selected_FB:
        plt.plot(sim_arr, weight_arr_FB[:, i], color=cmap(norm_FB(D_FB[i])), lw=1)
    sm2 = plt.cm.ScalarMappable(norm=norm_FB, cmap=cmap)
    sm2.set_array(D_FB)
    plt.colorbar(sm2, ax=plt.gca()).set_label('FB delay (ms)')
    plt.xlabel('Time (ms)'); plt.ylabel('FB weight (nS)')

    plt.tight_layout()
    plt.show()
