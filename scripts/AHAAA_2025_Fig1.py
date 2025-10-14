
from __future__ import division, print_function
import nest
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Seeds & pyrngs
# -----------------------
msd = 4652
n_vp = nest.GetKernelStatus()['total_num_virtual_procs']
msdrange1 = range(msd, msd + n_vp)
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
nest.SetKernelStatus({'grng_seed': msd + n_vp, 'rng_seeds': list(msdrange2)})

# -----------------------
# Network / sim parameters
# -----------------------
bin_w = 5.0
NP = 70
NE, NI = 200, 50

p_rate_ex = 6700.0
p_rate_in = 4000.0

j_exc_exc = 0.33
j_exc_inh = 5.5
j_inh_exc = -6.2
j_inh_inh = -15.0

epsilonEE = 0.2
epsilonIE = 0.2
epsilonEI = 0.2
epsilonII = 0.2
epsilonP  = 0.5

CEE = int(epsilonEE * NE)
CIE = int(epsilonIE * NI)
CEI = int(epsilonEI * NE)
CII = int(epsilonII * NI)
CEP = int(epsilonP * NP)

w_fw = 0.3
w_fb = 0.15

in_delay_ex = 1.0
in_delay_in = 2.0

delays_arr_FW = np.arange(5.0, 41.0, 1.0)
delays_arr_FB = np.arange(5.0, 41.0, 1.0)

N_trials = 4
simtime = 1000.0
pulse_time = 500.0
T_res = 40.0
n_cycles = 5

SNR_arr = np.zeros((len(delays_arr_FW), len(delays_arr_FB)))

# -----------------------
# Main loops
# -----------------------
for ii, d_FB in enumerate(delays_arr_FB):
    for jj, d_FW in enumerate(delays_arr_FW):
        snr_trials = []

        for trial in range(N_trials):
            nest.ResetKernel()

            grng = msd + n_vp + trial
            rngs_trial = [s + trial for s in msdrange2]
            nest.SetKernelStatus({
                'overwrite_files': True,
                'print_time': True,
                'local_num_threads': 1,
                'grng_seed': grng,
                'rng_seeds': rngs_trial
            })

            exc_params = {'V_th': -54., 'V_reset': -70., 'tau_syn_ex': 3., 'tau_syn_in': 8.,
                          'g_L': 16.67, 'C_m': 250.0,"E_in":-80.0}
            inh_params = exc_params.copy()

            pop1_ex = nest.Create("iaf_cond_exp", NE, params=exc_params)
            pop1_in = nest.Create("iaf_cond_exp", NI, params=inh_params)
            pop2_ex = nest.Create("iaf_cond_exp", NE, params=exc_params)
            pop2_in = nest.Create("iaf_cond_exp", NI, params=inh_params)
            nodes = pop1_ex + pop1_in + pop2_ex + pop2_in

            node_info = nest.GetStatus(nodes)
            local_nodes = [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]
            for gid, vp in local_nodes:
                nest.SetStatus([gid], {'V_m': pyrngs[vp].uniform(-75.0, -65.0)})

            pulse = nest.Create('pulsepacket_generator',
                                params={'activity': 50, 'sdev': 2.0, 'pulse_times': [pulse_time]})
            noise = nest.Create('poisson_generator', 2,
                                [{"rate": p_rate_ex}, {"rate": p_rate_in}])

            nest.CopyModel('static_synapse_hom_w', 'NOISE_EXCI_SYN', {'weight': 0.25, 'delay': 0.1})
            nest.CopyModel('static_synapse_hom_w', 'NOISE_INHI_SYN', {'weight': 0.4, 'delay': 0.1})
            nest.Connect(noise[:1], pop1_ex + pop2_ex, syn_spec='NOISE_EXCI_SYN')
            nest.Connect(noise[1:], pop1_in + pop2_in, syn_spec='NOISE_INHI_SYN')

            nest.CopyModel('static_synapse_hom_w', 'pp_excitatory', {'weight': 0.6, 'delay': 0.1})
            nest.Connect(pulse, pop1_ex[:NP], {'rule': 'fixed_indegree', 'indegree': 1}, 'pp_excitatory')

            nest.Connect(pop1_ex, pop1_ex, {'rule': 'fixed_indegree', 'indegree': CEE,'autapses': False,"multapses":False},
                         {'weight': j_exc_exc, 'delay': in_delay_ex})
            nest.Connect(pop1_ex, pop1_in, {'rule': 'fixed_indegree', 'indegree': CEI,'autapses': False,"multapses":False},
                         {'weight': j_exc_inh, 'delay': in_delay_ex})
            nest.Connect(pop1_in, pop1_ex, {'rule': 'fixed_indegree', 'indegree': CIE,'autapses': False,"multapses":False},
                         {'weight': j_inh_exc, 'delay': in_delay_in})
            nest.Connect(pop1_in, pop1_in, {'rule': 'fixed_indegree', 'indegree': CII,'autapses': False,"multapses":False},
                         {'weight': j_inh_inh, 'delay': in_delay_in})

            nest.Connect(pop2_ex, pop2_ex, {'rule': 'fixed_indegree', 'indegree': CEE,'autapses': False,"multapses":False},
                         {'weight': j_exc_exc, 'delay': in_delay_ex})
            nest.Connect(pop2_ex, pop2_in, {'rule': 'fixed_indegree', 'indegree': CEI,'autapses': False,"multapses":False},
                         {'weight': j_exc_inh, 'delay': in_delay_ex})
            nest.Connect(pop2_in, pop2_ex, {'rule': 'fixed_indegree', 'indegree': CIE,'autapses': False,"multapses":False},
                         {'weight': j_inh_exc, 'delay': in_delay_in})
            nest.Connect(pop2_in, pop2_in, {'rule': 'fixed_indegree', 'indegree': CII,'autapses': False,"multapses":False},
                         {'weight': j_inh_inh, 'delay': in_delay_in})

            nest.Connect(pop1_ex[:NP], pop2_ex[:NP], {'rule': 'fixed_indegree', 'indegree': CEP,'autapses': False,"multapses":False},
                         {'delay': d_FW, 'weight': w_fw})
            nest.Connect(pop2_ex[:NP], pop1_ex[:NP], {'rule': 'fixed_indegree', 'indegree': CEP,'autapses': False,"multapses":False},
                         {'delay': d_FB, 'weight': w_fb})

            SD = nest.Create('spike_detector')
            nest.Connect(pop2_ex, SD)

            nest.Simulate(simtime)

            events = nest.GetStatus(SD, keys='events')[0]
            ts = np.array(events['times']) if 'times' in events else np.array([])

            t_noise_start = 100.0
            t_noise_end = pulse_time + d_FW
            t_sig_start = t_noise_end
            t_sig_end = t_sig_start + n_cycles * T_res
            if t_sig_end > simtime:
                t_sig_end = simtime - 1e-6

            bins_noise = np.arange(t_noise_start, t_noise_end + 1e-9, bin_w)
            bins_signal = np.arange(t_sig_start, t_sig_end + 1e-9, bin_w)

            hist_noise, _ = np.histogram(ts[(ts >= t_noise_start) & (ts < t_noise_end)], bins=bins_noise)
            hist_signal, _ = np.histogram(ts[(ts >= t_sig_start) & (ts < t_sig_end)], bins=bins_signal)

            # --- Compute SNR 
            var_signal = np.var(hist_signal)
            var_noise = np.var(hist_noise)
            snr_val = var_signal / var_noise if var_noise > 0 else 0.0

            snr_trials.append(snr_val)

        SNR_arr[jj, ii] = np.mean(snr_trials)
        print("FW={:.1f} FB={:.1f} -> SNR={:.4f}".format(d_FW, d_FB, SNR_arr[jj, ii]))

# -----------------------
# Plot result
# -----------------------
X, Y = np.meshgrid(delays_arr_FB, delays_arr_FW)
plt.figure(figsize=(8, 6))
plt.pcolor(X, Y, SNR_arr, cmap='viridis')
plt.xlabel("FB delays (ms)")
plt.ylabel("FW delays (ms)")
plt.colorbar(label="SNR L 2")
plt.gca().set_aspect('equal', adjustable='box')

np.savetxt("SNR_arr",SNR_arr)

plt.show()
