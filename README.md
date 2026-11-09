# Delay selection by spike-timing-dependent plasticity shapes efficient networks for signal transmission

Simulation and analysis code for

> Ghadiri A, Taghavi S, Rezaei H, Aertsen A, Kumar A, Valizadeh A (2026).
> Delay selection by spike-timing-dependent plasticity shapes efficient networks for signal transmission.
> *Journal of Neuroscience* 46(32): e0103262026. https://doi.org/10.1523/JNEUROSCI.0103-26.2026

**Abstract:**

Information processing in the brain relies on efficient communication between different brain regions. Brain oscillations can control signal transmission in brain networks by modulating the timing and excitability of sender and receiver areas. For effective transmission, signals should arrive at target areas when their excitability is maximized. For reciprocally connected neural populations, this mechanism works if the transmission delay matches the period of their evoked oscillation. However, the mechanisms underlying such development of the connections with matched delays remain elusive.

While transmission delays in brain networks change during development, the process by which delays are tuned for efficient transmission is unknown. Here, we demonstrate that the well-known Hebbian learning rule can provide a mechanism for selecting connections with delays that match the period of network oscillations. We consider a reciprocally connected bi-layer network of excitatory and inhibitory neurons that generate network-level oscillations spontaneously or in response to external stimuli. When exposed to spike-timing-dependent plasticity (STDP), the network self-organizes to potentiate connections with delays matching the oscillation period, while depressing those with non-matching delays. Our findings shed light on how transmission delays may evolve during learning and development to optimize the organization of brain networks for efficient signal transmission.

## Repository layout

```
scripts/                                 main-text figures (one configuration per script)
  AHAAA_2025_Fig1.py                     Fig. 1: SNR of pulse-packet propagation vs FF and FB delays (static connections)
  AHAAA_2025_Fig2.py                     Fig. 2: plastic FF connections only, 500 s
  AHAAA_2025_Fig3.py                     Fig. 3: static FF and plastic FB connections, 2000 s
  AHAAA_2025_Fig4.py                     Fig. 4: plastic FF and FB connections, 2500 s
  AHAAA_2025_Fig6.py                     Fig. 6: plastic FF and FB connections in the oscillatory regime
supplementary/
  FigS1_resonance_frequency/             Fig. S1: resonance of a single layer network
  FigS2_S3_resonance_mismatch/           Figs. S2, S3: layers with different resonance frequencies
  FigS4_S5_connection_probability/       Figs. S4, S5: sparser initial inter-layer connectivity
```

Every supplementary script starts with a docstring that states what it simulates or plots, the
parameters, and how to run it. Each supplementary folder also contains its figures as published,
cropped from the supplementary PDF without captions, to compare against re-runs:
[Fig. S1](supplementary/FigS1_resonance_frequency/FigS1.png),
[Fig. S2](supplementary/FigS2_S3_resonance_mismatch/FigS2.png),
[Fig. S3](supplementary/FigS2_S3_resonance_mismatch/FigS3.png),
[Fig. S4](supplementary/FigS4_S5_connection_probability/FigS4.png),
[Fig. S5](supplementary/FigS4_S5_connection_probability/FigS5.png).
In the published Fig. S5A the legend reads "Average FF/FB delays"; the plot script labels the
curves "Feedforward (FF)" and "Feedback (FB)".

## Requirements

The code uses the NEST 2 API (`spike_detector`, `grng_seed`/`rng_seeds`, distribution dictionaries)
and runs with **NEST 2.20.2**. The simplest setup is the official Docker image, which provides
NEST 2.20.2, Python 3.10, NumPy 2.2 and SciPy. Plotting also needs matplotlib and seaborn:

```bash
docker run --rm -it -v "$(pwd)":/data -w /data nest/nest-simulator:2.20.2 bash
pip install matplotlib seaborn      # inside the container
```

PyNEST 2.20 still refers to `np.float_`, which NumPy 2 removed; the supplementary scripts restore
this alias at the top, so no NumPy downgrade is needed (for the main-text scripts, see
[Main-text figures](#main-text-figures)). All simulations run on a single
thread (`local_num_threads = 1`). This is required: the analysis matches each synapse's delay to
its weight trace using the connection order of a single-threaded kernel. Independent runs (for
example the seven conditions of Figs. S2/S3) are started as separate processes; the `run_*.sh`
scripts do this.

## Supplementary figures

| Figure | Simulation | Plot | Cost |
|---|---|---|---|
| S1A | `FigS1_resonance_frequency/S1A_single_network_response.py` | `S1A_plot.py` | seconds |
| S1B | `FigS1_resonance_frequency/S1B_resonance_vs_tau_in.py` | `S1B_plot.py` | 7 x 101 runs of 1 s, about 10 min |
| S2, S3 | `FigS2_S3_resonance_mismatch/run_S2_S3.sh` | `plot_S2.py`, `plot_S3A_stabilized_connections.py`, `plot_S3B_weighted_delays.py` | 7 runs of 2500 s, about 2 to 3 h each |
| S4, S5 | `FigS4_S5_connection_probability/run_S4_S5.sh` | `plot_S4.py`, `plot_S5A_stabilized_connections.py`, `plot_S5B_weighted_delays.py` | 3 runs of 1000 s, about 1 h each, plus the 100 % run from S2/S3 |

Run times are per run on one core of a desktop CPU; each run of Figs. S2 to S5 writes 10 to 100 MB.
The published results of Fig. S1 are included in `supplementary/FigS1_resonance_frequency/data/`,
so `S1A_plot.py` and `S1B_plot.py` work without running the simulations:

```bash
cd supplementary/FigS1_resonance_frequency
python3 S1A_plot.py && python3 S1B_plot.py
```

For Figs. S2 to S5, run the shell script of the folder inside the container; it starts the
simulations in parallel and then makes the plots:

```bash
bash supplementary/FigS2_S3_resonance_mismatch/run_S2_S3.sh
bash supplementary/FigS4_S5_connection_probability/run_S4_S5.sh   # after S2/S3, reuses sim3.npz
```

### Runs behind the published panels

| Panel | Condition | Script and options | Output |
|---|---|---|---|
| S2, S3 | tau_in,2 = 6, 7, 8, 9, 10, 11, 12 ms (layer 1: 8 ms) | `sim_bilayer_stdp_tau_mismatch.py --tau-in-l2 <tau> --msd <seed>`, seed 12345 for 7 ms and 1234 otherwise, 2500 s | `sim1.npz` ... `sim7.npz` |
| S4, S5 | p = 0.25, 0.5, 0.75 | `sim_bilayer_stdp_sparse.py --prob <p> --msd 1234`, weights and Wmax x 4, 2, 1.33, 1000 s | `sim8.npz`, `sim9.npz`, `sim10.npz` |
| S4, S5 | p = 1 (all-to-all) | the tau_in,2 = 8 ms run of Figs. S2/S3, 2500 s | `sim11.npz` (copy of `sim3.npz`) |
| S1A | tau_in = 9 ms, one pulse packet | `S1A_single_network_response.py` (master seed 46521) | `resonance_results.npz` |
| S1B | tau_in = 6 to 12 ms, 101 trials each | `S1B_resonance_vs_tau_in.py` (trial seed 4652 + trial + 100 tau_in) | `complete_resonance_metadata.npz` |

### Weight thresholds used in the analysis

A synapse counts as potentiated when its final weight reaches a threshold. The thresholds differ
between panels and are set in the plot scripts:

| Panel | Quantity | Threshold |
|---|---|---|
| S2; S4 lower rows | "Final" delay distributions (`D_*_final` saved by the simulation) | final weight > 0.15 nS for every condition |
| S3A | % stabilized connections | >= 0.15 nS |
| S3B | weight-averaged delays | none: all synapses, weighted by final weight (`--threshold 0.15` restricts the average) |
| S5A | % stabilized connections | scaled initial mean weight: 0.28, 0.14, 0.093, 0.07 nS for p = 0.25, 0.5, 0.75, 1 |
| S5B | weight-averaged delays | 0.15 nS scaled like the weights: 0.6, 0.3, 0.2, 0.15 nS |

### Reproducibility

- Fig. S1 is fully seeded: a re-run of `S1A_single_network_response.py` reproduces
  `data/resonance_results.npz` exactly, and the first trials of `S1B_resonance_vs_tau_in.py`
  reproduce the published per-trial values.
- The published runs of Figs. S2 to S5 seeded NEST and the membrane-potential initialization, but
  not NumPy's global random number generator, which draws the pulse-time jitter and, in the sparse
  runs, the inter-layer targets and delays. The scripts here seed it from `--msd`, so a re-run is
  repeatable but reproduces the published results statistically, not spike by spike. With the
  published seeds, the Fig. S2/S3 script rebuilds the published networks exactly (all FF and FB
  delays and initial weights); the sparse script reproduces the initial weights but draws
  different targets and delays.
- The plot scripts reproduce the values of Figs. S1B, S3 and S5 from the original simulation
  outputs.

## Main-text figures

The scripts in `scripts/` are the original simulation scripts of the main-text figures, kept as
first released: one parameter set per script (`AHAAA_2025_Fig1.py` sweeps the FF and FB delays),
each simulating and then plotting. To run them in the `nest/nest-simulator:2.20.2` image (NumPy 2.2,
and a current Matplotlib after `pip install matplotlib`), make two small edits first:

1. Right after `import numpy as np`, restore the alias that PyNEST 2.20 needs. Without it, the
   scripts stop with `AttributeError: np.float_ was removed in the NumPy 2.0 release` as soon as
   PyNEST returns numeric arrays (for example from `GetStatus`):

   ```python
   if not hasattr(np, 'float_'):
       np.float_ = np.float64
   ```

2. In `AHAAA_2025_Fig4.py` and `AHAAA_2025_Fig6.py`, pass the Axes to the two colorbars, which
   current Matplotlib (tested with 3.10) otherwise rejects with a `ValueError`:
   `plt.colorbar(sm1, ax=plt.gca())` and `plt.colorbar(sm2, ax=plt.gca())`.

Then run them like the other scripts, for example `python3 scripts/AHAAA_2025_Fig4.py` in the
container.

## Citation

```bibtex
@article{Ghadiri2026DelaySelection,
  author  = {Ghadiri, Ali and Taghavi, Saeed and Rezaei, Hedyeh and Aertsen, Ad and Kumar, Arvind and Valizadeh, Alireza},
  title   = {Delay Selection by Spike-Timing-Dependent Plasticity Shapes Efficient Networks for Signal Transmission},
  journal = {Journal of Neuroscience},
  year    = {2026},
  volume  = {46},
  number  = {32},
  pages   = {e0103262026},
  doi     = {10.1523/JNEUROSCI.0103-26.2026}
}
```
