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

The code uses the NEST 2 API, so we ran the scripts in Docker with the
`nest/nest-simulator:2.20.2` image.

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
