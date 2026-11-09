#!/usr/bin/env bash
# Runs behind Supplementary Figs. S2 and S3 (and the 100 % point of Figs. S4/S5).
#
# Seven independent single-thread simulations of 2500 s each, started in parallel
# (one core and about 0.3 GB of RAM per run). Run inside the NEST 2.20.2 container,
# see the top-level README. Plotting needs matplotlib and seaborn.
#
# Master seeds as in the published runs: 1234 for all tau_in,2 except 7 ms, which was
# re-run with 12345.
set -euo pipefail
cd "$(dirname "$0")"

SIM=sim_bilayer_stdp_tau_mismatch.py
python3 $SIM --tau-in-l2 6  --msd 1234  --out sim1.npz > sim1.log 2>&1 &
python3 $SIM --tau-in-l2 7  --msd 12345 --out sim2.npz > sim2.log 2>&1 &
python3 $SIM --tau-in-l2 8  --msd 1234  --out sim3.npz > sim3.log 2>&1 &
python3 $SIM --tau-in-l2 9  --msd 1234  --out sim4.npz > sim4.log 2>&1 &
python3 $SIM --tau-in-l2 10 --msd 1234  --out sim5.npz > sim5.log 2>&1 &
python3 $SIM --tau-in-l2 11 --msd 1234  --out sim6.npz > sim6.log 2>&1 &
python3 $SIM --tau-in-l2 12 --msd 1234  --out sim7.npz > sim7.log 2>&1 &
wait

python3 plot_S2.py                          # Fig. S2 panels (sim1, sim4, sim7 shown)
python3 plot_S3A_stabilized_connections.py  # Fig. S3A
python3 plot_S3B_weighted_delays.py         # Fig. S3B
