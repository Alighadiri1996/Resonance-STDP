#!/usr/bin/env bash
# Runs behind Supplementary Figs. S4 and S5.
#
# Three independent single-thread simulations of 1000 s each (p = 0.25, 0.5, 0.75),
# started in parallel. Run inside the NEST 2.20.2 container, see the top-level README.
# Plotting needs matplotlib and seaborn.
#
# The 100 % point (sim11.npz) is the all-to-all run with tau_in,2 = 8 ms from
# ../FigS2_S3_resonance_mismatch (sim3.npz, 2500 s), as in the published figure.
set -euo pipefail
cd "$(dirname "$0")"

SIM=sim_bilayer_stdp_sparse.py
python3 $SIM --prob 0.25 --msd 1234 --out sim8.npz  > sim8.log 2>&1 &
python3 $SIM --prob 0.50 --msd 1234 --out sim9.npz  > sim9.log 2>&1 &
python3 $SIM --prob 0.75 --msd 1234 --out sim10.npz > sim10.log 2>&1 &

if [ -f ../FigS2_S3_resonance_mismatch/sim3.npz ]; then
    cp ../FigS2_S3_resonance_mismatch/sim3.npz sim11.npz
else
    python3 ../FigS2_S3_resonance_mismatch/sim_bilayer_stdp_tau_mismatch.py \
        --tau-in-l2 8 --msd 1234 --out sim11.npz > sim11.log 2>&1 &
fi
wait

python3 plot_S4.py                          # Fig. S4 panels (sim8, sim9, sim10 shown)
python3 plot_S5A_stabilized_connections.py  # Fig. S5A
python3 plot_S5B_weighted_delays.py         # Fig. S5B
