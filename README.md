# Delay selection by spike-timing-dependent plasticity shapes efficient networks for signal transmission

**Abstract:**

Information processing in the brain relies on efficient communication between different brain regions. Brain oscillations can control signal transmission in brain networks by modulating the timing and excitability of sender and receiver areas. For effective transmission, signals should arrive at target areas when their excitability is maximized. For reciprocally connected neural populations, this mechanism works if the transmission delay matches the period of their evoked oscillation. However, the mechanisms underlying such development of the connections with matched delays remain elusive.

While transmission delays in brain networks change during development, the process by which delays are tuned for efficient transmission is unknown. Here, we demonstrate that the well-known Hebbian learning rule can provide a mechanism for selecting connections with delays that match the period of network oscillations. We consider a reciprocally connected bi-layer network of excitatory and inhibitory neurons that generate network-level oscillations spontaneously or in response to external stimuli. When exposed to spike-timing-dependent plasticity (STDP), the network self-organizes to potentiate connections with delays matching the oscillation period, while depressing those with non-matching delays. Our findings shed light on how transmission delays may evolve during learning and development to optimize the organization of brain networks for efficient signal transmission.

**Usage:**

The Python script `AHAAA_2025_Fig{1,2,3,4,6}_Plot.py` can be used to reproduce the key results presented in Figures 1, 2, 3, 4, and 6 of the manuscript.

**Requirements:**

- Python 2.7.17
- NEST Simulator (v2.16.0)
- NumPy
- pylab
- matplotlib


## Running with Docker (Recommended)
To ensure a consistent environment with the correct versions of NEST (v2.20.2 or v2.16.0) and Python, it is recommended to run the simulations using a Docker container. This avoids local installation conflicts.

```bash
# Allow container to access host display (for GUI/plotting support)
xhost +local:docker

# Launch the container
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)":/data \
    -w /data \
    nest/nest-simulator:2.20.2 \
    bash
```
Inside the running container's `bash` session, you can then execute simulations (`python your_simulation_script.py`) and manage the environment (`pip install seaborn`).

## Parallel Execution Strategy

To optimize computational throughput, this project uses Task-Level Parallelism rather than NEST's internal multithreading.
`local_num_threads` is set to 1 inside the NEST kernel.
Multiple independent simulations are run across CPU cores using a Python `multiprocessing.Pool`.
