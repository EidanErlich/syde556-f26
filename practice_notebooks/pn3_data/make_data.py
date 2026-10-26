"""Generates the simulation data plotted in the PN3 ramp practice and Practice Test 3 solutions.

Run from this directory: ../../.venv/bin/python make_data.py
"""

import warnings

import numpy as np
import nengo
from nengo.utils.connection import eval_point_decoding

warnings.filterwarnings("ignore")

TAU = 0.1
TAU_IN = 0.005
DT_OUT = 0.005


def test_input(t):
    if 0.1 <= t < 0.5:
        return 0.5
    if 0.5 <= t < 0.7:
        return -1.0
    return 0.0


def ideal_integral(t):
    if t < 0.1:
        return 0.0
    if t < 0.5:
        return 0.5 * (t - 0.1)
    if t < 0.7:
        return 0.2 - (t - 0.5)
    return 0.0


def run_integrator(neuron_type, seed):
    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node(test_input)
        x = nengo.Ensemble(200, 1, neuron_type=neuron_type)
        nengo.Connection(stim, x, transform=TAU, synapse=TAU_IN)
        nengo.Connection(x, x, transform=1.0, synapse=TAU)
        p = nengo.Probe(x, synapse=0.01)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(1.5)
    step = int(round(DT_OUT / sim.dt))
    return sim.trange()[::step], sim.data[p][::step, 0]


def mismatch_prediction(t_out, dt=1e-4):
    """x = int u + (tau - tau_in) (h_in * u), the exact response with unequal synapses."""
    ts = np.arange(0, 1.5 + dt, dt)
    u = np.array([test_input(ti) for ti in ts])
    u_in = np.zeros_like(u)
    for k in range(1, len(ts)):
        u_in[k] = u_in[k - 1] + dt / TAU_IN * (u[k - 1] - u_in[k - 1])
    x = np.cumsum(u) * dt + (TAU - TAU_IN) * u_in
    return np.interp(t_out, ts, x)


def write_integrator_runs():
    t, spk1 = run_integrator(nengo.LIF(), seed=1)
    _, spk2 = run_integrator(nengo.LIF(), seed=2)
    _, rate = run_integrator(nengo.LIFRate(), seed=1)
    lead = mismatch_prediction(t)
    with open("integrator_test.dat", "w") as f:
        f.write("t u ideal lead spike1 spike2 rate\n")
        for i, ti in enumerate(t):
            f.write(
                f"{ti:.4f} {test_input(ti):.4f} {ideal_integral(ti):.5f} {lead[i]:.5f} "
                f"{spk1[i]:.5f} {spk2[i]:.5f} {rate[i]:.5f}\n"
            )


def write_neural_phase_portrait():
    model = nengo.Network(seed=7)
    with model:
        x = nengo.Ensemble(30, 1, neuron_type=nengo.LIFRate())
        conn = nengo.Connection(x, x, synapse=TAU, solver=nengo.solvers.LstsqNoise(noise=0.1))
    with nengo.Simulator(model, progress_bar=False) as sim:
        grid = np.linspace(-1, 1, 201)[:, None]
        xs, _, xhat = eval_point_decoding(conn, sim, eval_points=grid)
    xs, xhat = xs[:, 0], xhat[:, 0]
    phi = (xhat - xs) / TAU
    with open("neural_phase.dat", "w") as f:
        f.write("x phi\n")
        for a, b in zip(xs, phi):
            f.write(f"{a:.4f} {b:.5f}\n")
    crossings = []
    for i in range(len(xs) - 1):
        if phi[i] == 0 or phi[i] * phi[i + 1] < 0:
            x0 = xs[i] - phi[i] * (xs[i + 1] - xs[i]) / (phi[i + 1] - phi[i])
            kind = "stable" if phi[i + 1] < phi[i] else "unstable"
            crossings.append((round(x0, 3), kind))
    print("neural phase crossings:", crossings)
    print("max |phi|:", np.abs(phi).max())


def write_sine_saturation():
    model = nengo.Network(seed=3)
    with model:
        stim = nengo.Node(lambda t: 3 * np.sin(3 * t))
        x = nengo.Ensemble(400, 1, neuron_type=nengo.LIFRate())
        nengo.Connection(stim, x, transform=TAU, synapse=TAU)
        nengo.Connection(x, x, synapse=TAU)
        p = nengo.Probe(x, synapse=0.01)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(4.0)
    with open("sine_saturation.dat", "w") as f:
        f.write("t x\n")
        for a, b in zip(sim.trange()[::10], sim.data[p][::10, 0]):
            f.write(f"{a:.3f} {b:.4f}\n")


if __name__ == "__main__":
    write_integrator_runs()
    write_neural_phase_portrait()
    write_sine_saturation()
