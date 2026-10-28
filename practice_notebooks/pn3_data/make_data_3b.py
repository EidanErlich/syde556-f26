"""Simulation data for Practice Test 3B solutions.

Run from this directory: ../../.venv/bin/python make_data_3b.py
"""

import warnings

import numpy as np
import nengo

warnings.filterwarnings("ignore")


def write(name, header, cols):
    with open(name, "w") as f:
        f.write(header + "\n")
        for row in zip(*cols):
            f.write(" ".join(f"{v:.5f}" for v in row) + "\n")


def probe_synapses():
    model = nengo.Network(seed=31)
    with model:
        stim = nengo.Node(lambda t: np.sin(4 * np.pi * t))
        ens = nengo.Ensemble(100, 1)
        nengo.Connection(stim, ens)
        p1 = nengo.Probe(ens, synapse=0.001)
        p10 = nengo.Probe(ens, synapse=0.01)
        p100 = nengo.Probe(ens, synapse=0.1)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(1.0)
    t = sim.trange()[::2]
    x = np.sin(4 * np.pi * t)
    write("q1b_probes.dat", "t x p1 p10 p100",
          [t, x, sim.data[p1][::2, 0], sim.data[p10][::2, 0], sim.data[p100][::2, 0]])


def oscillator():
    tau, omega = 0.1, 2 * np.pi
    model = nengo.Network(seed=32)
    with model:
        kick = nengo.Node(lambda t: [0, 20] if t < 0.05 else [0, 0])
        x = nengo.Ensemble(800, 2)
        nengo.Connection(kick, x, transform=tau, synapse=tau)
        nengo.Connection(x, x, transform=[[1, -tau * omega], [tau * omega, 1]], synapse=tau)
        p = nengo.Probe(x, synapse=0.01)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(3.0)
    t = sim.trange()[::5]
    write("q4_oscillator.dat", "t x1 x2", [t, sim.data[p][::5, 0], sim.data[p][::5, 1]])


def ramp_integrator():
    tau, tau_in = 0.1, 0.01
    u = lambda t: 4 * t if t < 0.5 else 0.0

    def run(neuron_type):
        model = nengo.Network(seed=33)
        with model:
            stim = nengo.Node(u)
            x = nengo.Ensemble(200, 1, neuron_type=neuron_type)
            nengo.Connection(stim, x, transform=tau, synapse=tau_in)
            nengo.Connection(x, x, synapse=tau)
            p = nengo.Probe(x, synapse=0.01)
        with nengo.Simulator(model, progress_bar=False) as sim:
            sim.run(1.0)
        return sim.trange(), sim.data[p][:, 0]

    t, spk = run(nengo.LIF())
    _, rate = run(nengo.LIFRate())
    dt = t[1] - t[0]
    uu = np.array([u(ti) for ti in t])
    u_in = np.zeros_like(uu)
    for k in range(1, len(t)):
        u_in[k] = u_in[k - 1] + dt / tau_in * (uu[k - 1] - u_in[k - 1])
    ideal = np.where(t < 0.5, 2 * t**2, 0.5)
    lead = np.cumsum(uu) * dt + (tau - tau_in) * u_in
    s = slice(None, None, 4)
    write("q5_ramp.dat", "t u ideal lead spike rate",
          [t[s], uu[s], ideal[s], lead[s], spk[s], rate[s]])


if __name__ == "__main__":
    probe_synapses()
    oscillator()
    ramp_integrator()
