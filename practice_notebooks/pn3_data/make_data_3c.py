"""Simulation data for Practice Test 3C solutions.

Run from this directory: ../../.venv/bin/python make_data_3c.py
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


def jmax(amax, tr=0.002, tc=0.02):
    return 1.0 / (1.0 - np.exp((tr - 1.0 / amax) / tc))


def rate(xi, xi0, amax, tr=0.002, tc=0.02):
    xi = np.asarray(xi, dtype=float)
    frac = (xi - xi0) / (1.0 - xi0)
    jm = jmax(amax, tr, tc)
    J = 1.0 + (jm - 1.0) * frac
    out = np.zeros_like(xi)
    on = J > 1.0
    out[on] = 1.0 / (tr - tc * np.log(1.0 - 1.0 / J[on]))
    return out


def tuning_curves():
    x = np.linspace(-2, 2, 401)
    r1 = rate(x, 0.25, 100)          # radius 1, positive encoder, xi = x
    r2 = rate(x / 2, 0.25, 100)      # radius 2, xi = x/2
    write("q1_radius.dat", "x r1 r2", [x, r1, r2])

    s = np.linspace(-3, 3, 401)
    # e = (-1, 0), radius 3: xi = -s/3, intercept 1/3, 90 Hz
    neg = rate(-s / 3, 1.0 / 3.0, 90)
    write("q1_encoder.dat", "s rate", [s, neg])
    print("radius1 at x=1,2", rate(1, 0.25, 100), rate(2, 0.25, 100))
    print("radius2 at x=0.5,2", rate(0.5 / 2, 0.25, 100), rate(1, 0.25, 100))
    print("encoder at s=-1,-3", rate(1 / 3, 1 / 3, 90), rate(1, 1 / 3, 90))


def cubic_phase(tau=0.1, radius=2.0):
    x = np.linspace(-2, 2, 401)
    phi = x**3 - x
    write("q3_phi.dat", "x phi", [x, phi])

    def roll(x0, t_end=1.5):
        dt = 1e-4
        t = np.arange(0, t_end + dt, dt)
        xs = np.empty_like(t)
        xs[0] = x0
        for i in range(1, len(t)):
            dx = (xs[i - 1] ** 3 - xs[i - 1]) / tau
            xs[i] = np.clip(xs[i - 1] + dt * dx, -radius, radius)
        return t, xs

    t, a = roll(0.5)
    _, b = roll(1.2)
    _, c = roll(-1.2)
    step = 20
    write("q3_traj.dat", "t a b c", [t[::step], a[::step], b[::step], c[::step]])
    print("cubic ends", a[-1], b[-1], c[-1], "a at 0.5s", a[5000])


def pulse_integrator():
    tau, tau_in = 0.1, 0.02
    u_of = lambda t: 3.0 if t < 0.4 else 0.0

    def run(neuron_type):
        model = nengo.Network(seed=41)
        with model:
            stim = nengo.Node(u_of)
            x = nengo.Ensemble(300, 1, radius=2, neuron_type=neuron_type)
            nengo.Connection(stim, x, transform=tau, synapse=tau_in)
            nengo.Connection(x, x, synapse=tau)
            p = nengo.Probe(x, synapse=0.01)
        with nengo.Simulator(model, progress_bar=False) as sim:
            sim.run(1.0)
        return sim.trange(), sim.data[p][:, 0]

    t, spk = run(nengo.LIF())
    _, rate_x = run(nengo.LIFRate())
    dt = t[1] - t[0]
    uu = np.array([u_of(ti) for ti in t])
    u_in = np.zeros_like(uu)
    for k in range(1, len(t)):
        u_in[k] = u_in[k - 1] + dt / tau_in * (uu[k - 1] - u_in[k - 1])
    ideal = np.minimum(3 * t, 1.2)
    lead = np.cumsum(uu) * dt + (tau - tau_in) * u_in
    s = slice(None, None, 4)
    write("q5_pulse.dat", "t u ideal lead spike rate",
          [t[s], uu[s], ideal[s], lead[s], spk[s], rate_x[s]])
    i = np.argmin(np.abs(t - 0.39))
    j = np.argmin(np.abs(t - 0.8))
    print("pulse near 0.39 ideal/lead/spike/rate", ideal[i], lead[i], spk[i], rate_x[i])
    print("pulse at 0.8", ideal[j], lead[j], spk[j], rate_x[j])


if __name__ == "__main__":
    tuning_curves()
    cubic_phase()
    pulse_integrator()
