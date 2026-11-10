"""Generates the simulation data plotted in the Practice Test 4 solutions.

Run from this directory: ../../.venv/bin/python make_data_test4.py
"""

import warnings

import numpy as np
import nengo

warnings.filterwarnings("ignore")

DT = 0.001


def save(name, header, cols, every=1):
    np.savetxt(name, np.column_stack(cols)[::every], header=header, comments="", fmt="%.5f")


def choice(x):
    if x[0] > 0.9:
        return 1
    elif x[0] < -0.9:
        return -1
    return 0


# Question 1: slow accumulator (recurrent 0.2 s, input 20 ms, transform 1) and decision.
def q1():
    cols = None
    for u in [0.1, -0.05]:
        model = nengo.Network(seed=4)
        with model:
            stim = nengo.Node([u])
            acc = nengo.Ensemble(100, 1)
            out = nengo.Ensemble(50, 1)
            nengo.Connection(stim, acc, synapse=0.02)
            nengo.Connection(acc, acc, synapse=0.2)
            nengo.Connection(acc, out, function=choice)
            p_acc = nengo.Probe(acc, synapse=0.01)
            p_out = nengo.Probe(out, synapse=0.01)
        with nengo.Simulator(model, progress_bar=False) as sim:
            sim.run(4.0)
        t = sim.trange()
        cols = [t] if cols is None else cols
        cols += [sim.data[p_acc][:, 0], sim.data[p_out][:, 0]]
        a = np.abs(sim.data[p_acc][:, 0])
        print(f"u={u}: |acc| > 0.9 at {t[np.argmax(a > 0.9)]:.3f} s, final acc "
              f"{sim.data[p_acc][-1, 0]:.3f}, final out {sim.data[p_out][-200:, 0].mean():.3f}")
    save("t4_q1.dat", "t acc1 out1 acc2 out2", cols, every=10)


# Question 3: PES learns f(x) = x, then the target flips to f(x) = -x at t = 10 s.
def q3():
    rs = []
    for seed in range(4):
        model = nengo.Network(seed=seed)
        with model:
            stim = nengo.Node(nengo.processes.WhiteSignal(period=100, high=2, rms=0.5))
            sign = nengo.Node(lambda t: 1.0 if t < 10 else -1.0)
            pre = nengo.Ensemble(200, 1)
            post = nengo.Ensemble(200, 1)
            error = nengo.Ensemble(200, 1)
            target = nengo.Node(lambda t, v: v[0] * v[1], size_in=2)
            nengo.Connection(stim, pre)
            nengo.Connection(stim, target[0], synapse=None)
            nengo.Connection(sign, target[1], synapse=None)
            c = nengo.Connection(pre, post, function=lambda x: 0,
                                 learning_rule_type=nengo.PES(learning_rate=1e-4))
            nengo.Connection(post, error)
            nengo.Connection(target, error, transform=-1)
            nengo.Connection(error, c.learning_rule)
            p_t = nengo.Probe(target)
            p_post = nengo.Probe(post, synapse=0.01)
        with nengo.Simulator(model, progress_bar=False) as sim:
            sim.run(20.0)
        e = sim.data[p_post][:, 0] - sim.data[p_t][:, 0]
        rs.append([np.sqrt(np.mean(e[k * 500:(k + 1) * 500] ** 2)) for k in range(40)])
        if seed == 0:
            save("t4_q3_trace.dat", "t target out",
                 [sim.trange(), sim.data[p_t][:, 0], sim.data[p_post][:, 0]], every=10)
    r = np.mean(rs, axis=0)
    np.savetxt("t4_q3_rmse.dat", np.column_stack([np.arange(40) * 0.5 + 0.25, r]),
               header="tmid rmse", comments="", fmt="%.5f")
    print("Q3 RMSE per 0.5 s:", r.round(3).tolist())


# Question 5: PD (kp = 16, kd = 8) with and without adaptation, target alternating 0 and pi/2.
G, KP, KD = 9.8, 16.0, 8.0


def target5(t):
    return 0.0 if (t % 10.0) < 5.0 else np.pi / 2


def run_pend(adaptive, T=40.0, lr=1e-5, seed=4):
    state = np.array([0.0, 0.0])

    def plant(t, u):
        theta, omega = state
        state[0] = theta + DT * omega
        state[1] = omega + DT * (u[0] - G * np.cos(theta))
        return state.copy()

    model = nengo.Network(seed=seed)
    with model:
        p = nengo.Node(plant, size_in=1, size_out=2)
        u_pd = nengo.Node(lambda t, s: KP * (target5(t) - s[0]) - KD * s[1], size_in=2)
        torque = nengo.Node(size_in=1)
        nengo.Connection(p, u_pd, synapse=None)
        nengo.Connection(u_pd, torque, synapse=None)
        nengo.Connection(torque, p, synapse=0)
        p_th = nengo.Probe(p)
        if adaptive:
            ens = nengo.Ensemble(1000, 2, radius=2.0)
            nengo.Connection(p, ens, transform=[[1, 0], [0, 0.2]], synapse=None)
            conn = nengo.Connection(ens, torque, function=lambda x: 0,
                                    learning_rule_type=nengo.PES(learning_rate=lr))
            nengo.Connection(u_pd, conn.learning_rule, transform=-1)
            p_ad = nengo.Probe(conn, "output", synapse=0.05)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(T)
    out = [sim.trange(), sim.data[p_th][:, 0]]
    if adaptive:
        out.append(sim.data[p_ad][:, 0])
    return out


def q5():
    t, th_pd = run_pend(False)
    _, th_ad, uad = run_pend(True)
    tg = np.array([target5(v) for v in t])
    save("t4_q5.dat", "t target thpd thad uad", [t, tg, th_pd, th_ad, uad], every=20)
    for k in range(8):
        i = int((k * 5 + 4.9) / DT)
        print(f"t={t[i]:.1f} target={tg[i]:.2f} PD={th_pd[i]:.3f} adaptive={th_ad[i]:.3f} u_ad={uad[i]:.2f}")


if __name__ == "__main__":
    q1()
    q3()
    q5()
