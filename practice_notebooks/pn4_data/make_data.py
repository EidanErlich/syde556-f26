"""Generates the simulation data plotted in the PN4 ramp practice solutions.

Run from this directory: ../../.venv/bin/python make_data.py
"""

import sys
import warnings

import numpy as np
import nengo
from nengo.utils.connection import eval_point_decoding
from scipy.linalg import expm

warnings.filterwarnings("ignore")

DT = 0.001


def save(name, header, cols, every=1):
    data = np.column_stack(cols)[::every]
    np.savetxt(name, data, header=header, comments="", fmt="%.5f")


# --------------------------------------------------------------------------
# 1. Accumulator (PN4 section 1)
# --------------------------------------------------------------------------
def choice(x):
    if x[0] > 0.9:
        return 1
    elif x[0] < -0.9:
        return -1
    return 0


def run_accumulator(u, rms=0.0, decision=False, out_intercepts=None, seed=556, T=2.0):
    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node([u])
        acc = nengo.Ensemble(50, 1)
        nengo.Connection(stim, acc)
        nengo.Connection(acc, acc, synapse=0.1)
        if rms > 0:
            acc.noise = nengo.processes.WhiteSignal(period=10, high=100, rms=rms)
        p_acc = nengo.Probe(acc, synapse=0.01)
        if decision:
            kw = {} if out_intercepts is None else {"intercepts": out_intercepts}
            out = nengo.Ensemble(50, 1, **kw)
            nengo.Connection(acc, out, function=choice)
            p_out = nengo.Probe(out, synapse=0.01)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(T)
    t = sim.trange()
    res = [t, sim.data[p_acc][:, 0]]
    if decision:
        res.append(sim.data[p_out][:, 0])
    return res


def accumulator_data():
    inputs = [0.2, 0.1, -0.1, -0.2]
    cols = None
    for u in inputs:
        t, x = run_accumulator(u)
        cols = [t] if cols is None else cols
        cols.append(x)
    save("acc_clean.dat", "t u02 u01 um01 um02", cols, every=10)

    for rms in [3, 10]:
        cols = None
        for u in inputs:
            t, x = run_accumulator(u, rms=rms)
            cols = [t] if cols is None else cols
            cols.append(x)
        save(f"acc_noise{rms}.dat", "t u02 u01 um01 um02", cols, every=10)

    for tag, icpt in [("default", None), ("highint", nengo.dists.Uniform(0.4, 0.9))]:
        cols = None
        for u in [0.2, -0.1]:
            t, x, y = run_accumulator(u, rms=3, decision=True, out_intercepts=icpt)
            cols = [t] if cols is None else cols
            cols += [x, y]
        save(f"decision_{tag}.dat", "t acc02 out02 accm01 outm01", cols, every=10)

    # Static decoding of choice(x) from a 50-neuron ensemble.
    model = nengo.Network(seed=556)
    with model:
        a = nengo.Ensemble(50, 1)
        b = nengo.Node(size_in=1)
        conn = nengo.Connection(a, b, function=choice)
    with nengo.Simulator(model, progress_bar=False) as sim:
        xs = np.linspace(-1, 1, 401)[:, None]
        _, act = nengo.utils.ensemble.tuning_curves(a, sim, inputs=xs)
        dec = sim.data[conn].weights
        yhat = act @ dec.T
    save("choice_decoded.dat", "x yhat", [xs[:, 0], yhat[:, 0]])


# --------------------------------------------------------------------------
# 2. LMU (PN4 section 2)
# --------------------------------------------------------------------------
def lmu_matrices(q, theta):
    A = np.zeros((q, q))
    B = np.zeros((q, 1))
    for i in range(q):
        B[i] = (-1.0) ** i * (2 * i + 1)
        for j in range(q):
            A[i, j] = (2 * i + 1) * (-1 if i < j else (-1.0) ** (i - j + 1))
    return A / theta, B / theta


def stim_12(t):
    return np.sin(2 * np.pi * t) if t < 2 else np.sin(2 * np.pi * t * 2)


def stim_21(t):
    return np.sin(2 * np.pi * t * 2) if t < 2 else np.sin(2 * np.pi * t)


def ideal_lmu(q, theta, stim, T=4.0):
    A, B = lmu_matrices(q, theta)
    Ad = expm(A * DT)
    Bd = np.linalg.solve(A, (Ad - np.eye(q)) @ B)
    ts = np.arange(1, int(round(T / DT)) + 1) * DT
    x = np.zeros(q)
    xs = []
    for t in ts:
        x = Ad @ x + Bd[:, 0] * stim(t)
        xs.append(x.copy())
    return ts, np.array(xs)


def run_lmu(n, q, stim, eval_points=None, target=None, theta=0.5, tau=0.1, seed=556, T=4.0):
    A, B = lmu_matrices(q, theta)
    Ap = tau * A + np.eye(q)
    Bp = tau * B
    model = nengo.Network(seed=seed)
    with model:
        u = nengo.Node(stim)
        a = nengo.Ensemble(n, q)
        nengo.Connection(u, a, transform=Bp, synapse=tau)
        nengo.Connection(a, a, transform=Ap, synapse=tau)
        p_a = nengo.Probe(a, synapse=0.01)
        if eval_points is not None:
            b = nengo.Ensemble(50, 1)
            nengo.Connection(a, b, eval_points=eval_points, function=target)
            p_b = nengo.Probe(b, synapse=0.01)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(T)
    out = sim.data[p_b][:, 0] if eval_points is not None else None
    return sim.trange(), sim.data[p_a], out


def lmu_target(nsteps):
    target = np.ones((nsteps, 1))
    target[nsteps // 2:] = -1
    return target


def lmu_data(do_sweeps=True):
    ts, xs = ideal_lmu(6, 0.5, stim_12)
    save("lmu_ideal.dat", "t " + " ".join(f"x{i}" for i in range(6)),
         [ts] + [xs[:, i] for i in range(6)], every=10)
    print("ideal LMU max |x| =", np.max(np.linalg.norm(xs, axis=1)))

    # q = 1 is a low-pass filter with time constant theta.
    ts1, xs1 = ideal_lmu(1, 0.5, stim_12)
    save("lmu_q1.dat", "t u x0", [ts1, [stim_12(t) for t in ts1], xs1[:, 0]], every=10)

    t, x, _ = run_lmu(1000, 6, stim_12)
    target = lmu_target(len(t))
    save("lmu_neural.dat", "t " + " ".join(f"x{i}" for i in range(6)),
         [t] + [x[:, i] for i in range(6)], every=10)
    _, _, cat = run_lmu(1000, 6, stim_12, eval_points=x, target=target)
    _, _, cat_sw = run_lmu(1000, 6, stim_21, eval_points=x, target=target)
    # Retrained on the swapped order (target still +1 for the first 2 s).
    t2, x2, _ = run_lmu(1000, 6, stim_21)
    _, _, cat_sw_re = run_lmu(1000, 6, stim_21, eval_points=x2, target=target)
    save("lmu_category.dat", "t target cat catsw catswre",
         [t, target[:, 0], cat, cat_sw, cat_sw_re], every=10)
    print("category RMSE 1->2:", np.sqrt(np.mean((cat - target[:, 0]) ** 2)))

    if not do_sweeps:
        return
    rows = []
    for n in [50, 100, 200, 500, 1000, 2000, 5000]:
        t, x, _ = run_lmu(n, 6, stim_12)
        _, _, c = run_lmu(n, 6, stim_12, eval_points=x, target=lmu_target(len(t)))
        rows.append((n, np.sqrt(np.mean((c - lmu_target(len(t))[:, 0]) ** 2))))
        print("n sweep", rows[-1])
    np.savetxt("lmu_sweep_n.dat", rows, header="n rmse", comments="", fmt="%.5f")
    rows = []
    for q in [1, 2, 4, 8, 16, 32, 64]:
        t, x, _ = run_lmu(1000, q, stim_12)
        _, _, c = run_lmu(1000, q, stim_12, eval_points=x, target=lmu_target(len(t)))
        rows.append((q, np.sqrt(np.mean((c - lmu_target(len(t))[:, 0]) ** 2))))
        print("q sweep", rows[-1])
    np.savetxt("lmu_sweep_q.dat", rows, header="q rmse", comments="", fmt="%.5f")


# --------------------------------------------------------------------------
# 3. PES learning (PN4 section 3)
# --------------------------------------------------------------------------
def run_pes(T, lr, seed, func=lambda x: x, rms=0.3, dims_out=1):
    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node(nengo.processes.WhiteSignal(period=100, high=2, rms=rms))
        pre = nengo.Ensemble(200, 1)
        post = nengo.Ensemble(200, dims_out)
        error = nengo.Ensemble(200, dims_out)
        nengo.Connection(stim, pre)
        c = nengo.Connection(pre, post, function=lambda x: np.zeros(dims_out),
                             learning_rule_type=nengo.PES(learning_rate=lr))
        nengo.Connection(post, error)
        nengo.Connection(stim, error, function=func, transform=-1)
        nengo.Connection(error, c.learning_rule)
        p_in = nengo.Probe(stim)
        p_post = nengo.Probe(post, synapse=0.01)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(T)
    target = np.array([np.atleast_1d(func(v)) for v in sim.data[p_in][:, 0]])
    return sim.trange(), sim.data[p_in][:, 0], target, sim.data[p_post]


def chunk_rmse(target, out, chunk_steps):
    n = len(target) // chunk_steps
    return np.array([
        np.sqrt(np.mean((target[k * chunk_steps:(k + 1) * chunk_steps]
                         - out[k * chunk_steps:(k + 1) * chunk_steps]) ** 2))
        for k in range(n)
    ])


def pes_data():
    t, u, target, out = run_pes(10.0, 1e-4, seed=1)
    save("pes_identity.dat", "t u out", [t, u, out[:, 0]], every=10)

    r10 = np.mean([chunk_rmse(*run_pes(10.0, 1e-4, seed=s)[2:], 2000) for s in range(10)], axis=0)
    np.savetxt("pes_rmse10.dat", np.column_stack([np.arange(1, 6) * 2 - 1, r10]),
               header="tmid rmse", comments="", fmt="%.5f")
    print("10 s chunks:", r10)

    for lr, tag in [(1e-4, "1e4"), (1e-5, "1e5")]:
        r = np.mean([chunk_rmse(*run_pes(100.0, lr, seed=s)[2:], 2000) for s in range(4)], axis=0)
        np.savetxt(f"pes_rmse100_{tag}.dat", np.column_stack([np.arange(1, 51) * 2 - 1, r]),
                   header="tmid rmse", comments="", fmt="%.5f")
        print(f"100 s chunks lr={lr}: first", r[:3], "last", r[-3:])

    # Linear versus quadratic target, relative error.
    for tag, f in [("lin", lambda x: x), ("sq", lambda x: x ** 2)]:
        rs = []
        for s in range(4):
            _, _, tg, o = run_pes(60.0, 1e-4, seed=s, func=f, rms=0.5)
            r = chunk_rmse(tg, o, 2000) / np.std(tg)
            rs.append(r)
        r = np.mean(rs, axis=0)
        np.savetxt(f"pes_rel_{tag}.dat", np.column_stack([np.arange(1, 31) * 2 - 1, r]),
                   header="tmid relrmse", comments="", fmt="%.5f")
        print(f"rel {tag}: first", r[:3], "last", r[-3:])


# --------------------------------------------------------------------------
# 4. Adaptive control (PN4 section 4)
# --------------------------------------------------------------------------
G, L, M = 9.8, 1.0, 1.0
KP, KD = 25.0, 10.0


def make_pendulum(theta0=np.pi / 4):
    state = np.array([theta0, 0.0])

    def plant(t, u):
        theta, omega = state
        omega_dot = (u[0] - M * G * L * np.cos(theta)) / (M * L ** 2)
        state[0] = theta + DT * omega
        state[1] = omega + DT * omega_dot
        return state.copy()
    return plant


def pend_target(t):
    return 0.0 if (t % 10.0) < 5.0 else 0.8


def run_pendulum(adaptive, T=40.0, lr=1e-5, seed=556):
    model = nengo.Network(seed=seed)
    with model:
        plant = nengo.Node(make_pendulum(), size_in=1, size_out=2)
        u_pd = nengo.Node(lambda t, s: KP * (pend_target(t) - s[0]) - KD * s[1], size_in=2)
        torque = nengo.Node(size_in=1)
        nengo.Connection(plant, u_pd, synapse=None)
        nengo.Connection(u_pd, torque, synapse=None)
        nengo.Connection(torque, plant, synapse=0)
        p_th = nengo.Probe(plant)
        p_pd = nengo.Probe(u_pd)
        if adaptive:
            ens = nengo.Ensemble(1000, 2, radius=1.5)
            nengo.Connection(plant, ens, transform=[[1, 0], [0, 0.2]], synapse=None)
            conn = nengo.Connection(ens, torque, function=lambda x: 0,
                                    learning_rule_type=nengo.PES(learning_rate=lr))
            nengo.Connection(u_pd, conn.learning_rule, transform=-1)
            p_ad = nengo.Probe(conn, "output", synapse=0.05)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(T)
    out = [sim.trange(), sim.data[p_th][:, 0], sim.data[p_pd][:, 0]]
    if adaptive:
        out.append(sim.data[p_ad][:, 0])
    return out


def pendulum_data():
    t, th_pd, upd_pd = run_pendulum(False)
    t, th_ad, upd_ad, uad = run_pendulum(True)
    target = np.array([pend_target(v) for v in t])
    save("pendulum.dat", "t target thpd thad upd uad", [t, target, th_pd, th_ad, upd_ad, uad],
         every=20)
    for k in range(8):
        i = int((k * 5 + 4.9) / DT)
        print(f"t={t[i]:.1f} target={target[i]} PD-only={th_pd[i]:.3f} adaptive={th_ad[i]:.3f} "
              f"u_ad={uad[i]:.2f}")


# --------------------------------------------------------------------------
# 5. SVD of the activity correlation matrix (PN4 section 5)
# --------------------------------------------------------------------------
def svd_data():
    xs = np.linspace(-1, 1, 501)[:, None]
    results = {}
    for tag, kw in [("hetero", {}),
                    ("clumped", {"intercepts": nengo.dists.Uniform(0.7, 0.95),
                                 "max_rates": nengo.dists.Uniform(150, 200)})]:
        model = nengo.Network(seed=556)
        with model:
            ens = nengo.Ensemble(100, 1, **kw)
        with nengo.Simulator(model, progress_bar=False) as sim:
            _, A = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=xs)
        gamma = A.T @ A
        U, S, _ = np.linalg.svd(gamma)
        results[tag] = (A, U, S)
    A, U, S = results["hetero"]
    N = len(xs)
    sigma = 0.1 * A.max()
    print("noise floor N sigma^2 =", N * sigma ** 2, " S[:8] =", S[:8])
    np.savetxt("svd_spectrum.dat",
               np.column_stack([np.arange(1, 41), results["hetero"][2][:40] / S[0],
                                results["clumped"][2][:40] / results["clumped"][2][0]]),
               header="m hetero clumped", comments="", fmt="%.6e")
    with open("svd_floor.txt", "w") as f:
        f.write(f"{N * sigma ** 2 / S[0]:.6e}\n")
    chi = A @ U[:, :5]
    chi /= np.max(np.abs(chi), axis=0)
    for m in range(5):
        if chi[np.argmax(np.abs(chi[:, m])), m] < 0 and m in (0,):
            chi[:, m] *= -1
    # Orient each chi so that it is positive at x = 1, like the Legendre polynomials.
    chi *= np.sign(chi[-1, :])
    save("svd_chi.dat", "x c0 c1 c2 c3 c4", [xs[:, 0]] + [chi[:, m] for m in range(5)], every=5)

    # 2-D population, for the multiset counting question.
    rng = np.random.RandomState(0)
    pts = rng.uniform(-1, 1, size=(4000, 2))
    pts = pts[np.linalg.norm(pts, axis=1) <= 1]
    model = nengo.Network(seed=556)
    with model:
        ens = nengo.Ensemble(300, 2)
    with nengo.Simulator(model, progress_bar=False) as sim:
        _, A2 = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=pts)
    S2 = np.linalg.svd(A2.T @ A2, compute_uv=False)
    np.savetxt("svd_spectrum_2d.dat", np.column_stack([np.arange(1, 22), S2[:21] / S2[0]]),
               header="m s", comments="", fmt="%.6e")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["acc", "lmu", "pes", "pend", "svd"]
    if "acc" in parts:
        accumulator_data()
    if "lmu" in parts:
        lmu_data()
    if "lmunosweep" in parts:
        lmu_data(do_sweeps=False)
    if "pes" in parts:
        pes_data()
    if "pend" in parts:
        pendulum_data()
    if "svd" in parts:
        svd_data()
