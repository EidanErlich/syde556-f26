"""Generates the simulation data plotted in the Test 1 and Test 2 solutions.

Run from this directory: ../../.venv/bin/python make_data.py
"""

import numpy as np

TAU_RC = 0.02
TAU_REF = 0.002


def lif_rate(J):
    J = np.asarray(J, dtype=float)
    out = np.zeros_like(J)
    m = J > 1
    out[m] = 1.0 / (TAU_REF - TAU_RC * np.log(1 - 1 / J[m]))
    return out


def lif_gain_bias(intercept, max_rate):
    """Gain and bias so that the neuron starts firing at `intercept` and hits `max_rate` at e.x = 1."""
    J_max = 1.0 / (1 - np.exp((TAU_REF - 1.0 / max_rate) / TAU_RC))
    alpha = (J_max - 1) / (1 - intercept)
    return alpha, 1 - alpha * intercept


def lif_spikes(J, dt):
    """Spike trains (impulses of height 1/dt) for currents J of shape (n, steps)."""
    n, steps = J.shape
    v = np.zeros(n)
    ref = np.zeros(n)
    s = np.zeros_like(J)
    for k in range(steps):
        dv = dt / TAU_RC * (J[:, k] - v)
        v = np.where(ref > 0, 0.0, v + dv)
        ref = np.maximum(ref - dt, 0)
        fired = v >= 1
        s[fired, k] = 1.0 / dt
        v[fired] = 0
        ref[fired] = TAU_REF
    return s


def exp_filter(sig, tau, dt):
    out = np.zeros_like(sig)
    acc = 0.0
    a = dt / tau
    for k in range(sig.size):
        acc += a * (sig[k] - acc)
        out[k] = acc
    return out


def bandlimited_noise(T, dt, L, rms, rng):
    N = int(round(T / dt))
    freqs = np.fft.fftfreq(N, dt)
    X = np.zeros(N, dtype=complex)
    pos = (freqs > 0) & (freqs <= L)
    X[pos] = rng.normal(size=pos.sum()) + 1j * rng.normal(size=pos.sum())
    for i in np.where(pos)[0]:
        X[-i] = np.conj(X[i])
    x = np.real(np.fft.ifft(X))
    x *= rms / np.sqrt(np.mean(x**2))
    return x, np.fft.fft(x), freqs


def save(name, header, *cols):
    np.savetxt(name, np.column_stack(cols), header=header, comments="", fmt="%.6g")


rng = np.random.default_rng(556)

# Test 2 Q1: average |X(f)| over many band-limited white noise signals.
T, dt, L, rms = 4.0, 1e-3, 5.0, 0.5
mags = []
for _ in range(400):
    x, X, freqs = bandlimited_noise(T, dt, L, rms, rng)
    mags.append(np.abs(X))
mag = np.mean(mags, axis=0)
order = np.argsort(freqs)
f_sorted, mag_sorted = freqs[order], mag[order]
keep = np.abs(f_sorted) <= 12
save("t2q1_spectrum.dat", "f mag", f_sorted[keep], mag_sorted[keep] / mag_sorted[keep].max())
x, _, _ = bandlimited_noise(T, dt, L, rms, rng)
ts = np.arange(x.size) * dt
save("t2q1_signal.dat", "t x", ts[ts <= 2][::5], x[ts <= 2][::5])

# Test 2 Q2: LIF membrane voltage for a constant supra-threshold input.
dt_v = 5e-5
ts = np.arange(0, 0.1, dt_v)
v_trace, v, ref = [], 0.0, 0.0
J = np.where(ts >= 0.01, 1.5, 0.0)
spike_times = []
for k, t in enumerate(ts):
    if ref > 0:
        v, ref = 0.0, ref - dt_v
    else:
        v += dt_v / TAU_RC * (J[k] - v)
        if v >= 1:
            spike_times.append(t)
            v_trace.append(1.0)
            v, ref = 0.0, TAU_REF
            continue
    v_trace.append(v)
save("t2q2_voltage.dat", "t v J", ts[::4] * 1e3, np.array(v_trace)[::4], J[::4])
save("t2q2_spikes.dat", "t", np.array(spike_times) * 1e3)

# Two symmetric LIF neurons (same alpha and bias, opposite encoders), used by Test 2 Q3, Q5, bonus.
MAX_RATE, INTERCEPT = 60.0, 0.0
alpha, J_bias = lif_gain_bias(INTERCEPT, MAX_RATE)
enc = np.array([1.0, -1.0])
xs = np.linspace(-1, 1, 201)
A = lif_rate(alpha * np.outer(enc, xs) + J_bias)
sigma = 0.1 * A.max()
d = np.linalg.solve(A @ A.T + xs.size * sigma**2 * np.eye(2), A @ xs)

dt = 1e-4
ts = np.arange(0, 2.0, dt)
x = np.sin(2 * np.pi * ts)
S = lif_spikes(alpha * np.outer(enc, x) + J_bias, dt)
decoded = d @ S
step = int(1e-3 / dt)
cols = [ts[::step], x[::step]]
for tau in (0.005, 0.02, 0.05):
    cols.append(exp_filter(decoded, tau, dt)[::step])
save("t2q5_decode.dat", "t x tau5 tau20 tau50", *cols)
for i in range(2):
    save(f"t2q5_spikes{i + 1}.dat", "t", ts[S[i] > 0])

# Test 2 bonus: optimal linear filter for the signed spike response r = s1 - s2.
dt_f, T_f, L_f = 1e-3, 2.0, 5.0
N = int(T_f / dt_f)
num = np.zeros(N, dtype=complex)
den = np.zeros(N)
for _ in range(60):
    x_tr, X_tr, freqs = bandlimited_noise(T_f, dt_f, L_f, 0.5, rng)
    S_tr = lif_spikes(alpha * np.outer(enc, x_tr) + J_bias, dt_f)
    R = np.fft.fft(S_tr[0] - S_tr[1])
    num += X_tr * np.conj(R)
    den += np.abs(R) ** 2
df = 1.0 / T_f
k = np.fft.fftfreq(N, 1.0 / N)
W = np.exp(-0.5 * (k * df / 1.0) ** 2)


def window(spec):
    return np.fft.ifft(np.fft.fft(spec) * np.fft.fft(W))


H = window(num) / np.real(window(den))
h = np.real(np.fft.ifft(H))
lag_t = np.fft.fftfreq(N, 1.0 / N) * dt_f
order = np.argsort(lag_t)
lag_t, h_sorted = lag_t[order], h[order]
keep = np.abs(lag_t) <= 0.15
save("t2b_filter.dat", "t h", lag_t[keep] * 1e3, h_sorted[keep] / dt_f)
order = np.argsort(freqs)
keep = np.abs(freqs[order]) <= 25
save("t2b_gain.dat", "f H", freqs[order][keep], np.abs(H)[order][keep] / np.abs(H)[order][keep].max())

x_te, _, _ = bandlimited_noise(T_f, dt_f, L_f, 0.5, rng)
S_te = lif_spikes(alpha * np.outer(enc, x_te) + J_bias, dt_f)
x_hat = np.real(np.fft.ifft(H * np.fft.fft(S_te[0] - S_te[1])))
ts_te = np.arange(N) * dt_f
save("t2b_decode.dat", "t x xhat", ts_te[::2], x_te[::2], x_hat[::2])

# Test 1 Q4: distortion and noise error against the number of neurons.
xs = np.linspace(-1, 1, 400)
ns = np.unique(np.round(np.logspace(np.log10(4), np.log10(1000), 14)).astype(int))
rows = []
for n in ns:
    e_dist, e_noise = [], []
    for _ in range(10):
        intercepts = rng.uniform(-0.95, 0.95, n)
        rates = rng.uniform(100, 200, n)
        encs = rng.choice([-1.0, 1.0], n)
        gb = [lif_gain_bias(c, r) for c, r in zip(intercepts, rates)]
        gains = np.array([g for g, _ in gb])
        biases = np.array([b for _, b in gb])
        A = lif_rate(gains[:, None] * encs[:, None] * xs[None, :] + biases[:, None])
        sig = 0.1 * 200
        D = np.linalg.solve(A @ A.T + xs.size * sig**2 * np.eye(n), A @ xs)
        e_dist.append(np.mean((xs - D @ A) ** 2))
        e_noise.append(sig**2 * D @ D)
    rows.append((n, np.mean(e_dist), np.mean(e_noise)))
rows = np.array(rows)
save("t1q4_errors.dat", "n dist noise", rows[:, 0], rows[:, 1], rows[:, 2])

# Test 1 Q3(c): decoding noisy activities with and without the noise term.
n = 20
xs = np.linspace(-1, 1, 200)
intercepts = rng.uniform(-0.95, 0.95, n)
rates = rng.uniform(100, 200, n)
encs = rng.choice([-1.0, 1.0], n)
gb = [lif_gain_bias(c, r) for c, r in zip(intercepts, rates)]
gains = np.array([g for g, _ in gb])
biases = np.array([b for _, b in gb])
A = lif_rate(gains[:, None] * encs[:, None] * xs[None, :] + biases[:, None])
sig = 0.1 * 200
d_plain = np.linalg.lstsq(A.T, xs, rcond=None)[0]
d_reg = np.linalg.solve(A @ A.T + xs.size * sig**2 * np.eye(n), A @ xs)
A_noisy = A + rng.normal(0, sig, A.shape)
xh_plain, xh_reg = d_plain @ A_noisy, d_reg @ A_noisy
save("t1q3_noise.dat", "x plain reg", xs, xh_plain, xh_reg)
print("Q3c rmse plain", np.sqrt(np.mean((xs - xh_plain) ** 2)), "reg", np.sqrt(np.mean((xs - xh_reg) ** 2)))
print("Q3c |d| plain", np.abs(d_plain).max(), "reg", np.abs(d_reg).max())
print("Q3c clean rmse plain", np.sqrt(np.mean((xs - d_plain @ A) ** 2)), "reg", np.sqrt(np.mean((xs - d_reg @ A) ** 2)))

print("alpha", alpha, "J_bias", J_bias, "d", d)
print(rows)
