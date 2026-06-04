# NOTE: experimental module. Imports below reference the legacy flat-file
# layout (d2c_loop_1d). To run, point them at thermaloop.thermal.loop_1d.
# Kept verbatim from the original training run for provenance; see README.
"""
DeepONet surrogate for the 1D D2C loop.

Learns the operator G that maps the per-GPU power trace P(t) to the
spatial-temporal loop temperature field T(x, t):

    G : C([0, T]) -> C([0, L] x [0, T])
        P(t)    |->  T(x, t)

Branch net encodes P sampled at fixed sensor times.
Trunk net encodes the query point (x, t).
Output is the dot product of branch and trunk latent vectors plus bias.

This is the methodological keystone of the repo: the same operator-learning
pattern I use for poroelastic stress fields in subsurface work and for
parameter identification in regulatory hydrology, here applied to a
thermal-fluid system. Sub-millisecond inference after offline training,
suitable for inner loops of control optimization and Monte Carlo studies.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from d2c_loop_1d import default_params_1d, simulate_1d


# =============================================================================
# 1. TRAINING DATA GENERATION
# =============================================================================
def random_power_trace(t_axis, rng):
    """
    Synthesize a per-GPU power trace by blending a few canonical patterns.
    The surrogate generalizes only over the distribution it sees here, so
    this set should span the operating envelope.
    """
    mode = rng.choice(['bursty', 'steady_high', 'steady_low',
                       'ramp_up', 'ramp_down', 'cyclic'])
    T = t_axis[-1]
    P_idle, P_dec, P_pf = 84.0, 384.0, 700.0
    P = np.full_like(t_axis, P_idle, dtype=float)

    if mode == 'bursty':
        rate = rng.uniform(0.5, 4.0)
        n = int(rate * T)
        starts = rng.uniform(0, T, size=n)
        durs = rng.uniform(0.5, 8.0, size=n)
        for s, d in zip(starts, durs):
            mask = (t_axis >= s) & (t_axis < s + d)
            P[mask] = rng.choice([P_pf, P_dec])
    elif mode == 'steady_high':
        P[:] = P_pf
        # Inject random idle gaps
        n_gaps = rng.integers(2, 8)
        for _ in range(n_gaps):
            s = rng.uniform(0, T)
            d = rng.uniform(1.0, 20.0)
            mask = (t_axis >= s) & (t_axis < s + d)
            P[mask] = P_idle
    elif mode == 'steady_low':
        P[:] = P_dec
    elif mode == 'ramp_up':
        P = P_idle + (P_pf - P_idle) * (t_axis / T)
        P += rng.normal(0, 30, len(t_axis))
        P = np.clip(P, P_idle, P_pf)
    elif mode == 'ramp_down':
        P = P_pf - (P_pf - P_idle) * (t_axis / T)
        P += rng.normal(0, 30, len(t_axis))
        P = np.clip(P, P_idle, P_pf)
    elif mode == 'cyclic':
        period = rng.uniform(30, 180)
        amp = (P_pf - P_idle) / 2
        mid = (P_pf + P_idle) / 2
        P = mid + amp * np.sin(2 * np.pi * t_axis / period
                                + rng.uniform(0, 2 * np.pi))
        P = np.clip(P, P_idle, P_pf)

    return P


def generate_dataset(n_rollouts=300, T_horizon=300.0, dt=1.0, seed=0,
                     verbose=True):
    """
    Generate (P trace, T field) training pairs by running the 1D solver
    on randomly sampled power traces.
    Returns:
        P_traces : (n_rollouts, n_t)              per-GPU power, W
        T_fields : (n_rollouts, n_x, n_t)         loop temperature, C
        t_axis   : (n_t,)                         time grid
        x_grid   : (n_x,)                         spatial cell positions (0..1)
    """
    rng = np.random.default_rng(seed)
    params = default_params_1d()
    t_axis = np.arange(0, T_horizon, dt)
    n_t = len(t_axis)
    n_x = params['geom']['N']
    x_grid = (np.arange(n_x) + 0.5) / n_x   # cell centers in [0, 1]

    P_traces = np.zeros((n_rollouts, n_t))
    T_fields = np.zeros((n_rollouts, n_x, n_t))

    t0 = time.time()
    for k in range(n_rollouts):
        P = random_power_trace(t_axis, rng)
        sol = simulate_1d(t_axis, P, params)
        T_loop = sol.y[3:3 + n_x, :]   # (N, n_t)
        P_traces[k] = P
        T_fields[k] = T_loop
        if verbose and (k + 1) % 50 == 0:
            dt_elapsed = time.time() - t0
            eta = dt_elapsed / (k + 1) * (n_rollouts - k - 1)
            print(f"    rollout {k+1}/{n_rollouts}  "
                  f"({dt_elapsed:.0f}s elapsed, ~{eta:.0f}s left)")
    if verbose:
        print(f"    Done in {time.time()-t0:.0f}s")
    return P_traces, T_fields, t_axis, x_grid


# =============================================================================
# 2. DEEPONET MODEL
# =============================================================================
class MLP(nn.Module):
    def __init__(self, sizes, act=nn.SiLU):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """
    Branch net : R^{n_sensors}  -> R^{p}
    Trunk net  : R^{d_query}    -> R^{p}
    Output     : <branch, trunk> + bias
    """
    def __init__(self, n_sensors, p=64, d_query=2,
                 branch_hidden=(128, 128),
                 trunk_hidden=(128, 128, 128)):
        super().__init__()
        self.branch = MLP([n_sensors, *branch_hidden, p])
        self.trunk = MLP([d_query, *trunk_hidden, p])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        """
        u : (B, n_sensors)   the input function P sampled
        y : (B, K, d_query)  K query points per sample
        returns (B, K) predictions
        """
        b = self.branch(u)                # (B, p)
        tr = self.trunk(y)                # (B, K, p)
        out = (b.unsqueeze(1) * tr).sum(-1) + self.bias
        return out


# =============================================================================
# 3. NORMALIZATION
# =============================================================================
class FieldNormalizer:
    def __init__(self, P_traces, T_fields):
        self.P_mean = float(P_traces.mean())
        self.P_std = float(P_traces.std() + 1e-8)
        self.T_mean = float(T_fields.mean())
        self.T_std = float(T_fields.std() + 1e-8)

    def norm_P(self, P): return (P - self.P_mean) / self.P_std
    def norm_T(self, T): return (T - self.T_mean) / self.T_std
    def denorm_T(self, Tn): return Tn * self.T_std + self.T_mean


# =============================================================================
# 4. TRAINING
# =============================================================================
def train_surrogate(P_traces, T_fields, t_axis, x_grid,
                    n_sensors=64, K_queries=200,
                    epochs=300, batch_size=32, lr=1e-3,
                    seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Normalize
    norm = FieldNormalizer(P_traces, T_fields)
    P_n = norm.norm_P(P_traces)
    T_n = norm.norm_T(T_fields)
    n_roll, n_t = P_n.shape
    n_x = T_n.shape[1]

    # Subsample P to fixed sensor times for branch input
    sensor_idx = np.linspace(0, n_t - 1, n_sensors).astype(int)
    P_sens = torch.tensor(P_n[:, sensor_idx], dtype=torch.float32)  # (n_roll, n_sensors)

    # Build (x_norm, t_norm) query grid for trunk
    x_norm = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min() + 1e-8)
    t_norm = t_axis / t_axis[-1]
    # Pre-flatten field for convenient query sampling
    # T_n shape: (n_roll, n_x, n_t)

    # Train/val split
    perm = rng.permutation(n_roll)
    n_val = max(1, n_roll // 10)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    device = torch.device('cpu')
    model = DeepONet(n_sensors=n_sensors, p=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {'train': [], 'val': []}
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        rng.shuffle(train_idx)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_idx), batch_size):
            bidx = train_idx[i:i + batch_size]
            B = len(bidx)
            # Sample K random (x, t) queries per rollout in batch
            xi = rng.integers(0, n_x, size=(B, K_queries))
            ti = rng.integers(0, n_t, size=(B, K_queries))
            # Query coordinates (x_norm, t_norm)
            q_x = x_norm[xi]
            q_t = t_norm[ti]
            queries = np.stack([q_x, q_t], axis=-1)            # (B, K, 2)
            # Targets
            targets = T_n[bidx[:, None, None], xi[..., None], ti[..., None]]
            targets = targets.squeeze(-1)                       # (B, K)

            u = P_sens[bidx].to(device)
            y = torch.tensor(queries, dtype=torch.float32, device=device)
            t_target = torch.tensor(targets, dtype=torch.float32, device=device)

            pred = model(u, y)
            loss = nn.functional.mse_loss(pred, t_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        sched.step()
        train_mse = ep_loss / max(1, n_batches)

        # Validation: evaluate on full field for val rollouts
        model.eval()
        with torch.no_grad():
            u_val = P_sens[val_idx].to(device)
            X, Tg = np.meshgrid(x_norm, t_norm, indexing='ij')
            full_q = np.stack([X.ravel(), Tg.ravel()], axis=-1)   # (n_x*n_t, 2)
            y_val = torch.tensor(full_q, dtype=torch.float32, device=device)
            y_val = y_val.unsqueeze(0).expand(len(val_idx), -1, -1)
            pred_val = model(u_val, y_val).cpu().numpy()           # (n_val, n_x*n_t)
            pred_val = pred_val.reshape(len(val_idx), n_x, n_t)
            target_val = T_n[val_idx]
            val_mse = float(((pred_val - target_val) ** 2).mean())

        history['train'].append(train_mse)
        history['val'].append(val_mse)

        if verbose and (ep % 30 == 0 or ep == epochs - 1):
            elapsed = time.time() - t0
            print(f"  epoch {ep:3d}/{epochs}  train MSE {train_mse:.4e}  "
                  f"val MSE {val_mse:.4e}  ({elapsed:.0f}s)")

    return model, norm, history, sensor_idx, x_norm, t_norm


# =============================================================================
# 5. PREDICTION INTERFACE
# =============================================================================
def predict_field(model, norm, P_trace, sensor_idx, x_norm, t_norm,
                  device='cpu', _cache={}):
    """Return predicted T(x, t) for a single power trace. Caches query grid."""
    model.eval()
    cache_key = (id(x_norm), id(t_norm), device)
    if cache_key not in _cache:
        X, Tg = np.meshgrid(x_norm, t_norm, indexing='ij')
        q = np.stack([X.ravel(), Tg.ravel()], axis=-1)
        _cache[cache_key] = torch.tensor(q[None], dtype=torch.float32,
                                          device=device)
    y = _cache[cache_key]
    with torch.no_grad():
        P_n = norm.norm_P(P_trace)
        u = torch.tensor(P_n[sensor_idx][None], dtype=torch.float32,
                         device=device)
        pred_n = model(u, y).cpu().numpy().reshape(len(x_norm), len(t_norm))
        return norm.denorm_T(pred_n)


def predict_field_batch(model, norm, P_batch, sensor_idx, x_norm, t_norm,
                        device='cpu'):
    """
    Predict T(x, t) for a batch of B power traces simultaneously.
    P_batch shape (B, n_t). Returns (B, n_x, n_t).
    The operator-learning advantage: this scales near-linearly with the
    model forward pass, not B times the physics solver.
    """
    model.eval()
    B = P_batch.shape[0]
    n_x, n_t = len(x_norm), len(t_norm)
    X, Tg = np.meshgrid(x_norm, t_norm, indexing='ij')
    q = np.stack([X.ravel(), Tg.ravel()], axis=-1)
    y = torch.tensor(q, dtype=torch.float32, device=device)
    y = y.unsqueeze(0).expand(B, -1, -1)
    with torch.no_grad():
        P_n = norm.norm_P(P_batch)
        u = torch.tensor(P_n[:, sensor_idx], dtype=torch.float32,
                         device=device)
        pred_n = model(u, y).cpu().numpy().reshape(B, n_x, n_t)
        return norm.denorm_T(pred_n)


def benchmark_inference(model, norm, P_trace, sensor_idx, x_norm, t_norm,
                        n_reps=100):
    """Time per-inference call vs physics solver."""
    # Surrogate
    t0 = time.time()
    for _ in range(n_reps):
        _ = predict_field(model, norm, P_trace, sensor_idx, x_norm, t_norm)
    t_surr = (time.time() - t0) / n_reps
    # Physics solver
    params = default_params_1d()
    t_axis = np.arange(0, len(P_trace), 1.0)
    t0 = time.time()
    sol = simulate_1d(t_axis, P_trace, params)
    t_phys = time.time() - t0
    return dict(surrogate_s=t_surr, physics_s=t_phys,
                speedup=t_phys / t_surr)
