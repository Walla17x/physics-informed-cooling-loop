"""
Train the DeepONet surrogate on a pre-generated dataset.

Saves a checkpoint after every checkpoint_every epochs and at end, so a long
training run can be split across multiple invocations.

Usage:
    python train_surrogate.py --epochs 80 --resume

Outputs:
    surrogate_model.pt     model weights + normalization + training history
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn

from d2c_surrogate import DeepONet, FieldNormalizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), 'surrogate_model.pt')
DATA_PATH = '/tmp/dataset.npz'


def main(epochs, resume, lr, batch_size, K_queries,
         net_p, branch_hidden, trunk_hidden):
    d = np.load(DATA_PATH)
    P, T, t_axis, x_grid = d['P'], d['T'], d['t_axis'], d['x_grid']
    n_roll, n_t = P.shape
    n_x = T.shape[1]

    torch.manual_seed(7)
    rng = np.random.default_rng(7)

    norm = FieldNormalizer(P, T)
    P_n = norm.norm_P(P)
    T_n = norm.norm_T(T)

    n_sensors = 64
    sensor_idx = np.linspace(0, n_t - 1, n_sensors).astype(int)
    P_sens = torch.tensor(P_n[:, sensor_idx], dtype=torch.float32)
    x_norm = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min() + 1e-8)
    t_norm = t_axis / t_axis[-1]

    perm = rng.permutation(n_roll)
    n_val = 40
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    model = DeepONet(n_sensors=n_sensors, p=net_p,
                     branch_hidden=branch_hidden,
                     trunk_hidden=trunk_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    history = {'train': [], 'val': []}
    best_val = float('inf')
    best_state = None

    if resume and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        history = ckpt.get('history', history)
        best_val = ckpt.get('best_val_mse', best_val)
        best_state = ckpt['state_dict']
        start_epoch = len(history['train'])
        if 'opt_state_dict' in ckpt:
            opt.load_state_dict(ckpt['opt_state_dict'])
        print(f"  Resumed from epoch {start_epoch}, best val {best_val:.4e}",
              flush=True)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       last_epoch=start_epoch - 1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)
    print(f"Training epochs {start_epoch} -> {epochs}", flush=True)

    X, Tg = np.meshgrid(x_norm, t_norm, indexing='ij')
    full_q = np.stack([X.ravel(), Tg.ravel()], axis=-1)
    y_val_full = torch.tensor(full_q, dtype=torch.float32).unsqueeze(0).expand(n_val, -1, -1)

    t0 = time.time()
    for ep in range(start_epoch, epochs):
        model.train()
        rng.shuffle(train_idx)
        ep_loss = 0.0
        nb = 0
        for i in range(0, len(train_idx), batch_size):
            bidx = train_idx[i:i + batch_size]
            B = len(bidx)
            xi = rng.integers(0, n_x, size=(B, K_queries))
            ti = rng.integers(0, n_t, size=(B, K_queries))
            queries = np.stack([x_norm[xi], t_norm[ti]], axis=-1)
            targets = T_n[bidx[:, None, None], xi[..., None], ti[..., None]].squeeze(-1)
            u = P_sens[bidx]
            y = torch.tensor(queries, dtype=torch.float32)
            tt = torch.tensor(targets, dtype=torch.float32)
            pred = model(u, y)
            loss = nn.functional.mse_loss(pred, tt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.detach().item()
            nb += 1
        sched.step()
        train_mse = ep_loss / nb

        model.eval()
        with torch.no_grad():
            pred_val = model(P_sens[val_idx], y_val_full).numpy()
            pred_val = pred_val.reshape(n_val, n_x, n_t)
            val_mse = float(((pred_val - T_n[val_idx]) ** 2).mean())
        history['train'].append(train_mse)
        history['val'].append(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == epochs - 1:
            rmse = np.sqrt(val_mse) * norm.T_std
            elapsed = time.time() - t0
            print(f"  ep {ep:3d}  tr {train_mse:.3e}  val {val_mse:.3e}  "
                  f"RMSE {rmse:.3f}K  t={elapsed:.0f}s", flush=True)

    # Save
    torch.save({
        'state_dict': best_state,
        'opt_state_dict': opt.state_dict(),
        'norm_P_mean': float(norm.P_mean), 'norm_P_std': float(norm.P_std),
        'norm_T_mean': float(norm.T_mean), 'norm_T_std': float(norm.T_std),
        'sensor_idx': sensor_idx, 'x_norm': x_norm, 't_norm': t_norm,
        'history': history, 'best_val_mse': best_val,
        'config': {'n_sensors': n_sensors, 'p': net_p,
                   'branch_hidden': tuple(branch_hidden),
                   'trunk_hidden': tuple(trunk_hidden)},
    }, CKPT_PATH)
    print(f"saved {CKPT_PATH}")
    print(f"best val MSE {best_val:.4e}, RMSE {np.sqrt(best_val)*norm.T_std:.3f} K",
          flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--K_queries', type=int, default=192)
    ap.add_argument('--net_p', type=int, default=96)
    args = ap.parse_args()
    main(epochs=args.epochs, resume=args.resume,
         lr=args.lr, batch_size=args.batch_size, K_queries=args.K_queries,
         net_p=args.net_p,
         branch_hidden=(192, 192),
         trunk_hidden=(192, 192, 192))
