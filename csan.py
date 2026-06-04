from __future__ import annotations
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

import xarray as xr

# ============================================================
# 1) Core CSAN pieces (single-parameter alpha, uniform pooling)
# ============================================================

import numpy as np

def _cov_shrinkage(X: np.ndarray, lam_diag: float = 0.1, ridge: float = 1e-6) -> np.ndarray:
    """
    X: (m, n) blocks x peers
    Returns SPD-ish covariance with diagonal shrinkage + ridge.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # MLE covariance (divide by m, not m-1) to reduce noise for small m
    m = Xc.shape[0]
    Sigma = (Xc.T @ Xc) / max(m, 1)
    diag = np.diag(np.diag(Sigma))
    Sigma = (1.0 - lam_diag) * Sigma + lam_diag * diag
    Sigma = Sigma + ridge * np.eye(Sigma.shape[0], dtype=np.float64)
    return Sigma


def apply_linear_allocation(X: np.ndarray, alpha: float, A_bar: np.ndarray) -> np.ndarray:
    """
    Generalization of pooling: y = x A(alpha)^T with A(alpha)=(1-alpha)I + alpha A_bar.
    X: (B, n). A_bar: (n, n) row-stochastic.
    """
    X = np.asarray(X, dtype=np.float32)
    A_bar = np.asarray(A_bar, dtype=np.float32)
    n = X.shape[1]
    A = (1.0 - float(alpha)) * np.eye(n, dtype=np.float32) + float(alpha) * A_bar
    return X @ A


def build_variance_optimal_Abar(
    XT: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    lam_diag: float = 0.1,
    ridge: float = 1e-6,
    n_iter: int = 400,
    lr: float = 0.2,
    seed: int = 0,
    use_torch: bool = True,
) -> np.ndarray:
    """
    Build row-stochastic A_bar by minimizing sum_i a_i^T Sigma a_i subject to a_i in simplex.
    Optionally restrict each row's support using support_mask (n,n) boolean: True means allowed.

    Returns: A_bar (n,n) float32 row-stochastic.
    """
    Sigma = _cov_shrinkage(XT, lam_diag=lam_diag, ridge=ridge)  # (n,n) float64
    n = Sigma.shape[0]

    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)
        assert support_mask.shape == (n, n)

    if not use_torch:
        # Fallback: simple projected gradient in numpy (slower / less stable). Prefer torch.
        return _build_variance_optimal_Abar_numpy(Sigma, support_mask, n_iter=n_iter, lr=lr, seed=seed)

    import torch

    torch.manual_seed(seed)
    device = torch.device("cpu")

    Sigma_t = torch.tensor(Sigma, dtype=torch.float64, device=device)

    def proj_simplex(v: torch.Tensor) -> torch.Tensor:
        """
        Euclidean projection onto simplex {x>=0, sum x = 1}.
        v: (k,)
        """
        # From Duchi et al. (2008)
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1.0
        ind = torch.arange(1, v.numel() + 1, dtype=v.dtype, device=v.device)
        cond = u - cssv / ind > 0
        rho = torch.nonzero(cond, as_tuple=False).max()
        theta = cssv[rho] / (rho.item() + 1.0)
        w = torch.clamp(v - theta, min=0.0)
        return w

    A_bar = np.zeros((n, n), dtype=np.float32)

    # Solve each row independently: min a^T Sigma a s.t. a in simplex (with optional support)
    for i in range(n):
        if support_mask is None:
            idx = np.arange(n)
        else:
            idx = np.where(support_mask[i])[0]
            if idx.size == 0:
                idx = np.array([i], dtype=int)  # safety fallback: self only

        k = idx.size
        # initialize uniformly over allowed support
        a = torch.full((k,), 1.0 / k, dtype=torch.float64, device=device)

        # Pre-extract submatrix for speed
        Sig_sub = Sigma_t[idx][:, idx]  # (k,k)

        for _ in range(n_iter):
            # grad of a^T Sig a is 2 Sig a
            grad = 2.0 * (Sig_sub @ a)
            a = a - lr * grad
            a = proj_simplex(a)

        row = np.zeros(n, dtype=np.float32)
        row[idx] = a.detach().cpu().numpy().astype(np.float32)
        A_bar[i] = row

    # final safety normalize
    rs = A_bar.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    A_bar = A_bar / rs
    return A_bar


def _build_variance_optimal_Abar_numpy(
    Sigma: np.ndarray,
    support_mask: np.ndarray | None,
    *,
    n_iter: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    """
    Numpy fallback if torch isn't available (kept simple).
    """
    rng = np.random.default_rng(seed)
    n = Sigma.shape[0]
    A_bar = np.zeros((n, n), dtype=np.float32)

    def proj_simplex_np(v: np.ndarray) -> np.ndarray:
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, len(v) + 1)
        cond = u - cssv / ind > 0
        rho = np.where(cond)[0].max()
        theta = cssv[rho] / (rho + 1.0)
        w = np.maximum(v - theta, 0.0)
        return w

    for i in range(n):
        if support_mask is None:
            idx = np.arange(n)
        else:
            idx = np.where(support_mask[i])[0]
            if idx.size == 0:
                idx = np.array([i], dtype=int)

        Sig_sub = Sigma[np.ix_(idx, idx)]
        k = idx.size
        a = np.ones(k, dtype=np.float64) / k

        for _ in range(n_iter):
            grad = 2.0 * (Sig_sub @ a)
            a = a - lr * grad
            a = proj_simplex_np(a)

        row = np.zeros(n, dtype=np.float32)
        row[idx] = a.astype(np.float32)
        A_bar[i] = row

    rs = A_bar.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (A_bar / rs).astype(np.float32)

def apply_uniform_pooling(X: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mean_per_block = X.mean(axis=1, keepdims=True)
    return (1.0 - alpha) * X + alpha * mean_per_block

def empirical_quantile_per_peer(X: np.ndarray, tau: float) -> np.ndarray:
    return np.quantile(X, tau, axis=0)

def conf_caps_split_conformal(X_cal: np.ndarray, delta: float) -> np.ndarray:
    X_cal = np.asarray(X_cal, dtype=np.float64)  # float64 for safety
    m, n = X_cal.shape
    tau = 1.0 - delta
    k = int(np.ceil((m + 1) * tau))  # 1-indexed
    k = min(max(k, 1), m)

    caps = np.empty(n, dtype=np.float64)
    for i in range(n):
        caps[i] = np.partition(X_cal[:, i], k - 1)[k - 1]
    return caps.astype(np.float32)

def proxy_harm(q_alpha: np.ndarray, q0: np.ndarray, w: np.ndarray, eta: float) -> float:
    return float(np.sum(w * np.maximum(q_alpha - q0 - eta, 0.0)))


# def _apply_pooling(X: np.ndarray, alpha: float, W_local: np.ndarray | None) -> np.ndarray:
#     """Global if W_local is None; local otherwise."""
#     if W_local is None:
#         return apply_uniform_pooling(X, alpha)
#     return apply_local_pooling(X, alpha, W_local)

def _apply_mechanism(
    X: np.ndarray,
    alpha: float,
    W_local: np.ndarray | None,
    A_bar: np.ndarray | None,
) -> np.ndarray:
    """
    Priority:
      - if A_bar is provided: use linear allocation with A(alpha)=(1-a)I + a A_bar
      - else: fall back to your existing global/local pooling
    """
    if A_bar is not None:
        return apply_linear_allocation(X, alpha, A_bar)
    # existing behavior
    if W_local is None:
        return apply_uniform_pooling(X, alpha)
    return apply_local_pooling(X, alpha, W_local)


def csan_select_alpha_on_V(
    XV: np.ndarray,
    delta: float,
    eta: float,
    eps: float,
    w: np.ndarray,
    alphas: np.ndarray,
    W_local: np.ndarray | None = None,
    c_min: float = 0.0,
    A_bar: np.ndarray | None = None
):
    tau = 1.0 - delta
    q_val0 = empirical_quantile_per_peer(XV, tau)

    # --- floor the proxy baseline quantiles ---
    q_val0_f = apply_cap_floor(q_val0, c_min)
    H_val = float(eps * np.dot(w, q_val0_f))

    obj_list, harm_list = [], []
    for a in alphas:
        # YV = _apply_pooling(XV, float(a), W_local)
        YV = _apply_mechanism(XV, float(a), W_local, A_bar=A_bar)  # no A_bar in this step; just use pooling as before
        q_a = empirical_quantile_per_peer(YV, tau)

        # --- NEW: floor candidate quantiles for proxy harm too ---
        q_a_f = apply_cap_floor(q_a, c_min)

        obj_list.append(float(np.dot(w, q_a_f)))
        harm_list.append(proxy_harm(q_a_f, q_val0_f, w, eta))

    obj_list = np.array(obj_list, dtype=np.float64)
    harm_list = np.array(harm_list, dtype=np.float64)
    feasible = harm_list <= H_val

    if not np.any(feasible):
        chosen_idx = int(np.where(alphas == 0.0)[0][0]) if np.any(alphas == 0.0) else 0
    else:
        chosen_idx = int(np.argmin(np.where(feasible, obj_list, np.inf)))

    alpha_star = float(alphas[chosen_idx])
    return alpha_star, {
        "alphas": alphas, "obj": obj_list, "harm": harm_list,
        "H_val": H_val, "feasible": feasible, "chosen_idx": chosen_idx,
    }


def csan_certify_on_C(
    XC: np.ndarray,
    alpha_star: float,
    delta: float,
    eta: float,
    eps: float,
    w: np.ndarray,
    W_local: np.ndarray | None = None,
    c_min: float = 0.0,
    A_bar: np.ndarray | None = None
):
    # baseline (identity)
    c0 = conf_caps_split_conformal(XC, delta)

    # candidate
    # YC = _apply_pooling(XC, float(alpha_star), W_local)  # your wrapper: global if None else local
    YC = _apply_mechanism(XC, float(alpha_star), W_local, A_bar=A_bar)  # no A_bar in certification step; just use pooling as before
    c_star = conf_caps_split_conformal(YC, delta)

    # --- apply floor (componentwise) ---
    c0_f = apply_cap_floor(c0, c_min)
    c_star_f = apply_cap_floor(c_star, c_min)

    # --- compute budget/harm using FLOORED caps ---
    H = float(eps * np.dot(w, c0_f))
    Harm = float(np.sum(w * np.maximum(c_star_f - c0_f - eta, 0.0)))
    status = "PASS" if Harm <= H else "FAIL"

    alpha_op = float(alpha_star) if status == "PASS" else 0.0
    c_op_raw = c_star if status == "PASS" else c0

    # --- operational caps should also be floored (more conservative) ---
    c_op = apply_cap_floor(c_op_raw, c_min)

    return {
        "c0": c0_f,                # <-- return floored versions everywhere
        "c_star": c_star_f,
        "c_op": c_op,
        "alpha_star": float(alpha_star),
        "alpha_op": alpha_op,
        "status": status,
        "H_cert": H,
        "Harm_cert": Harm,
    }

# ============================================================
# 2) Split generators
# ============================================================

def rolling_backtest_splits(
    years: np.ndarray,
    nT: int,
    nV: int,
    nC: int,
    nTest: int = 1,
    step: int = 1,
    start_at: int | None = None,
):
    years = np.asarray(years)
    order = np.argsort(years)
    yrs_sorted = years[order]

    B = len(years)
    window = nT + nV + nC + nTest
    if B < window:
        raise ValueError(f"Not enough blocks: B={B}, need at least {window}.")

    if start_at is None:
        start_at = 0

    splits = []
    t = start_at
    while t + window <= B:
        T_idx = order[t : t + nT]
        V_idx = order[t + nT : t + nT + nV]
        C_idx = order[t + nT + nV : t + nT + nV + nC]
        Test_idx = order[t + nT + nV + nC : t + window]

        splits.append({
            "T": T_idx,
            "V": V_idx,
            "C": C_idx,
            "Test": Test_idx,
            "years_T": yrs_sorted[t : t + nT],
            "years_V": yrs_sorted[t + nT : t + nT + nV],
            "years_C": yrs_sorted[t + nT + nV : t + nT + nV + nC],
            "years_Test": yrs_sorted[t + nT + nV + nC : t + window],
        })
        t += step

    return splits

def random_backtest_splits(
    years: np.ndarray,
    nT: int,
    nV: int,
    nC: int,
    nTest: int = 1,
    n_splits: int = 30,
    seed: int = 0,
):
    years = np.asarray(years)
    B = len(years)
    window = nT + nV + nC + nTest
    if B < window:
        raise ValueError(f"Not enough blocks: B={B}, need at least {window}.")

    rng = np.random.default_rng(seed)
    splits = []
    all_idx = np.arange(B)

    for r in range(n_splits):
        perm = rng.permutation(all_idx)
        T_idx = perm[:nT]
        V_idx = perm[nT:nT + nV]
        C_idx = perm[nT + nV:nT + nV + nC]
        Test_idx = perm[nT + nV + nC:window]

        splits.append({
            "T": T_idx,
            "V": V_idx,
            "C": C_idx,
            "Test": Test_idx,
            "years_T": np.sort(years[T_idx]),
            "years_V": np.sort(years[V_idx]),
            "years_C": np.sort(years[C_idx]),
            "years_Test": np.sort(years[Test_idx]),
            "split_id": r,
        })

    return splits


def build_local_pool_matrix_from_grid(
    grid_shape: tuple[int, int],
    radius: int = 1,
    include_self: bool = True,
) -> np.ndarray:
    """
    Build a row-stochastic matrix W (n x n) where row i averages a local
    (2*radius+1)x(2*radius+1) neighborhood around peer i on a 2D grid.

    If y = X @ W.T, then y[b,i] is the neighborhood-average obligation for peer i in block b.
    """
    H, Wd = grid_shape
    n = H * Wd
    Wmat = np.zeros((n, n), dtype=np.float32)

    def idx(r, c):
        return r * Wd + c

    for r in range(H):
        for c in range(Wd):
            i = idx(r, c)
            neigh = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < Wd:
                        if (dr == 0 and dc == 0 and not include_self):
                            continue
                        neigh.append(idx(rr, cc))

            # Safety: if include_self=False and radius=0, neighborhood could be empty
            if len(neigh) == 0:
                neigh = [i]

            w = 1.0 / len(neigh)
            Wmat[i, neigh] = w

    return Wmat


def build_local_pool_matrix_from_coords(
    peer_coords: np.ndarray,
    k: int = 9,
    include_self: bool = True,
) -> np.ndarray:
    """
    Build a row-stochastic matrix W (n x n) using k-nearest neighbors in coordinate space.
    peer_coords: (n,2) numeric array (e.g., lat/lon or grid indices).
    """
    coords = np.asarray(peer_coords, dtype=np.float64)
    n = coords.shape[0]
    Wmat = np.zeros((n, n), dtype=np.float32)

    # Pairwise squared distances (n x n). O(n^2), fine for n~144.
    d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)

    for i in range(n):
        order = np.argsort(d2[i])  # includes i itself at position 0
        if include_self:
            neigh = order[:k]
        else:
            neigh = order[1:k+1]
        if len(neigh) == 0:
            neigh = np.array([i], dtype=int)

        Wmat[i, neigh] = 1.0 / len(neigh)

    return Wmat


def apply_local_pooling(X: np.ndarray, alpha: float, W_local: np.ndarray) -> np.ndarray:
    """
    Local pooling transform:
      y_i = (1-alpha) x_i + alpha * (local-average around i)

    W_local must be row-stochastic (rows sum to 1). We compute local averages via X @ W_local.
    """
    X = np.asarray(X, dtype=np.float32)
    W_local = np.asarray(W_local, dtype=np.float32)

    # (B,n) @ (n,n) -> (B,n): y[b,i] = sum_j X[b,j] * W_local[i,j]
    X_local_avg = X @ W_local
    return (1.0 - alpha) * X + alpha * X_local_avg

def make_synth_blocks_spatial_storms(
    B: int = 300,
    grid_shape: tuple[int, int] = (12, 12),
    year_start: int = 1950,

    # Event / trigger model
    p_event: float = 1.0,                # probability the year has a storm at all
    p_min: float = 0.02,                 # baseline annual trigger prob far from storm
    p_max: float = 0.30,                 # near-center annual trigger prob (before heterogeneity)
    storm_lengthscale: float = 2.0,      # in grid-cell units; smaller => tighter cluster

    # Peer heterogeneity (optional)
    heterogeneity: float = 0.6,          # 0 = none; larger => wider spread in baseline risk
    risk_clip: tuple[float, float] = (0.01, 0.99),  # clamp after sigmoid if used

    # Exposures and severity
    exposure_lognormal_sigma: float = 0.5,
    normalize_exposure_mean1: bool = True,
    severity: str = "pareto",            # "pareto" or "lognormal"
    pareto_alpha: float = 1.8,
    pareto_scale: float = 1.0,
    lognorm_mu: float = 0.0,
    lognorm_sigma: float = 1.0,

    # Additional noise (optional)
    idio_severity_noise: float = 0.0,    # multiplies each peer payout by lognormal noise, 0 = none

    seed: int = 0,
):
    """
    Spatially-clustered DGP for parametric insurance blocks.
    Each year:
      - with prob p_event: sample a storm center on the grid
      - each peer's trigger prob increases as exp(-distance/lengthscale)
      - triggers are Bernoulli; payout = trigger * (severity S_b) * exposure E_i * idio_noise

    Returns:
      X_np: (B, n) float32
      years: (B,) int
      grid_shape: (H,W)
      peer_coords: (n,2) float (row,col)
    """
    rng = np.random.default_rng(seed)
    H, W = grid_shape
    n = H * W

    # grid coordinates in "cell units"
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([rr.reshape(-1), cc.reshape(-1)], axis=1).astype(np.float32)  # (n,2)

    # peer exposures
    E = rng.lognormal(mean=0.0, sigma=exposure_lognormal_sigma, size=n).astype(np.float32)
    if normalize_exposure_mean1:
        E = (E / float(E.mean())).astype(np.float32)

    # peer risk scores (heterogeneity) -> multiplicative factor on odds via a sigmoid model
    # We'll add a peer-specific log-odds shift; set to 0 if heterogeneity=0.
    if heterogeneity > 0:
        r = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float32)
        beta_peer = heterogeneity * r
    else:
        beta_peer = np.zeros(n, dtype=np.float32)

    years = np.arange(year_start, year_start + B, dtype=int)

    X = np.zeros((B, n), dtype=np.float32)

    # helpers
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def sample_severity():
        if severity == "pareto":
            # Pareto with tail index pareto_alpha and scale pareto_scale
            # If U ~ Unif(0,1), then S = scale * (1-U)^(-1/alpha)
            U = rng.random()
            return float(pareto_scale * (1.0 - U) ** (-1.0 / pareto_alpha))
        elif severity == "lognormal":
            return float(rng.lognormal(mean=lognorm_mu, sigma=lognorm_sigma))
        else:
            raise ValueError("severity must be 'pareto' or 'lognormal'")

    for b in range(B):
        if rng.random() > p_event:
            continue  # all zeros year

        # storm center uniform on grid
        center_r = rng.integers(0, H)
        center_c = rng.integers(0, W)
        center = np.array([center_r, center_c], dtype=np.float32)

        # distance from center (Euclidean in grid units)
        d = np.sqrt(((coords - center[None, :]) ** 2).sum(axis=1))  # (n,)

        # base storm-shaped trigger probability (before heterogeneity)
        # p(d) = p_min + (p_max - p_min) * exp(-d / L)
        p_base = p_min + (p_max - p_min) * np.exp(-d / float(storm_lengthscale))

        # add peer heterogeneity by shifting log-odds
        # logit(p) + beta_peer then back through sigmoid
        p = sigmoid(np.log(p_base / np.maximum(1.0 - p_base, 1e-12)) + beta_peer)
        p = np.clip(p, risk_clip[0], risk_clip[1]).astype(np.float32)

        # sample triggers
        hit = (rng.random(n) < p).astype(np.float32)

        # severity for the year
        S_b = sample_severity()

        # optional idiosyncratic payout noise per peer
        if idio_severity_noise > 0:
            noise = rng.lognormal(mean=0.0, sigma=idio_severity_noise, size=n).astype(np.float32)
        else:
            noise = 1.0

        X[b, :] = hit * (S_b * E) * noise

    return X.astype(np.float32), years, grid_shape, coords


# --- quick diagnostics helper (same interface you already use) ---
def peer_hit_rates(X_np: np.ndarray) -> tuple[float, float, float, float]:
    """Interprets hit as X>0 and returns avg/min/median/max annual trigger rate across peers."""
    hit = (X_np > 0).astype(np.float32)
    rates = hit.mean(axis=0)  # per peer
    return float(rates.mean()), float(rates.min()), float(np.median(rates)), float(rates.max())

def plot_certified_relief_scatter(
    c0: np.ndarray,
    c_alt: np.ndarray,
    delta: float,
    eta: float = 0.0,
    top_frac: float = 0.10,
    title_prefix: str = "CSAN",
    out_dir: str = "results_coverage",
    fname: str = "csan_relief_scatter.png",
    annotate: bool = True,
):
    """
    Scatter: alternative certified caps vs baseline caps.
    Highlights top 'top_frac' peers by baseline cap.
    Uses operational caps by default if you pass c_alt=c_op.
    """
    c0 = np.asarray(c0, dtype=float)
    c_alt = np.asarray(c_alt, dtype=float)
    assert c0.shape == c_alt.shape

    n = c0.size
    k_top = max(1, int(np.ceil(top_frac * n)))
    top_idx = np.argpartition(c0, n - k_top)[n - k_top:]
    mask_top = np.zeros(n, dtype=bool)
    mask_top[top_idx] = True
    mask_other = ~mask_top

    # Diagnostics for annotations
    ratio_top = np.divide(c_alt[mask_top], np.maximum(c0[mask_top], 1e-12))
    ratio_all = np.divide(c_alt, np.maximum(c0, 1e-12))
    frac_help_top = float(np.mean(c_alt[mask_top] < c0[mask_top]))
    frac_help_all = float(np.mean(c_alt < c0))

    # Plot
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7.5, 6.0))

    plt.scatter(c0[mask_other], c_alt[mask_other], s=25, alpha=0.5, label="others")
    plt.scatter(c0[mask_top], c_alt[mask_top], s=35, alpha=0.8, label=f"top {int(100*top_frac)}% risk")

    mx = float(max(c0.max(), c_alt.max()))
    plt.plot([0, mx], [0, mx], linewidth=2)  # y=x

    # Optional materiality line y=x+eta (helps interpret harm definition)
    if eta > 0:
        plt.plot([0, mx], [eta, mx + eta], linestyle="--", linewidth=1.5, label=r"$y=x+\eta$")

    plt.title(f"{title_prefix}: certified caps vs baseline (nominal {1-delta:.2f})")
    plt.xlabel(r"baseline certified cap $c_0$")
    plt.ylabel(r"alternative certified cap $c$")

    if annotate:
        text = (
            f"Top{int(100*top_frac)}% median ratio: {np.median(ratio_top):.3f}\n"
            f"All median ratio: {np.median(ratio_all):.3f}\n"
            f"Frac helped (top): {frac_help_top:.3f}\n"
            f"Frac helped (all): {frac_help_all:.3f}"
        )
        plt.gca().text(
            0.02, 0.98, text, transform=plt.gca().transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )

    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=300)
    plt.show()
    return out_path

def per_peer_coverage_across_all_tests(hit_tensor_list: list[np.ndarray]) -> np.ndarray:
    """
    hit_tensor_list: list of (nTest, n) indicator arrays
    returns per-peer coverage aggregated across all splits and test blocks: (n,)
    """
    all_hits = np.concatenate(hit_tensor_list, axis=0)  # (total_test_blocks, n)
    return all_hits.mean(axis=0)

def summarize_peer_coverage(peer_cov: np.ndarray, nominal: float) -> dict:
    peer_cov = np.asarray(peer_cov, dtype=np.float64)
    return {
        "mean": float(peer_cov.mean()),
        "median": float(np.median(peer_cov)),
        "p10": float(np.quantile(peer_cov, 0.10)),
        "p05": float(np.quantile(peer_cov, 0.05)),
        "min": float(peer_cov.min()),
        "frac_below_nominal": float(np.mean(peer_cov < nominal)),
    }

def worst_peer_shortfall(peer_cov: np.ndarray, nominal: float) -> dict:
    """
    How far below nominal are the worst peers?
    Returns worst (min) coverage and its shortfall from nominal (clipped at 0).
    Also returns p05 shortfall (often nicer than min).
    """
    peer_cov = np.asarray(peer_cov, dtype=float)
    worst = float(np.min(peer_cov))
    p05 = float(np.quantile(peer_cov, 0.05))
    shortfall_worst = float(max(0.0, nominal - worst))
    shortfall_p05 = float(max(0.0, nominal - p05))
    return {
        "worst": worst,
        "p05": p05,
        "shortfall_worst": shortfall_worst,
        "shortfall_p05": shortfall_p05,
    }

def _safe_ratio(num: float, den: float) -> float:
    return (num / den) if den > 0.0 else np.nan

# ============================================================
# 3) Validity-first backtest runner + plots (updated)
# ============================================================

import os

def apply_cap_floor(c: np.ndarray, c_min: float) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    if c_min is None or c_min <= 0.0:
        return c
    return np.maximum(c, np.float32(c_min))

def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    den2 = den if den > eps else np.nan
    return float(num / den2)


# --- helper: choose global vs local transform ---
def apply_pooling(X: np.ndarray, alpha: float, W_local: np.ndarray | None) -> np.ndarray:
    """
    If W_local is None -> global uniform pooling.
    Else -> local pooling with row-stochastic W_local.
    """
    if W_local is None:
        return apply_uniform_pooling(X, float(alpha))
    return apply_local_pooling(X, float(alpha), W_local)


def build_variance_optimal_Abar_doubly_stochastic(
    XT: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,   # (n,n) bool, True = allowed
    lam_diag: float = 0.1,
    ridge: float = 1e-6,
    n_iter: int = 500,
    lr: float = 0.5,
    sinkhorn_iter: int = 200,
    temperature: float = 1.0,
    l2_to_uniform: float = 0.0,              # optional stabilizer
    seed: int = 0,
) -> np.ndarray:
    """
    Variance-optimal baseline under a doubly-stochastic constraint (fair pooling):
        minimize tr(A Sigma A^T)  s.t. A >= 0, A 1 = 1, A^T 1 = 1
    Uses differentiable Sinkhorn normalization (KL projection) each step.

    Returns:
        A_bar: (n,n) float32 doubly-stochastic (up to numerical tolerance)
    """
    import torch

    Sigma = _cov_shrinkage(XT, lam_diag=lam_diag, ridge=ridge)  # (n,n) float64
    n = Sigma.shape[0]

    torch.manual_seed(seed)
    device = torch.device("cpu")
    dtype = torch.float64

    Sigma_t = torch.tensor(Sigma, dtype=dtype, device=device)

    if support_mask is None:
        mask_t = torch.ones((n, n), dtype=torch.bool, device=device)
    else:
        mask_t = torch.tensor(np.asarray(support_mask, dtype=bool), dtype=torch.bool, device=device)
        assert mask_t.shape == (n, n)

        # Basic feasibility check: each row/col must have at least one allowed entry
        if torch.any(mask_t.sum(dim=1) == 0) or torch.any(mask_t.sum(dim=0) == 0):
            raise ValueError("support_mask is infeasible for doubly-stochastic: some row/col has zero allowed entries.")

    # logits parameterization for positivity; masked entries are forced to -inf-like
    # start near-uniform on allowed entries
    logits = torch.zeros((n, n), dtype=dtype, device=device)
    logits = logits + .1 * torch.randn_like(logits)
    logits = logits.masked_fill(~mask_t, -1e9)  # effectively zero after exp
    logits.requires_grad_(True) 

    # Precompute a "uniform over allowed support" matrix for optional regularization
    if l2_to_uniform > 0:
        U = torch.zeros((n, n), dtype=dtype, device=device)
        # uniform per row over allowed entries, then Sinkhorn to make it doubly stochastic-ish
        row_counts = mask_t.sum(dim=1, keepdim=True).clamp_min(1)
        U = mask_t.to(dtype) / row_counts
        U = _sinkhorn_project(U, mask_t=mask_t, n_iter=sinkhorn_iter)
    else:
        U = None

    for _ in range(n_iter):
        # Positive matrix on allowed entries
        K = torch.exp(logits / float(temperature)) * mask_t.to(dtype)

        # Project to doubly-stochastic via Sinkhorn
        A = _sinkhorn_project(K, mask_t=mask_t, n_iter=sinkhorn_iter)

        # Objective: tr(A Sigma A^T) = sum_i a_i^T Sigma a_i
        obj = torch.trace(A.t() @ Sigma_t @ A)

        # Optional: discourage extreme concentration even within DS set
        if l2_to_uniform > 0:
            obj = obj + float(l2_to_uniform) * torch.sum((A - U) ** 2)

        # Gradient step on logits
        obj.backward()

        with torch.no_grad():
            logits -= float(lr) * logits.grad
            logits.grad.zero_()
            # keep masked entries pinned
            logits.masked_fill_(~mask_t, -1e9)


        # # --- after logits is initialized, and after you construct initial A via Sinkhorn ---
        # with torch.no_grad():
        #     K0 = torch.exp(logits / temperature)
        #     K0 = K0 * mask_t.to(dtype=dtype)
        #     A0 = _sinkhorn_project(K0, mask_t=mask_t, n_iter=sinkhorn_iter)

        #     U = torch.full((n, n), 1.0 / n, dtype=dtype, device=device)
        #     if support_mask is not None:
        #         # only compare to dense U if you want a rough reference; otherwise skip/replace
        #         pass

        #     J_A0 = torch.trace(A0.t() @ Sigma_t @ A0).item()   # use your chosen convention
        #     J_U  = torch.trace(U.t() @ Sigma_t @ U).item()     # same convention
        #     fro_to_U = torch.norm(A0 - U, p='fro').item()

        #     print("[VO-DS init] J(U)={:.6g} J(A0)={:.6g} ||A0-U||_F={:.6g}".format(J_U, J_A0, fro_to_U))
        #     print("[VO-DS init] row sums min/med/max = {:.6f}/{:.6f}/{:.6f}".format(
        #         A0.sum(dim=1).min().item(), A0.sum(dim=1).median().item(), A0.sum(dim=1).max().item()))
        #     print("[VO-DS init] col sums min/med/max = {:.6f}/{:.6f}/{:.6f}".format(
        #         A0.sum(dim=0).min().item(), A0.sum(dim=0).median().item(), A0.sum(dim=0).max().item()))
        #     print("[VO-DS init] A0 min/max = {:.6g}/{:.6g}".format(A0.min().item(), A0.max().item()))



    # Final A
    K = torch.exp(logits / float(temperature)) * mask_t.to(dtype)
    A = _sinkhorn_project(K, mask_t=mask_t, n_iter=sinkhorn_iter)

    A_np = A.detach().cpu().numpy().astype(np.float32)
    return A_np


def _sinkhorn_project(M: "torch.Tensor", *, mask_t: "torch.Tensor", n_iter: int = 200) -> "torch.Tensor":
    """
    Sinkhorn-Knopp scaling to make M approximately doubly-stochastic,
    respecting a boolean support mask (masked entries stay 0).
    This is KL projection, not Euclidean.
    """
    import torch

    dtype = M.dtype
    eps = torch.tensor(1e-12, dtype=dtype, device=M.device)

    X = M * mask_t.to(dtype)

    # Initialize scaling
    r = torch.ones((X.shape[0],), dtype=dtype, device=X.device)
    c = torch.ones((X.shape[1],), dtype=dtype, device=X.device)

    for _ in range(n_iter):
        # Row normalize
        row_sums = (X @ c).clamp_min(eps)          # (n,)
        r = 1.0 / row_sums
        # Col normalize
        col_sums = (X.t() @ r).clamp_min(eps)      # (n,)
        c = 1.0 / col_sums

    # Apply scaling: diag(r) X diag(c)
    A = (r[:, None] * X) * c[None, :]
    # Guarantee masked zeros
    A = A * mask_t.to(dtype)
    return A

def validity_first_backtest(
    X_np: np.ndarray,
    years: np.ndarray,
    delta: float = 0.1,
    eta: float = 0.0,
    eps: float = 0.05,
    w: np.ndarray | None = None,
    alphas: np.ndarray | None = None,
    # rolling windows:
    nT: int = 25,
    nV: int = 10,
    nC: int = 10,
    nTest: int = 1,
    step: int = 1,
    split_mode: str = "time",   # "time" or "random"
    n_splits: int = 30,         # used only for random
    seed: int = 0,
    make_plots: bool = True,
    out_dir: str = "results_coverage",
    W_local: np.ndarray | None = None,
    # -------------------------
    # NEW: variance-optimal baseline knobs (build Abar on T only)
    # -------------------------
    use_vo_Abar: bool = False,
    vo_use_local_support: bool = True,  # if True and W_local is provided, restrict row supports to W_local>0
    vo_lam_diag: float = 0.1,
    vo_ridge: float = 1e-6,
    vo_n_iter: int = 1000,
    vo_lr: float = 0.05,
    vo_seed: int | None = None,
    vo_use_torch: bool = True,
    # plotting: make just one relief-scatter (avoid writing dozens)
    relief_scatter: bool = False,
    relief_scatter_pick: str = "first_pass",  # "first_pass" | "median_top10" | "best_top10"
):
    """
    Validity-first backtest supporting:
      - global pooling (W_local=None) OR local pooling (W_local is row-stochastic n×n)
      - candidate-only vs operational vs identity coverage
      - theorem-aligned per-peer marginal coverage across all test blocks & splits
      - certified cap efficiency + targeted utility metrics
    """
    os.makedirs(out_dir, exist_ok=True)

    X_np = np.asarray(X_np, dtype=np.float32)
    years = np.asarray(years)
    B, n = X_np.shape

    # sanity on W_local
    if W_local is not None:
        W_local = np.asarray(W_local, dtype=np.float32)
        if W_local.shape != (n, n):
            raise ValueError(f"W_local must have shape (n,n)=({n},{n}), got {W_local.shape}")
        # optional check row-stochastic (tolerant)
        rs = W_local.sum(axis=1)
        if not np.allclose(rs, 1.0, atol=1e-3):
            raise ValueError("W_local rows must sum to 1 (row-stochastic).")

    if w is None:
        w = np.ones(n, dtype=np.float32) / n
    else:
        w = np.asarray(w, dtype=np.float32)
        w = w / float(w.sum())

    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    else:
        alphas = np.asarray(alphas, dtype=np.float32)

    if split_mode == "time":
        splits = rolling_backtest_splits(years, nT=nT, nV=nV, nC=nC, nTest=nTest, step=step)
        x_label = "test year"
        title_suffix = "(time-ordered)"
    elif split_mode == "random":
        splits = random_backtest_splits(
            years, nT=nT, nV=nV, nC=nC, nTest=nTest,
            n_splits=n_splits, seed=seed
        )
        x_label = "random split id"
        title_suffix = "(random splits)"
    else:
        raise ValueError("split_mode must be 'time' or 'random'")

    if len(splits) < 3:
        print(f"Warning: only {len(splits)} backtest splits.")

    nominal = 1.0 - delta

    # -------------------------
    # Storage (split-level)
    # -------------------------
    x_axis = []
    status_list = []
    alpha_star_list, alpha_op_list = [], []
    harm_cert_list, H_cert_list = [], []

    # certified caps
    cap_wsum_id_list = []
    cap_wsum_candidate_list = []
    cap_wsum_oper_list = []
    cap_ratio_candidate_list = []
    cap_ratio_oper_list = []

    # targeted utility
    top10_ratio_candidate_list = []
    top10_ratio_oper_list = []
    maxcap_ratio_candidate_list = []
    maxcap_ratio_oper_list = []
    frac_help_candidate_list = []
    frac_help_oper_list = []

    # split-level mean coverages (not theorem-aligned)
    cov_mean_candidate_list, cov_mean_oper_list, cov_mean_id_list = [], [], []
    cov_wmean_candidate_list, cov_wmean_oper_list, cov_wmean_id_list = [], [], []

    # store hit tensors to aggregate per-peer coverage across all tests
    hits_candidate_all, hits_oper_all, hits_id_all = [], [], []

    # for picking one representative relief-scatter
    relief_records = []  # (split_idx, top10_ratio_oper, status, alpha_op, c0, c_op)

    # sets a minimum cap floor equal to x% of the median nonzero payout size in your dataset
    x_nonzero = X_np[X_np > 0]
    c_min = 0.01 * float(np.quantile(x_nonzero, 0.5)) if x_nonzero.size else 0.0

    vo_dbg = {
        "J_I": [],          # objective for identity
        "J_U": [],          # objective for uniform pooling
        "J_Abar": [],       # objective for A_bar
        "J_Aalpha_grid": [],# objectives for A(alpha) over your alphas grid
    }
    # Optional: keep A_bar for a few splits for plotting later
    Abar_snapshots = []    # list of (split_j, A_bar)

    for split_j, s in enumerate(splits):
        # -------------------------
        # Split data
        # -------------------------
        XT = X_np[s["T"]]
        XV = X_np[s["V"]]
        XC = X_np[s["C"]]
        XTest = X_np[s["Test"]]

        # -------------------------
        # NEW: Build variance-optimal A_bar on T only (optional)
        # -------------------------
        A_bar = None
        if use_vo_Abar:
            # optional locality restriction: only allow sharing where W_local has support
            support_mask = None
            if vo_use_local_support and (W_local is not None):
                support_mask = (np.asarray(W_local) > 0)

            # seed control: default to split-specific deterministic seed if not provided
            _seed = seed if vo_seed is None else vo_seed
            _seed = int(_seed + 10007 * split_j)

            A_bar = build_variance_optimal_Abar_doubly_stochastic(
                XT,
                support_mask=support_mask,
                lam_diag=vo_lam_diag,
                ridge=vo_ridge,
                n_iter=vo_n_iter,
                lr=vo_lr,
                sinkhorn_iter=50,
                temperature=1.0,      # neutral; don't force sparsity
                l2_to_uniform=0.0,
                seed=seed + 10007 * split_j,
            )
            # A_bar = build_variance_optimal_Abar(
            #     XT,
            #     support_mask=support_mask,
            #     lam_diag=vo_lam_diag,
            #     ridge=vo_ridge,
            #     n_iter=vo_n_iter,
            #     lr=vo_lr,
            #     seed=seed + 10007 * split_j,
            # )

            rs = A_bar.sum(axis=1)
            cs = A_bar.sum(axis=0)
            print("row sums:", rs.min(), rs.mean(), rs.max())
            print("col sums:", cs.min(), cs.mean(), cs.max())
            print("min entry:", A_bar.min(), "max entry:", A_bar.max())

            I = np.eye(XT.shape[1], dtype=np.float32)
            U = np.ones((XT.shape[1],XT.shape[1]), dtype=np.float32) / n

            # scalar objectives on training covariance
            J_I = variance_objective(I, XT, lam_diag=vo_lam_diag, ridge=vo_ridge)
            J_U = variance_objective(U, XT, lam_diag=vo_lam_diag, ridge=vo_ridge)
            J_Abar = variance_objective(A_bar, XT, lam_diag=vo_lam_diag, ridge=vo_ridge)

            # objectives along the same alpha grid you will validate over
            J_grid = []
            for a in alphas:
                a = float(a)
                A = (1.0 - a) * I + a * A_bar
                J_grid.append(variance_objective(A, XT, lam_diag=vo_lam_diag, ridge=vo_ridge))
            J_grid = np.asarray(J_grid, dtype=np.float64)

            vo_dbg["J_I"].append(J_I)
            vo_dbg["J_U"].append(J_U)
            vo_dbg["J_Abar"].append(J_Abar)
            vo_dbg["J_Aalpha_grid"].append(J_grid)

            # print occasionally (so logs don't explode)
            if (split_j % 10) == 0:
                print(f"[split {split_j}] VO objective: J(I)={J_I:.4g} J(U)={J_U:.4g} J(A_bar)={J_Abar:.4g}  "
                    f"min_J_over_alphas={J_grid.min():.4g} at alpha={float(alphas[J_grid.argmin()]):.2f}")

            # keep a few matrices for later visualization (optional)
            if split_j in (0, len(splits)//2, len(splits)-1):
                Abar_snapshots.append((split_j, A_bar.copy()))

        # -------------------------
        # Select alpha on V (same as before, but now pass A_bar)
        # -------------------------
        alpha_star, _ = csan_select_alpha_on_V(
            XV=XV, delta=delta, eta=eta, eps=eps, w=w, alphas=alphas,
            W_local=W_local,
            A_bar=A_bar,          
        )

        # -------------------------
        # Certify on C (same as before, but now pass A_bar)
        # -------------------------
        cert = csan_certify_on_C(
            XC=XC, alpha_star=alpha_star, delta=delta, eta=eta, eps=eps, w=w,
            W_local=W_local, c_min=c_min,
            A_bar=A_bar,          
        )

        c0 = cert["c0"]
        c_star = cert["c_star"]
        c_op = cert["c_op"]
        alpha_op = cert["alpha_op"]

        # -------------------------
        # Evaluate on test blocks
        # -------------------------
        # candidate-only: alpha_star + c_star
        YTest_star = _apply_mechanism(XTest, float(alpha_star), W_local, A_bar=A_bar)
        hit_star = (YTest_star <= c_star[None, :]).astype(np.float32)

        # operational: alpha_op + c_op
        YTest_op = _apply_mechanism(XTest, float(alpha_op), W_local, A_bar=A_bar)
        hit_op = (YTest_op <= c_op[None, :]).astype(np.float32)

        # identity
        hit_id = (XTest <= c0[None, :]).astype(np.float32)

        # split-level peer coverages
        cov_peer_star = hit_star.mean(axis=0)
        cov_peer_op = hit_op.mean(axis=0)
        cov_peer_id = hit_id.mean(axis=0)

        cov_mean_candidate_list.append(float(cov_peer_star.mean()))
        cov_mean_oper_list.append(float(cov_peer_op.mean()))
        cov_mean_id_list.append(float(cov_peer_id.mean()))

        cov_wmean_candidate_list.append(float(np.dot(w, cov_peer_star)))
        cov_wmean_oper_list.append(float(np.dot(w, cov_peer_op)))
        cov_wmean_id_list.append(float(np.dot(w, cov_peer_id)))

        hits_candidate_all.append(hit_star)
        hits_oper_all.append(hit_op)
        hits_id_all.append(hit_id)

        # meta
        status_list.append(cert["status"])
        alpha_star_list.append(alpha_star)
        alpha_op_list.append(alpha_op)
        harm_cert_list.append(cert["Harm_cert"])
        H_cert_list.append(cert["H_cert"])

        # -------------------------
        # Certified cap efficiency
        # -------------------------
        cap0 = float(np.dot(w, c0))
        cap_star_w = float(np.dot(w, c_star))
        cap_op_w = float(np.dot(w, c_op))

        cap_wsum_id_list.append(cap0)
        cap_wsum_candidate_list.append(cap_star_w)
        cap_wsum_oper_list.append(cap_op_w)

        if cap0 > 0.0:
            cap_ratio_candidate_list.append(cap_star_w / cap0)
            cap_ratio_oper_list.append(cap_op_w / cap0)
        else:
            cap_ratio_candidate_list.append(np.nan)
            cap_ratio_oper_list.append(np.nan)

        # -------------------------
        # Targeted utility metrics
        # -------------------------
        k_top = max(1, int(np.ceil(0.10 * n)))
        top_idx = np.argpartition(c0, n - k_top)[n - k_top:]

        top10_ratio_candidate_list.append(
            _safe_ratio(float(np.dot(w[top_idx], c_star[top_idx])),
                        float(np.dot(w[top_idx], c0[top_idx])))
        )
        top10_ratio_oper_list.append(
            _safe_ratio(float(np.dot(w[top_idx], c_op[top_idx])),
                        float(np.dot(w[top_idx], c0[top_idx])))
        )

        maxcap_ratio_candidate_list.append(_safe_ratio(float(np.max(c_star)), float(np.max(c0))))
        maxcap_ratio_oper_list.append(_safe_ratio(float(np.max(c_op)), float(np.max(c0))))

        frac_help_candidate_list.append(float(np.mean(c_star < c0)))
        frac_help_oper_list.append(float(np.mean(c_op < c0)))

        # save record for optional single scatter plot
        relief_records.append({
            "split_j": split_j,
            "x": s["split_id"] if split_mode == "random" else int(np.min(s["years_Test"])),
            "status": cert["status"],
            "alpha_star": float(alpha_star),
            "alpha_op": float(alpha_op),
            "top10_ratio_oper": float(top10_ratio_oper_list[-1]),
            "c0": c0.copy(),
            "c_op": c_op.copy(),
        })

        if split_mode == "random":
            x_axis.append(s["split_id"])
        else:
            x_axis.append(int(np.min(s["years_Test"])))

    # -------------------------
    # Convert to arrays
    # -------------------------
    x_axis = np.array(x_axis)
    alpha_star_list = np.array(alpha_star_list)
    alpha_op_list = np.array(alpha_op_list)
    harm_cert_list = np.array(harm_cert_list)
    H_cert_list = np.array(H_cert_list)

    cov_mean_candidate_list = np.array(cov_mean_candidate_list)
    cov_mean_oper_list = np.array(cov_mean_oper_list)
    cov_mean_id_list = np.array(cov_mean_id_list)

    cov_wmean_candidate_list = np.array(cov_wmean_candidate_list)
    cov_wmean_oper_list = np.array(cov_wmean_oper_list)
    cov_wmean_id_list = np.array(cov_wmean_id_list)

    cap_wsum_id_list = np.array(cap_wsum_id_list, dtype=float)
    cap_wsum_candidate_list = np.array(cap_wsum_candidate_list, dtype=float)
    cap_wsum_oper_list = np.array(cap_wsum_oper_list, dtype=float)
    cap_ratio_candidate_list = np.array(cap_ratio_candidate_list, dtype=float)
    cap_ratio_oper_list = np.array(cap_ratio_oper_list, dtype=float)

    top10_ratio_candidate_list = np.array(top10_ratio_candidate_list, dtype=float)
    top10_ratio_oper_list = np.array(top10_ratio_oper_list, dtype=float)
    maxcap_ratio_candidate_list = np.array(maxcap_ratio_candidate_list, dtype=float)
    maxcap_ratio_oper_list = np.array(maxcap_ratio_oper_list, dtype=float)
    frac_help_candidate_list = np.array(frac_help_candidate_list, dtype=float)
    frac_help_oper_list = np.array(frac_help_oper_list, dtype=float)

    S = len(x_axis)
    pass_rate = float(np.mean([st == "PASS" for st in status_list])) if S else np.nan

    # theorem-aligned per-peer coverage across all tests
    peer_cov_candidate = per_peer_coverage_across_all_tests(hits_candidate_all)
    peer_cov_oper = per_peer_coverage_across_all_tests(hits_oper_all)
    peer_cov_id = per_peer_coverage_across_all_tests(hits_id_all)

    worst_candidate = worst_peer_shortfall(peer_cov_candidate, nominal)
    worst_oper = worst_peer_shortfall(peer_cov_oper, nominal)
    worst_id = worst_peer_shortfall(peer_cov_id, nominal)

    summ_candidate = summarize_peer_coverage(peer_cov_candidate, nominal)
    summ_oper = summarize_peer_coverage(peer_cov_oper, nominal)
    summ_id = summarize_peer_coverage(peer_cov_id, nominal)

    # -------------------------
    # Print summary
    # -------------------------
    def _nanmean(x): return float(np.nanmean(x))
    def _nanstd(x):  return float(np.nanstd(x))

    alpha_star = np.array([r.get("alpha_star", np.nan) for r in relief_records], dtype=float)
    alpha_op   = np.array([r.get("alpha_op",   np.nan) for r in relief_records], dtype=float)


    pool_name = "global-uniform" if W_local is None else "local"
    print("=== Validity-first backtest (updated) ===")
    print(f"Pooling: {pool_name}")
    print(f"Mode: {split_mode} | Splits: {S} | nTest={nTest} | nC={nC}")
    print(f"Nominal coverage: {nominal:.3f}")
    print(f"PASS rate: {pass_rate:.3f}")
    print("")
    print("Split-level mean coverage across peers (not theorem-aligned, but intuitive):")
    print(f"  candidate : mean={cov_mean_candidate_list.mean():.3f}, std={cov_mean_candidate_list.std():.3f}")
    print(f"  operational: mean={cov_mean_oper_list.mean():.3f}, std={cov_mean_oper_list.std():.3f}")
    print(f"  identity  : mean={cov_mean_id_list.mean():.3f}, std={cov_mean_id_list.std():.3f}")
    print("")
    print("Worst-peer shortfall from nominal (theorem-aligned empirical):")
    print(f"  candidate   : worst={worst_candidate['worst']:.3f} (shortfall={worst_candidate['shortfall_worst']:.3f}), "
          f"p05={worst_candidate['p05']:.3f} (shortfall={worst_candidate['shortfall_p05']:.3f})")
    print(f"  operational : worst={worst_oper['worst']:.3f} (shortfall={worst_oper['shortfall_worst']:.3f}), "
          f"p05={worst_oper['p05']:.3f} (shortfall={worst_oper['shortfall_p05']:.3f})")
    print(f"  identity    : worst={worst_id['worst']:.3f} (shortfall={worst_id['shortfall_worst']:.3f}), "
          f"p05={worst_id['p05']:.3f} (shortfall={worst_id['shortfall_p05']:.3f})")
    print("")
    print("Certified cap efficiency (lower is better):")
    print(f"  mean <w,c0> (identity)     : {_nanmean(cap_wsum_id_list):.3g} ± {_nanstd(cap_wsum_id_list):.3g}")
    print(f"  mean <w,c*> (candidate)    : {_nanmean(cap_wsum_candidate_list):.3g} ± {_nanstd(cap_wsum_candidate_list):.3g}")
    print(f"  mean <w,c_op> (operational): {_nanmean(cap_wsum_oper_list):.3g} ± {_nanstd(cap_wsum_oper_list):.3g}")
    print(f"  AggCapRatio candidate      : {_nanmean(cap_ratio_candidate_list):.3f} ± {_nanstd(cap_ratio_candidate_list):.3f}")
    print(f"  AggCapRatio operational    : {_nanmean(cap_ratio_oper_list):.3f} ± {_nanstd(cap_ratio_oper_list):.3f}")
    print("")
    print("Additional certified utility metrics (lower ratios are better):")
    print(f"  Top10 cap ratio (candidate)   : {_nanmean(top10_ratio_candidate_list):.3f} ± {_nanstd(top10_ratio_candidate_list):.3f}")
    print(f"  Top10 cap ratio (operational) : {_nanmean(top10_ratio_oper_list):.3f} ± {_nanstd(top10_ratio_oper_list):.3f}")
    print(f"  Max-cap ratio (candidate)     : {_nanmean(maxcap_ratio_candidate_list):.3f} ± {_nanstd(maxcap_ratio_candidate_list):.3f}")
    print(f"  Max-cap ratio (operational)   : {_nanmean(maxcap_ratio_oper_list):.3f} ± {_nanstd(maxcap_ratio_oper_list):.3f}")
    print(f"  Frac helped (candidate)       : {_nanmean(frac_help_candidate_list):.3f} ± {_nanstd(frac_help_candidate_list):.3f}")
    print(f"  Frac helped (operational)     : {_nanmean(frac_help_oper_list):.3f} ± {_nanstd(frac_help_oper_list):.3f}")
    print("")
    print("Per-peer marginal coverage aggregated across all test blocks & splits (theorem-aligned):")
    print(f"  candidate : mean={summ_candidate['mean']:.3f}, median={summ_candidate['median']:.3f}, "
          f"p10={summ_candidate['p10']:.3f}, p05={summ_candidate['p05']:.3f}, "
          f"min={summ_candidate['min']:.3f}, frac<nominal={summ_candidate['frac_below_nominal']:.3f}")
    print(f"  operational: mean={summ_oper['mean']:.3f}, median={summ_oper['median']:.3f}, "
          f"p10={summ_oper['p10']:.3f}, p05={summ_oper['p05']:.3f}, "
          f"min={summ_oper['min']:.3f}, frac<nominal={summ_oper['frac_below_nominal']:.3f}")
    print(f"  identity  : mean={summ_id['mean']:.3f}, median={summ_id['median']:.3f}, "
          f"p10={summ_id['p10']:.3f}, p05={summ_id['p05']:.3f}, "
          f"min={summ_id['min']:.3f}, frac<nominal={summ_id['frac_below_nominal']:.3f}")
    print("")
    print("\nAlpha (per split):")
    print(f"  alpha_star mean±std: {np.nanmean(alpha_star):.3f} ± {np.nanstd(alpha_star):.3f}")
    print(f"  alpha_op   mean±std: {np.nanmean(alpha_op):.3f} ± {np.nanstd(alpha_op):.3f} "
        f"(frac==0: {np.mean(alpha_op <= 1e-12):.3f})")
    print("")

    # -------------------------
    # Optional: write ONE relief-scatter (avoid spamming files)
    # -------------------------
    if relief_scatter and len(relief_records) > 0:
        # choose candidate record
        recs = relief_records
        pass_recs = [r for r in recs if r["status"] == "PASS" and r["alpha_op"] > 0.0]
        pick_pool = pass_recs if len(pass_recs) > 0 else recs

        if relief_scatter_pick == "first_pass":
            pick = pick_pool[0]
        elif relief_scatter_pick == "best_top10":
            pick = min(pick_pool, key=lambda r: r["top10_ratio_oper"])  # smallest is best
        elif relief_scatter_pick == "median_top10":
            vals = np.array([r["top10_ratio_oper"] for r in pick_pool], dtype=float)
            med = np.nanmedian(vals)
            pick = min(pick_pool, key=lambda r: abs(r["top10_ratio_oper"] - med))
        else:
            pick = pick_pool[0]

        print("pick c0: min/median/max", np.min(pick["c0"]), np.median(pick["c0"]), np.max(pick["c0"]))
        print("pick c_op: min/median/max", np.min(pick["c_op"]), np.median(pick["c_op"]), np.max(pick["c_op"]))

        plot_certified_relief_scatter(
            c0=pick["c0"],
            c_alt=pick["c_op"],
            delta=delta,
            eta=eta,
            top_frac=0.10,
            out_dir=out_dir,
            title_prefix=(f"CSAN {pool_name} ({split_mode}, "
              f"alpha_op={pick['alpha_op']:.2f}, status={pick['status']})"),
            fname=(f"csan_relief_scatter_{pool_name}_{split_mode}_"
                f"alphaop_{pick['alpha_op']:.2f}_delta_{delta}.png")
        )

    # -------------------------
    # Plots
    # -------------------------
    if make_plots:
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, cov_mean_oper_list, marker="o", label="Operational (mean across peers)")
        plt.plot(x_axis, cov_mean_candidate_list, marker="o", label="Candidate-only (mean across peers)")
        plt.plot(x_axis, cov_mean_id_list, marker="o", label="Identity vs identity caps")
        plt.axhline(nominal, linestyle="--", label="Nominal")
        plt.title(f"Out-of-sample coverage across splits {title_suffix}")
        plt.xlabel(x_label)
        plt.ylabel("coverage")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"coverage_over_splits_{pool_name}_{split_mode}_delta_{delta}.png"), dpi=300)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.hist(peer_cov_oper, bins=25, alpha=0.7, label="operational")
        plt.hist(peer_cov_candidate, bins=25, alpha=0.7, label="candidate-only")
        plt.axvline(nominal, linestyle="--")
        plt.title(f"Per-peer marginal coverage across all test blocks {title_suffix}")
        plt.xlabel("coverage (per peer)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"peer_coverage_hist_{pool_name}_{split_mode}_delta_{delta}.png"), dpi=300)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, alpha_star_list, marker="o", label="alpha* (selected on V)")
        plt.plot(x_axis, alpha_op_list, marker="o", label="alpha_op (after PASS/FAIL)")
        plt.title(f"Selected vs operational alpha {title_suffix}")
        plt.xlabel(x_label)
        plt.ylabel("alpha")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"alpha_over_splits_{pool_name}_{split_mode}_delta_{delta}.png"), dpi=300)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, harm_cert_list, marker="o", label="Certified harm")
        plt.plot(x_axis, H_cert_list, marker="o", label="Certified budget")
        plt.title(f"Certified harm vs budget (computed on calibration) {title_suffix}")
        plt.xlabel(x_label)
        plt.ylabel("value")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"harm_vs_budget_{pool_name}_{split_mode}_delta_{delta}.png"), dpi=300)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.hist(cov_mean_oper_list, bins=15, alpha=0.7, label="operational")
        plt.hist(cov_mean_candidate_list, bins=15, alpha=0.7, label="candidate-only")
        plt.axvline(nominal, linestyle="--")
        plt.title(f"Distribution of split-level mean coverages {title_suffix}")
        plt.xlabel("mean coverage across peers (per split)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"split_mean_cov_hist_{pool_name}_{split_mode}_delta_{delta}.png"), dpi=300)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, cap_ratio_candidate_list, marker="o", label="candidate ratio")
        plt.plot(x_axis, cap_ratio_oper_list, marker="o", label="operational ratio")
        plt.axhline(1.0, linestyle="--", label="identity = 1.0")
        plt.title(f"Aggregate certified cap ratio across splits {title_suffix}")
        plt.xlabel(x_label)
        plt.ylabel(r"$\langle w,c(A)\rangle / \langle w,c_0\rangle$")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"cap_ratio_{pool_name}_{split_mode}_delta_{delta}.png"), dpi=300)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, top10_ratio_candidate_list, marker="o", label="candidate top10 ratio")
        plt.plot(x_axis, top10_ratio_oper_list, marker="o", label="operational top10 ratio")
        plt.axhline(1.0, linestyle="--", label="identity = 1.0")
        plt.title(f"Top-decile certified cap ratio across splits {title_suffix}")
        plt.xlabel(x_label)
        plt.ylabel(r"$\langle w_H,c_H(A)\rangle / \langle w_H,c_H(c_0)\rangle$")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"top10_cap_ratio_{pool_name}_{split_mode}_delta_{delta}.png"), dpi=300)
        plt.show()

    results = {
        "mode": split_mode,
        "pooling": pool_name,
        "x_axis": x_axis,
        "status": np.array(status_list),
        "alpha_star": alpha_star_list,
        "alpha_op": alpha_op_list,
        "harm_cert": harm_cert_list,
        "H_cert": H_cert_list,
        # split-level means
        "cov_mean_candidate": cov_mean_candidate_list,
        "cov_mean_operational": cov_mean_oper_list,
        "cov_mean_identity": cov_mean_id_list,
        "cov_wmean_candidate": cov_wmean_candidate_list,
        "cov_wmean_operational": cov_wmean_oper_list,
        "cov_wmean_identity": cov_wmean_id_list,
        # theorem-aligned per-peer marginals
        "peer_cov_candidate": peer_cov_candidate,
        "peer_cov_operational": peer_cov_oper,
        "peer_cov_identity": peer_cov_id,
        "peer_cov_summary_candidate": summ_candidate,
        "peer_cov_summary_operational": summ_oper,
        "peer_cov_summary_identity": summ_id,
        "nominal": nominal,
        "splits": splits,
        "params": {"delta": delta, "eta": eta, "eps": eps, "nT": nT, "nV": nV, "nC": nC, "nTest": nTest},
        "worst_shortfall_candidate": worst_candidate,
        "worst_shortfall_operational": worst_oper,
        "worst_shortfall_identity": worst_id,
        # Certified-cap efficiency
        "cap_wsum_identity": cap_wsum_id_list,
        "cap_wsum_candidate": cap_wsum_candidate_list,
        "cap_wsum_operational": cap_wsum_oper_list,
        "cap_ratio_candidate": cap_ratio_candidate_list,
        "cap_ratio_operational": cap_ratio_oper_list,
        # Additional certified utility metrics
        "top10_ratio_candidate": top10_ratio_candidate_list,
        "top10_ratio_operational": top10_ratio_oper_list,
        "maxcap_ratio_candidate": maxcap_ratio_candidate_list,
        "maxcap_ratio_operational": maxcap_ratio_oper_list,
        "frac_help_candidate": frac_help_candidate_list,
        "frac_help_operational": frac_help_oper_list,
    }
    if A_bar is not None:
        results["A_bar"] = A_bar
    return results



def make_synth_blocks(
    B: int = 75,
    grid_shape: tuple[int, int] = (12, 12),   # n = 144 peers
    year_start: int = 1950,
    # rare-shock / discreteness controls
    p_event: float = 0.12,                    # probability a year is a "shock year"
    p_hit_given_event: float = 0.25,          # per-peer hit prob in a shock year
    # severity controls (heavy-tailed)
    severity: str = "pareto",                 # "pareto" or "lognormal"
    pareto_alpha: float = 2.0,                # tail index (>1 gives finite mean)
    pareto_scale: float = 1.0,
    lognorm_mu: float = 0.0,
    lognorm_sigma: float = 1.0,
    # heterogeneity across peers
    heterogeneity: float = 0.8,               # 0 = homogeneous, larger = more varied risk
    # dependence controls
    common_factor_strength: float = 0.7,      # 0 = independent peers, 1 = strong common shock
    spatial_factor_strength: float = 0.0,     # optional spatial correlation (0 to ~0.5)
    spatial_corr_len: float = 3.0,            # larger = smoother spatial factor
    seed: int = 0,
):
    """
    Returns (X_np, years, grid_shape, peer_coords) in the same format as eobs_to_blocks.

    Model:
      - Each year b: with prob p_event => "shock year", else mostly zeros.
      - In a shock year:
           each peer i hits with prob p_i = sigmoid(base + heterogeneity * risk_i + common_factor_strength * Z_b + spatial_factor_strength * S_{b,i})
           loss = hit * severity_b * exposure_i
      - This yields zero-inflated, discrete-ish block sums with optional dependence.

    Designed to stress-test conformal caps under:
      - many ties / zeros,
      - heavy tails,
      - common shocks (dependence).
    """
    rng = np.random.default_rng(seed)
    latN, lonN = grid_shape
    n = latN * lonN

    years = np.arange(year_start, year_start + B)

    # peer coordinates (for plotting)
    lats = np.linspace(0, latN - 1, latN)
    lons = np.linspace(0, lonN - 1, lonN)
    coords = np.array([(float(la), float(lo)) for la in lats for lo in lons], dtype=np.float32)

    # peer heterogeneity: exposure_i >0 and risk_i (propensity to hit)
    # exposures make some peers "bigger" insured values
    exposure = rng.lognormal(mean=0.0, sigma=0.5, size=n).astype(np.float32)
    exposure = exposure / exposure.mean()  # normalize

    # baseline risk score per peer (centered)
    risk = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    risk = (risk - risk.mean()) / (risk.std() + 1e-8)

    # Optional spatial factor: smooth random field per year
    # We'll generate it by filtering white noise in Fourier domain (cheap and simple).
    def smooth_field(shape, corr_len):
        wn = rng.normal(0.0, 1.0, size=shape)
        # FFT-based Gaussian smoothing
        ky = np.fft.fftfreq(shape[0])[:, None]
        kx = np.fft.fftfreq(shape[1])[None, :]
        k2 = kx**2 + ky**2
        # exp(-k^2 * sigma^2) smoother when corr_len larger
        sigma = corr_len / max(shape)
        filt = np.exp(-0.5 * (2*np.pi*sigma)**2 * k2)
        f = np.fft.ifft2(np.fft.fft2(wn) * filt).real
        f = (f - f.mean()) / (f.std() + 1e-8)
        return f.astype(np.float32)

    X = np.zeros((B, n), dtype=np.float32)

    # helper for severity draws
    def draw_severity(size):
        if severity == "pareto":
            # Pareto(alpha, scale): X = scale * (1 + Pareto(alpha))
            # numpy's pareto(a) has support [0, inf) with tail (1+x)^(-a)
            return (pareto_scale * (1.0 + rng.pareto(pareto_alpha, size=size))).astype(np.float32)
        elif severity == "lognormal":
            return rng.lognormal(mean=lognorm_mu, sigma=lognorm_sigma, size=size).astype(np.float32)
        else:
            raise ValueError("severity must be 'pareto' or 'lognormal'")

    # logistic link
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    for b in range(B):
        is_event = (rng.random() < p_event)
        if not is_event:
            # non-event year: keep mostly zeros (optional tiny background)
            continue

        # common factor for the year
        Zb = rng.normal(0.0, 1.0)

        # spatial factor field for the year (optional)
        if spatial_factor_strength > 0:
            Sb_grid = smooth_field((latN, lonN), spatial_corr_len)
            Sb = Sb_grid.reshape(-1)
        else:
            Sb = np.zeros(n, dtype=np.float32)

        # baseline logit to achieve roughly p_hit_given_event on average
        # We'll set base so that sigmoid(base) ~ p_hit_given_event
        base = np.log(p_hit_given_event / (1.0 - p_hit_given_event))

        # per-peer hit probabilities
        logits = (
            base
            + heterogeneity * risk
            + common_factor_strength * Zb
            + spatial_factor_strength * Sb
        )
        p_hit = sigmoid(logits).astype(np.float32)

        hit = (rng.random(n) < p_hit).astype(np.float32)

        # severity multiplier for the year (shared across peers in event years)
        sev = float(draw_severity(1)[0])

        # losses
        X[b, :] = hit * sev * exposure

    peer_coords = coords  # matches your style of "peer" coordinate array
    return X.astype(np.float32), years.astype(int), grid_shape, peer_coords





from typing import Dict, Tuple

def make_time_splits(
    B: int,
    frac_train: float = 0.50,
    frac_val: float = 0.20,
    frac_cal: float = 0.20,
    frac_eval: float = 0.10,
    *,
    strict: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Deterministic, time-ordered splits for a sequence of length B.

    Returns dict with keys: train, val, cal, eval (val may be empty if frac_val=0).
    Uses floor via int() for sizes; eval gets the remainder to ensure full coverage.

    If strict=True, checks that fractions are nonnegative and sum to 1 (up to 1e-6).
    """
    fracs = np.array([frac_train, frac_val, frac_cal, frac_eval], dtype=float)
    if np.any(fracs < 0):
        raise ValueError("Split fractions must be nonnegative.")
    if strict:
        if abs(fracs.sum() - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1. Got {fracs.sum():.6f}")

    idx = np.arange(B)

    n_train = int(frac_train * B)
    n_val   = int(frac_val   * B)
    n_cal   = int(frac_cal   * B)

    # eval gets remainder so indices cover [0..B)
    start_train = 0
    start_val   = start_train + n_train
    start_cal   = start_val   + n_val
    start_eval  = start_cal   + n_cal

    idx_train = idx[start_train:start_val]
    idx_val   = idx[start_val:start_cal]
    idx_cal   = idx[start_cal:start_eval]
    idx_eval  = idx[start_eval:]  # remainder

    # sanity checks
    if strict:
        all_idx = np.concatenate([idx_train, idx_val, idx_cal, idx_eval])
        if all_idx.size != B or np.any(all_idx != idx):
            raise RuntimeError("Split indices do not cover [0..B) contiguously.")

    return {"train": idx_train, "val": idx_val, "cal": idx_cal, "eval": idx_eval}


def apply_splits(X: np.ndarray, splits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convenience: slice X (B,n) into X_train/X_val/X_cal/X_eval using indices from make_time_splits.
    """
    return {k: X[v] for k, v in splits.items()}


def make_network_plots(c0, c_star, weights=None, H=None, top_frac=0.10, path="results-eobs/"):
    c0 = np.asarray(c0).astype(float)
    c_star = np.asarray(c_star).astype(float)
    n = c0.size

    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights).astype(float)
        weights = weights / weights.sum()

    delta = c_star - c0  # >0 means worse off after joining

    harmed = delta > 1e-12
    harmed_frac = harmed.mean()
    total_harm = float(np.sum(weights * np.maximum(delta, 0.0)))
    total_gain = float(np.sum(weights * np.maximum(-delta, 0.0)))

    # Identify highest-risk members by baseline call limit
    k = max(1, int(np.ceil(top_frac * n)))
    top_idx = np.argsort(c0)[-k:]
    rest_idx = np.setdiff1d(np.arange(n), top_idx)

    # ========== Plot 1: who gets relief? (highlight highest-risk) ==========
    plt.figure()
    plt.scatter(c0[rest_idx], c_star[rest_idx], s=12, alpha=0.6, label=f"Other {int((1-top_frac)*100)}%")
    plt.scatter(c0[top_idx],  c_star[top_idx],  s=18, alpha=0.9, label=f"Highest-risk top {int(top_frac*100)}%")

    lo = min(c0.min(), c_star.min())
    hi = max(c0.max(), c_star.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Baseline certified worst-case bill (c0)")
    plt.ylabel("Network certified worst-case bill (c*)")
    plt.title("Highest-risk members’ relief (points below diagonal)")
    plt.legend()
    plt.tight_layout()
    full_path = os.path.join(path, "relief_scatter.png")
    plt.savefig(full_path, dpi=300)

    # Add a small printout for “meaningful relief”
    relief_top = -delta[top_idx]  # positive = improvement
    print("\n[Top-risk relief]")
    print(f"  top {int(top_frac*100)}% by c0: median relief = {np.median(relief_top):.3f}, "
          f"mean relief = {np.mean(relief_top):.3f}, "
          f"fraction improved = {(relief_top>0).mean():.3f}")

    # ========== Plot 2: most people improve (distribution of changes) ==========
    plt.figure()
    plt.hist(delta, bins=40)
    plt.text(0.02, 0.95, f"harmed frac={harmed_frac:.1%}\nimproved frac={(1-harmed_frac):.1%}",
         transform=plt.gca().transAxes, va="top")
    plt.axvline(0.0, color="orange", linestyle="--")
    plt.xlabel("Change in certified worst-case bill (c* - c0)")
    plt.ylabel("Number of peers")
    plt.title("Most peers’ worst-case bills go down (mass left of 0)")
    plt.tight_layout()
    full_path = os.path.join(path, "delta_hist.png")
    plt.savefig(full_path, dpi=300)

    print("\n[Population summary]")
    print(f"  fraction improved (delta<0): {(delta < -1e-12).mean():.3f}")
    print(f"  fraction harmed frac  (delta>0): {harmed_frac:.3f}")
    print(f"  median change: {np.median(delta):.3f}   mean change: {np.mean(delta):.3f}")

    # ========== Plot 3: harm is tightly limited (weighted harm accounting) ==========
    # Sort harms to show how concentrated they are
    # h = np.maximum(c_star - c0, 0.0)
    # h_sorted = np.sort(h)[::-1]
    # cum = np.cumsum(h_sorted) / len(h)  # if you want average-per-member (w_i=1/n)

    h_i = weights * np.maximum(c_star - c0, 0.0)   # per-peer weighted harm contribution
    hs = np.sort(h_i)[::-1]
    cum = np.cumsum(hs)                       # plateau = total weighted harm

    k_nonzero = int(np.sum(hs > 1e-12))

    plt.figure()
    plt.plot(np.arange(1, len(hs)+1), cum)
    plt.axhline(H, linestyle="--")
    plt.axvline(k_nonzero, linestyle="--")
    plt.text(k_nonzero+1, cum[k_nonzero-1], f"nonzero harm: {k_nonzero}/{len(h)}")
    plt.xlabel("Top-k most-harmed members included")
    plt.ylabel("Cumulative average harm")
    plt.title("Total downside is capped, and concentrated")
    plt.tight_layout()
    full_path = os.path.join(path, "harm_budget.png")
    plt.savefig(full_path, dpi=300)

    print("\n[Harm budget accounting]")
    print(f"  total weighted harm: {total_harm:.6f}")
    if H is not None:
        print(f"  harm budget H:       {float(H):.6f}   (harm/H = {total_harm/float(H):.3f})")
    print(f"  total weighted gain: {total_gain:.6f}")

    harm = np.maximum(c_star - c0, 0.0)
    hs = np.sort(harm)[::-1]
    plt.figure()
    plt.plot(hs)
    plt.xlabel("Peers sorted by harm (most harmed first)")
    plt.ylabel("Individual harm (c* - c0)+")
    plt.title("Who pays the downside (most are zero)")
    plt.tight_layout()
    full_path = os.path.join(path, "harm_sorted.png")
    plt.savefig(full_path, dpi=300)

    # Optional: show how many peers account for 90% of harm
    if total_harm > 0:
        t90 = 0.9 * total_harm
        m90 = int(np.searchsorted(cum, t90) + 1)
        print(f"  #peers accounting for 90% of harm: {m90} / {n} ({m90/n:.3f})")

    plt.show()

# ---------------------------
# Data loading and processing: E-OBS daily precipitation to seasonal blocks
# ---------------------------

def load_eobs_rr(
    path: str,
    lat_slice: tuple[float, float],
    lon_slice: tuple[float, float],
    coarsen: tuple[int, int] | None = (4, 4),
    engine: str = "netcdf4",
    chunks: dict | None = None,
    time_slice: tuple[str, str] | None = None,       # e.g. ("1950-01-01","2000-12-31")
    year_range: tuple[int, int] | None = None,       # e.g. (1950, 2000)
) -> xr.DataArray:
    """
    Load E-OBS daily precipitation sum (rr) over a lat/lon window, optionally coarsened.
    Optionally restrict to a time interval or year range.

    Args:
        path: path to E-OBS NetCDF.
        lat_slice: (lat_min, lat_max) in degrees (E-OBS lat ascending).
        lon_slice: (lon_min, lon_max) in degrees (E-OBS lon ascending).
        coarsen: (k_lat, k_lon) factors for block-mean downsampling. If None, no coarsen.
        engine: xarray engine (default netcdf4).
        chunks: optional dask chunks, e.g. {"time": 365}.
        time_slice: optional (start, end) as ISO date strings, inclusive bounds.
        year_range: optional (year_start, year_end), inclusive bounds.

    Returns:
        rr_small: DataArray with dims (time, latitude, longitude).
                  If coarsened, latitude/longitude are reduced.
    """
    ds = xr.open_dataset(path, engine=engine, chunks=chunks) if chunks else xr.open_dataset(path, engine=engine)

    if "rr" not in ds:
        raise KeyError(f"'rr' not found in dataset variables: {list(ds.data_vars)}")

    rr = ds["rr"]  # (time, latitude, longitude)

    # --- time restriction (apply before spatial subsetting/coarsen) ---
    if time_slice is not None and year_range is not None:
        raise ValueError("Provide at most one of time_slice or year_range.")

    if time_slice is not None:
        t0, t1 = time_slice
        rr = rr.sel(time=slice(t0, t1))

    if year_range is not None:
        y0, y1 = year_range
        rr = rr.sel(time=(rr["time.year"] >= y0) & (rr["time.year"] <= y1))

    lat_min, lat_max = lat_slice
    lon_min, lon_max = lon_slice

    # E-OBS coords are typically ascending; slicing assumes that
    rr_win = rr.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    if coarsen is not None:
        k_lat, k_lon = coarsen
        rr_small = rr_win.coarsen(latitude=k_lat, longitude=k_lon, boundary="trim").mean()
    else:
        rr_small = rr_win

    return rr_small


def eobs_to_blocks(rr_small, u, months=[10,11,12,1,2,3], payout=1.0):
    hit = (rr_small > float(u)).astype("float32")
    daily_payout = float(payout) * hit

    season = daily_payout.where(daily_payout["time.month"].isin(months), drop=True)
    blocks = season.groupby("time.year").sum("time")  # (year, lat, lon)

    X = blocks.stack(peer=("latitude", "longitude")).transpose("year", "peer")
    X_np = X.to_numpy().astype("float32")

    years = X["year"].values
    latN = blocks.sizes["latitude"]
    lonN = blocks.sizes["longitude"]
    grid_shape = (latN, lonN)
    peer_coords = X["peer"].values

    return X_np, years, grid_shape, peer_coords


def basic_X_diagnostics(X_np: np.ndarray, label: str = "X") -> dict:
    """
    Print and return basic diagnostics used throughout your experiments.

    Args:
        X_np: (B, n) float array.
        label: label for printing.

    Returns:
        stats dict.
    """
    X_np = np.asarray(X_np)
    if X_np.ndim != 2:
        raise ValueError(f"X_np must be 2D (B,n), got shape {X_np.shape}")

    B, n = X_np.shape
    zero_frac = float((X_np == 0).mean())
    tot = X_np.sum(axis=1)

    p95 = float(np.quantile(tot, 0.95))
    p99 = float(np.quantile(tot, 0.99))
    mx = float(tot.max())
    mean_tot = float(tot.mean())
    med_tot = float(np.median(tot))

    print(f"[{label}] shape: {X_np.shape}, zero frac: {zero_frac:.3f}")
    print(f"[{label}] total loss per block: mean={mean_tot:.3g}, median={med_tot:.3g}, "
          f"p95/p99/max: {p95:.3g}/{p99:.3g}/{mx:.3g}")

    stats = {
        "B": B,
        "n": n,
        "zero_frac_entries": zero_frac,
        "total_mean": mean_tot,
        "total_median": med_tot,
        "total_p95": p95,
        "total_p99": p99,
        "total_max": mx,
    }
    return stats

# ------------------------
# Plotting
# ------------------------
from pathlib import Path

# ---------- saving helpers ----------
def ensure_dir(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def savefig(out_dir, name, dpi=300, close=True):
    out_dir = ensure_dir(out_dir)
    path = out_dir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close()
    return path

# ---------- small plot helpers ----------
def plot_compare(c_ref, c_alt, title, out_dir=None, name="call_limit_scatter", show=False):
    c_ref = np.asarray(c_ref)
    c_alt = np.asarray(c_alt)

    plt.figure()
    mx = float(max(c_ref.max(), c_alt.max()))
    plt.scatter(c_ref, c_alt, s=12, alpha=0.7)
    plt.plot([0, mx], [0, mx], linewidth=1.5)
    plt.xlabel("baseline c0")
    plt.ylabel("alternative c")
    plt.title(title)

    if out_dir is not None:
        return savefig(out_dir, name)
    if show:
        plt.show()
    else:
        plt.close()

def plot_delta_hist(c_ref, c_alt, title, out_dir=None, name="delta_hist", bins=40, show=False):
    c_ref = np.asarray(c_ref)
    c_alt = np.asarray(c_alt)
    dc = c_alt - c_ref

    plt.figure()
    plt.hist(dc, bins=bins)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("c_alt - c0")
    plt.ylabel("count")
    plt.title(title)

    if out_dir is not None:
        return savefig(out_dir, name)
    if show:
        plt.show()
    else:
        plt.close()

def plot_tail(c_ref, c_alt, title, topk=20, out_dir=None, name="tail_plot", show=False, logy=True):
    c_ref = np.asarray(c_ref)
    c_alt = np.asarray(c_alt)
    top = np.argsort(-c_ref)[:min(topk, len(c_ref))]

    plt.figure()
    plt.plot(c_ref[top], marker="o", linewidth=1.5, label="baseline")
    plt.plot(c_alt[top], marker="o", linewidth=1.5, label="alternative")
    if logy:
        plt.yscale("log")
    plt.xlabel("top-risk peers by baseline (ranked)")
    plt.ylabel("certified c")
    plt.title(title)
    plt.legend()

    if out_dir is not None:
        return savefig(out_dir, name)
    if show:
        plt.show()
    else:
        plt.close()

def plot_training_traces(survivors, steps=2000, topk=3,
                         title="Training traces (best validation candidates)",
                         out_dir=None, name="training_traces", show=False):
    plt.figure()
    for d in survivors[:topk]:
        y = d["hist"]["loss"]
        x = np.linspace(0, steps, num=len(y))
        plt.plot(x, y, label=f"r={d.get('radius')},lam={d.get('lam_noharm')}")
    plt.xlabel("step")
    plt.ylabel("train loss (logged checkpoints)")
    plt.title(title)
    plt.legend()

    if out_dir is not None:
        return savefig(out_dir, name)
    if show:
        plt.show()
    else:
        plt.close()

def plot_call_limit_hist(c0, c1, title="Certified call limits: baseline vs selected",
                         bins=40, out_dir=None, name="call_limit_hist", show=False):
    c0 = np.asarray(c0); c1 = np.asarray(c1)
    plt.figure()
    plt.hist(c0, bins=bins, alpha=0.6, label="Baseline")
    plt.hist(c1, bins=bins, alpha=0.6, label="Selected")
    plt.xlabel(r"$c_i$ (certified on CAL)")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()

    if out_dir is not None:
        return savefig(out_dir, name)
    if show:
        plt.show()
    else:
        plt.close()

def plot_coverage_hist(cov0_i, cov1_i, target,
                       title="Eval coverage distribution across peers",
                       bins=15, out_dir=None, name="coverage_hist", show=False):
    cov0_i = np.asarray(cov0_i); cov1_i = np.asarray(cov1_i)
    plt.figure()
    plt.hist(cov0_i, bins=bins, alpha=0.6, label="Baseline")
    plt.hist(cov1_i, bins=bins, alpha=0.6, label="Selected")
    plt.axvline(float(target), linewidth=1.5)
    plt.xlabel("empirical eval coverage per peer")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()

    if out_dir is not None:
        return savefig(out_dir, name)
    if show:
        plt.show()
    else:
        plt.close()

def plot_allocation_matrix_offdiag(A, c0=None, log=True,
                                   title="Allocation matrix (diag hidden)",
                                   out_dir=None, name="allocation_matrix_offdiag", show=False,
                                   figsize=(6, 6)):
    A = np.asarray(A, dtype=float)
    if c0 is not None:
        order = np.argsort(np.asarray(c0))
        A = A[order][:, order]

    A_plot = A.copy()
    np.fill_diagonal(A_plot, np.nan)

    plt.figure(figsize=figsize)
    M = np.log1p(A_plot) if log else A_plot
    im = plt.imshow(M, aspect="auto")
    plt.colorbar(im, label=("log(1 + A_ij)" if log else "A_ij") + " (diag hidden)")
    plt.title(title)

    if out_dir is not None:
        return savefig(out_dir, name)
    if show:
        plt.show()
    else:
        plt.close()

# ---------- one driver to generate everything from csan_quantile_pareto_selection() output ----------
def make_all_plots(results: dict, out_dir: str, topk_tail=20, show=False):
    out_dir = ensure_dir(out_dir)

    c0 = results["baseline"]["c"]
    c1 = results["selected"]["c"]
    A1 = results["selected"]["A"]

    cov0_i = results["baseline"].get("cov_i", None)
    cov1_i = results["selected"].get("cov_i", None)
    delta = results.get("meta", {}).get("delta", None)
    target = None if delta is None else (1.0 - float(delta))

    survivors = results.get("survivors", [])

    if survivors:
        plot_training_traces(survivors, out_dir=out_dir, show=show, name="training_traces")

    plot_call_limit_hist(c0, c1, out_dir=out_dir, show=show, name="call_limit_hist")
    plot_compare(c0, c1, "Certified call limits: selected vs baseline", out_dir=out_dir, show=show, name="call_limit_scatter")
    plot_delta_hist(c0, c1, "Change in certified call limits: selected - baseline", out_dir=out_dir, show=show, name="delta_hist")
    plot_tail(c0, c1, "Tail (top baseline risk): selected vs baseline", topk=topk_tail, out_dir=out_dir, show=show, name="tail_plot")

    if cov0_i is not None and cov1_i is not None and target is not None:
        plot_coverage_hist(cov0_i, cov1_i, target=target, out_dir=out_dir, show=show, name="coverage_hist")

    plot_allocation_matrix_offdiag(A1, c0=c0, log=True,
                                   title="Selected allocation matrix (ordered by baseline risk; diag hidden)",
                                   out_dir=out_dir, show=show, name="allocation_matrix_offdiag")
    


import json

def save_results(out_dir, name, **arrays_and_meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # split arrays vs meta
    arrays = {k: v for k, v in arrays_and_meta.items() if isinstance(v, np.ndarray)}
    meta   = {k: v for k, v in arrays_and_meta.items() if not isinstance(v, np.ndarray)}

    np.savez(out_dir / f"{name}.npz", **arrays)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(meta, f, indent=2)

def load_results(out_dir, name):
    out_dir = Path(out_dir)
    arrays = dict(np.load(out_dir / f"{name}.npz"))
    with open(out_dir / f"{name}.json", "r") as f:
        meta = json.load(f)
    return arrays, meta


def plot_peer_event_rates(X_train: np.ndarray, grid_shape, title_prefix="", out_dir=None):
    """
    X_train: (B,n) seasonal block totals per peer (e.g. yearly sum of trigger-days)
    grid_shape: (latN, lonN) such that n=latN*lonN
    """
    B, n = X_train.shape
    latN, lonN = grid_shape
    assert latN * lonN == n

    hit = (X_train > 0).astype(np.float32)   # (B,n)

    p = hit.mean(axis=0)                     # event rate per peer
    m = X_train.mean(axis=0)                 # mean seasonal payout per peer

    # reshape to map
    p_map = p.reshape(latN, lonN)
    m_map = m.reshape(latN, lonN)

    plt.figure(figsize=(6,5))
    plt.imshow(p_map, aspect="auto")
    plt.colorbar(label="P(season has >=1 trigger)")
    plt.title(f"{title_prefix} Event rate per peer")
    plt.tight_layout()
    if out_dir: 
        plt.savefig(f"{out_dir}/peer_event_rate.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(6,5))
    plt.imshow(m_map, aspect="auto")
    plt.colorbar(label="E[seasonal payout]")
    plt.title(f"{title_prefix} Mean seasonal payout per peer")
    plt.tight_layout()
    if out_dir: 
        plt.savefig(f"{out_dir}/peer_mean_payout.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_cooccurrence_matrices(X_train: np.ndarray, title_prefix="", out_dir=None):
    """
    Produces n×n matrices (heatmaps):
      - P_ij = P(i hits AND j hits)
      - C_ij = P(j hits | i hits)
      - Corr(X_i, X_j) on seasonal totals
    """
    B, n = X_train.shape
    hit = (X_train > 0).astype(np.float32)      # (B,n)
    p = hit.mean(axis=0)                        # (n,)

    # co-occurrence probability matrix
    P = (hit.T @ hit) / B                       # (n,n) since hit is 0/1

    # conditional probability matrix: P(j|i) = P_ij / p_i
    denom = np.maximum(p[:, None], 1e-12)
    C = P / denom

    # correlation of seasonal totals
    Xc = X_train - X_train.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, keepdims=True) + 1e-12
    Xz = Xc / std
    Corr = (Xz.T @ Xz) / (B - 1)

    # Plot helpers (log1p helps if very sparse)
    def heat(M, name, cmap_label):
        plt.figure(figsize=(6,5))
        plt.imshow(np.log1p(M), aspect="auto")
        plt.colorbar(label=f"log(1+{cmap_label})")
        plt.title(f"{title_prefix} {name} (log1p)")
        plt.tight_layout()
        if out_dir:
            plt.savefig(f"{out_dir}/{name.lower().replace(' ','_')}.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    heat(P, "Co-occurrence P(i and j)", "P_ij")
    heat(C, "Conditional P(j|i)", "C_ij")

    plt.figure(figsize=(6,5))
    plt.imshow(Corr, aspect="auto")
    plt.colorbar(label="corr")
    plt.title(f"{title_prefix} Corr(season totals)")
    plt.tight_layout()
    if out_dir:
        plt.savefig(f"{out_dir}/corr_season_totals.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return {"P": P, "C": C, "Corr": Corr, "p": p}

def plot_A_heatmap(A, title="A", max_n=200):
    A = np.asarray(A)
    n = A.shape[0]
    if n > max_n:
        idx = np.linspace(0, n-1, max_n).astype(int)
        A = A[np.ix_(idx, idx)]
        title = f"{title} (subsampled to {max_n})"
    plt.figure()
    plt.imshow(A, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("j (source)")
    plt.ylabel("i (receiver)")
    plt.tight_layout()

def plot_A_diagnostics(A, title="A"):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    diag = np.diag(A)
    # row entropy (natural log)
    eps = 1e-12
    ent = -(A * np.log(A + eps)).sum(axis=1)

    plt.figure()
    plt.hist(diag, bins=30)
    plt.title(f"{title}: diag(A) distribution")
    plt.xlabel("A_ii"); plt.ylabel("count")
    plt.tight_layout()

    plt.figure()
    plt.hist(ent, bins=30)
    plt.title(f"{title}: row entropy distribution")
    plt.xlabel("row entropy"); plt.ylabel("count")
    plt.tight_layout()

def plot_row_weights_on_grid(A, peer_coords, i, title=None, s=30):
    A = np.asarray(A)
    coords = np.asarray(peer_coords)
    w = A[i]
    plt.figure()
    plt.scatter(coords[:,0], coords[:,1], c=w, s=s)
    plt.colorbar()
    plt.title(title or f"Row {i}: weights sent to peers (A[{i},:])")
    plt.tight_layout()


def plot_support_overlap(A, W_local):
    A = np.asarray(A)
    W = np.asarray(W_local)
    suppA = (A > 1e-10)
    suppW = (W > 0)

    overlap = (suppA & suppW).sum()
    extra = (suppA & ~suppW).sum()
    missing = (~suppA & suppW).sum()

    print("Support overlap:", overlap, "extra outside W_local:", extra, "missing from W_local:", missing)

    plt.figure()
    plt.imshow(suppA.astype(float), aspect="auto")
    plt.title("Support(A): A_ij > 1e-10")
    plt.tight_layout()

    plt.figure()
    plt.imshow(suppW.astype(float), aspect="auto")
    plt.title("Support(W_local): W_ij > 0")
    plt.tight_layout()


def variance_objective(A, XT, lam_diag=0.1, ridge=1e-6):
    Sigma = _cov_shrinkage(XT, lam_diag=lam_diag, ridge=ridge)
    A = np.asarray(A, dtype=float)
    return float(np.trace(A.T @ Sigma @ A))

def compare_objectives(XT, A_bar, alpha_list):
    n = XT.shape[1]
    I = np.eye(n)
    for a in alpha_list:
        A = (1-a)*I + a*A_bar
        print(f"alpha={a:0.2f}  J(A)={variance_objective(A, XT):.6g}")



# ============================================================
# Print LaTeX tables
# ============================================================

def print_nc_sensitivity_row(res, nC):
    """Print one row for the nC sensitivity table."""
    s = res["peer_cov_summary_operational"]
    pass_rate = float(np.mean([st == "PASS" for st in res["status"]]))
    print(
        f" & {nC} & {pass_rate:.2f} & "
        f"{s['mean']:.3f} & {s['p05']:.3f} & {s['min']:.3f} & "
        f"{s['frac_below_nominal']:.3f} \\\\"
    )

def print_nc_table(results_global, results_local, nC_list, caption, label):
    for i, nC in enumerate(nC_list):
        prefix = r"\multirow{" + str(len(nC_list)) + r"}{*}{\rotatebox{90}{\tiny Global}}" if i == 0 else ""
        print(prefix, end="")
        print_nc_sensitivity_row(results_global[nC], nC)
    print(r"\midrule")
    for i, nC in enumerate(nC_list):
        prefix = r"\multirow{" + str(len(nC_list)) + r"}{*}{\rotatebox{90}{\tiny Local}}" if i == 0 else ""
        print(prefix, end="")
        print_nc_sensitivity_row(results_local[nC], nC)


def print_identity_nc_table(results_dict, nC_list, caption, label):
    for nC in nC_list:
        s = results_dict[nC]["peer_cov_summary_identity"]
        print(
            f"{nC} & {s['mean']:.3f} & {s['p05']:.3f} & "
            f"{s['min']:.3f} & {s['frac_below_nominal']:.3f} \\\\"
        )


def print_latex_row(res, label):
    """Print one LaTeX table row from a validity_first_backtest result dict."""
    s = res["peer_cov_summary_operational"]
    pass_rate = float(np.mean([st == "PASS" for st in res["status"]]))
    agg = res["cap_ratio_operational"]
    top = res["top10_ratio_operational"]
    alp = res["alpha_op"]
    print(
        f"{label} & {pass_rate:.2f} & "
        f"{np.mean(alp):.2f}\\tiny$\\pm${np.std(alp):.2f} & "
        f"{s['mean']:.3f} ({s['p05']:.3f}) & "
        f"{np.nanmean(agg):.3f}\\tiny$\\pm${np.nanstd(agg):.3f} & "
        f"{np.nanmean(top):.3f}\\tiny$\\pm${np.nanstd(top):.3f} \\\\"
    )

def print_latex_identity_row(res):
    """Print identity baseline row (same regardless of pooling method)."""
    s = res["peer_cov_summary_identity"]
    print(f"Identity & -- & 0 & {s['mean']:.3f} ({s['p05']:.3f}) & -- & -- \\\\")