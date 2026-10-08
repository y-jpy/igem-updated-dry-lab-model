"""
closure.py — Model 6: radial Fisher-KPP fibroblast front with effect
compartment, plus Model 7 loading bound. All coupling constants are T4
priors (no MTT/qPCR data — scenario analysis, never fitted).

Geometry: wound void at r < R_w (n ~ 0), intact tissue reservoir at r > R_w
(n ~ K). Fibroblasts invade INWARD; r_f(t) = innermost radius where
n >= thr; closure when r_f -> 0.
"""

import numpy as np
from scipy.integrate import solve_ivp
import params as P


def _EC50():
    """FGF2 proliferative EC50 [mol/cm^3] from ng/mL prior."""
    return P.closure["EC50_F_ngmL"] * 1e-9 * 1e-3 / P.FGF["MW"]


def rp(Ce, Ie):
    """Eq. (58): FGF-enhanced, inflammation-penalised proliferation [1/d]."""
    c = P.closure
    fgf = 1 + c["E_max"] * Ce / (_EC50() + Ce)
    infl = 1 - c["beta_r"] * Ie / (c["K_I_r"] + Ie)
    return c["r0_chronic"] * fgf * infl


def d_rate(Ie):
    """Eq. (59): inflammation-driven death rate [1/d]."""
    c = P.closure
    return c["d0_chronic"] + c["Delta_d"] * Ie ** c["s_I"] / \
        (c["K_I_d"] ** c["s_I"] + Ie ** c["s_I"])


def solve_closure(rp_of_t=None, d_of_t=None, R_w=None, t_end_d=None, nr=800,
                  pulse=None, Ce_of_t=None, Ie_of_t=None):
    """Radial Fisher-KPP method of lines.

    rp_of_t/d_of_t: callables t[d] -> rate [1/d]. If Ce_of_t/Ie_of_t given,
    rates come from the coupling functions Eqs. (58)-(59) instead.
    pulse: (T_pulse_d, rp_pulse, d_pulse) — square therapeutic pulse on the
    chronic baseline outside the window (doc 9.4 protocol).
    Returns t [d], r_f(t) [cm], A_c(t), tc [d] (np.inf if never closes).
    """
    c = P.closure
    R_w = c["R_w_cm"] if R_w is None else R_w
    t_end_d = c["t_end_closure"] if t_end_d is None else t_end_d
    K = c["K_cells"]
    w = c["w_smooth_cm"]
    R_inf = 3 * R_w

    if rp_of_t is None:
        if Ce_of_t is not None or Ie_of_t is not None:
            rp_of_t = lambda t: rp(Ce_of_t(t) if Ce_of_t is not None else 0.0,
                                   Ie_of_t(t) if Ie_of_t is not None else 0.0)
            d_of_t = lambda t: d_rate(Ie_of_t(t) if Ie_of_t is not None else 0.0)
        else:
            rp_of_t = lambda t: c["r0_chronic"]
            d_of_t = lambda t: c["d0_chronic"]
    if d_of_t is None:
        d_of_t = lambda t: c["d0_chronic"]

    if pulse is not None:
        Tp, rp_p, d_p = pulse
        base_rp, base_d = rp_of_t, d_of_t
        rp_of_t = lambda t: rp_p if t < Tp else base_rp(t)
        d_of_t = lambda t: d_p if t < Tp else base_d(t)
        nss_treated = K * max(0.0, 1 - d_p / rp_p)
    else:
        # n_ss from the best achievable (1 - d/rp) over the run; for
        # stationary rates this is the plain chronic value, for transient
        # treatment it captures the peak therapeutic state
        t_scan = np.concatenate([[0], np.logspace(-3, np.log10(max(t_end_d, 1)),
                                                  200)])
        nss_treated = K * max(0.0, np.max(
            1 - np.array([d_of_t(t) for t in t_scan]) /
            np.array([rp_of_t(t) for t in t_scan])))

    r = np.linspace(0, R_inf, nr)
    dr = r[1] - r[0]
    n0 = K / 2 * (1 + np.tanh((r - R_w) / w))   # void inside, reservoir outside
    n0[-1] = K                                   # pin reservoir node

    if nss_treated <= 0 and pulse is None and Ce_of_t is None and Ie_of_t is None:
        # stationary chronic baseline with reff <= 0: front never advances,
        # wound never closes (time-varying rates must still be integrated)
        t = np.linspace(0, t_end_d, 400)
        return t, R_w * np.ones_like(t), np.zeros_like(t), np.inf

    thr = c["n_thr"] * nss_treated

    D_day = c["D_n"] * 86400.0   # cm^2/s -> cm^2/d (t is in days, rates in 1/d)

    def rhs(t, n):
        flux = (r[:-1] + dr / 2) * (n[1:] - n[:-1]) / dr      # F_{i+1/2}
        lap = np.zeros_like(n)
        lap[1:-1] = (flux[1:] - flux[:-1]) / (dr * r[1:-1])
        lap[0] = 2 * (n[1] - n[0]) / dr ** 2                   # symmetry
        lap[-1] = 0                                            # pinned reservoir
        dn = D_day * lap + rp_of_t(t) * n * (1 - n / K) - d_of_t(t) * n
        dn[-1] = 0.0
        return dn

    t_eval = np.linspace(0, t_end_d, 400)
    sol = solve_ivp(rhs, (0, t_end_d), n0, t_eval=t_eval,
                    method="BDF", rtol=1e-6, atol=1e-2)
    t, N = sol.t, sol.y

    r_f = np.zeros_like(t)
    for i in range(len(t)):
        above = np.where(N[:, i] >= thr)[0]
        r_f[i] = r[above[0]] if len(above) else 0.0
    A_c = 1 - (r_f / R_w) ** 2
    closed = np.where(r_f <= dr)[0]
    tc = t[closed[0]] if len(closed) else np.inf
    return t, r_f, A_c, tc


def loading_bound(C_target_uM=None, f_rel=None, delta=None, Lg=None,
                  eps_g=None, eps_t=1.0):
    """Eq. (75): lower bound on free loading C_V,0 [uM]."""
    d = P.design
    C_target_uM = d["C_V_target_uM"] if C_target_uM is None else C_target_uM
    f_rel = d["f_rel"] if f_rel is None else f_rel
    delta = P.tissue["delta"] if delta is None else delta
    Lg = P.gel["L"] if Lg is None else Lg
    eps_g = P.gel["eps"] if eps_g is None else eps_g
    return C_target_uM * eps_t * delta / (f_rel * eps_g * Lg)
