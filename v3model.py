"""
v3model.py — Reference implementation of the iGEM V3 bio-patch model.
Linear (no-heparin) configuration: both peptides obey linear Fickian release.

Units: cm, s, mol, mol/cm^3 (interstitial basis). Reads ONLY from params.py.

Unit conversions used throughout:
    1 mol/cm^3 = 1 mol/mL = 1e3 mol/L
    mol/cm^3 -> nM:  x 1e12      mol/cm^3 -> uM:  x 1e9
"""

import numpy as np
from scipy.integrate import solve_ivp
import params as P


def uM(c):    # mol/cm^3 -> uM
    return np.asarray(c) * 1e9


def nM(c):    # mol/cm^3 -> nM
    return np.asarray(c) * 1e12


def from_uM(x):
    return np.asarray(x) * 1e-9


# ======================================================================
# Model 1/2 — Transport
# ======================================================================

def crank_concentration(x, t, C0, D, k, L, m_max=400):
    """Closed form Eq. (18): C(x,t) for desorption from a sealed-face slab,
    perfect sink at x=0. x in (0, L]."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    C = np.zeros_like(x)
    for m in range(m_max):
        coef = 4.0 / (2 * m + 1) / np.pi
        C += C0 * coef * np.sin((2 * m + 1) * np.pi * x / (2 * L)) \
             * np.exp(-(D * (2 * m + 1) ** 2 * np.pi ** 2 / (4 * L ** 2) + k) * t)
    return C


def crank_fraction_released(t, D, L, m_max=400):
    """Cumulative fractional release, no degradation, Eq. (19)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    s = np.zeros_like(t)
    for m in range(m_max):
        s += np.exp(-(2 * m + 1) ** 2 * np.pi ** 2 * D * t / (4 * L ** 2)) \
             / (2 * m + 1) ** 2
    return 1 - 8 / np.pi ** 2 * s


def _two_domain_matrices(Dg, kg, Dt, kt, Lg, Lt, eps_g, eps_t, Kp, nx_g, nx_t):
    """Finite-volume matrices for the monolithic gel+tissue problem.

    Unknowns: pore concentrations C on gel cells (nx_g) then tissue cells (nx_t).
    Interface: C_g(0) = Kp*C_t(0), superficial-flux matching (Eqs. 11-12).
    BCs: zero flux at x=Lg (backing), Dirichlet 0 at x=-Lt (systemic sink).
    Returns (A, J_row): dC/dt = A C ; J(t) = J_row @ C  [mol/cm2/s into wound].
    """
    dxg, dxt = Lg / nx_g, Lt / nx_t
    N = nx_g + nx_t
    A = np.zeros((N, N))

    # gel interior: cell i -> i+1 flux F = eps_g*Dg/dxg * (C_i - C_{i+1})
    for i in range(nx_g - 1):
        F = eps_g * Dg / dxg
        A[i, i] += -F / (eps_g * dxg) - kg
        A[i, i + 1] += F / (eps_g * dxg)
        A[i + 1, i] += F / (eps_g * dxg)
        A[i + 1, i + 1] += -F / (eps_g * dxg)
    A[nx_g - 1, nx_g - 1] += -kg          # backing: zero-flux, degradation only

    # tissue interior
    for j in range(nx_g + 1, N):
        F = eps_t * Dt / dxt
        A[j, j] += -F / (eps_t * dxt) - kt
        A[j, j - 1] += F / (eps_t * dxt)
        A[j - 1, j] += F / (eps_t * dxt)
        A[j - 1, j - 1] += -F / (eps_t * dxt)
    # Dirichlet sink at -Lt (dermal base = LAST tissue cell): flux to a
    # 0-reservoir over half a cell. Cell nx_g (interface, depth 0) only
    # needs its degradation term here; the interior loop starts at nx_g+1.
    F_sink = eps_t * Dt / (dxt / 2)
    A[nx_g, nx_g] += -kt
    A[N - 1, N - 1] += -F_sink / (eps_t * dxt)

    # interface: R1 = (dxg/2)/(eps_g Dg), R2 = (dxt/2)/(eps_t Dt)
    # F = (C_gN - Kp*C_t1) / (R1 + Kp*R2)   [superficial flux, mol/cm2/s]
    R1 = (dxg / 2) / (eps_g * Dg)
    R2 = (dxt / 2) / (eps_t * Dt)
    denom = R1 + Kp * R2
    gi, ti = nx_g - 1, nx_g
    A[gi, gi] += -1.0 / (eps_g * dxg * denom)
    A[gi, ti] += +Kp / (eps_g * dxg * denom)
    A[ti, ti] += -Kp / (eps_t * dxt * denom) - kt
    A[ti, gi] += +1.0 / (eps_t * dxt * denom)

    J_row = np.zeros(N)
    J_row[gi] = +1.0 / denom
    J_row[ti] = -Kp / denom
    return A, J_row


def solve_two_domain(species="V", t_out=None, C0_g=1.0, **overrides):
    """Monolithic gel+tissue release solve (linear). Returns
    t, J(t) [mol/cm2/s per unit C0_g], f(t) [fraction released], C(t)."""
    p = dict(
        Dg=P.Deff_g_V if species == "V" else P.Deff_g_F,
        kg=P.k_deg_g_V if species == "V" else P.k_deg_g_F,
        Dt=P.tissue["Deff_t_V"] if species == "V" else P.tissue["Deff_t_F"],
        kt=P.tissue["k_prot_V"] if species == "V" else P.tissue["k_prot_F"],
        Lg=P.gel["L"], Lt=P.tissue["L"],
        eps_g=P.gel["eps"], eps_t=P.tissue["eps"], Kp=P.tissue["Kp"],
        nx_g=P.numerics["nx_gel"], nx_t=P.numerics["nx_tis"],
    )
    p.update(overrides)
    if t_out is None:
        t_out = np.linspace(0, P.numerics["t_end_release"], P.numerics["n_t_release"])
    A, J_row = _two_domain_matrices(
        p["Dg"], p["kg"], p["Dt"], p["kt"], p["Lg"], p["Lt"],
        p["eps_g"], p["eps_t"], p["Kp"], p["nx_g"], p["nx_t"])
    N = p["nx_g"] + p["nx_t"]
    C0 = np.zeros(N)
    C0[:p["nx_g"]] = C0_g
    sol = solve_ivp(lambda t, C: A @ C, (0, t_out[-1]), C0,
                    t_eval=t_out, method="BDF", rtol=1e-6, atol=1e-14)
    J = J_row @ sol.y
    loaded = p["eps_g"] * p["Lg"] * C0_g
    f = np.concatenate([[0], np.cumsum(0.5 * (J[1:] + J[:-1]) *
                                       np.diff(t_out))]) / loaded
    return t_out, J, f, sol.y


# ======================================================================
# Models 3-4 — MD2 competition and NF-kB signaling
# ======================================================================

def theta_LPS(C_V, KD_P, L=None):
    """Eq. (31): fraction of MD2 occupied by LPS (absolute form)."""
    L = from_uM(P.md2["L_uM"]) if L is None else L
    KD_L = P.md2["KD_L"]
    return (L / KD_L) / (1 + L / KD_L + C_V / KD_P)


def scaled_arm(psi, a, gamma, h=None):
    """Eqs. (37)-(38): the whole V14 arm in dimensionless form.
    Returns (theta_inhib, theta_LPS, pi) with pi = normalised p65 response."""
    h = P.signaling["h"] if h is None else h
    th_i = psi / (1 + a + psi)
    th_L = a / (1 + a + psi)
    pi_n = (a * gamma) ** h / ((a * gamma) ** h + (1 + a + psi) ** h)
    return th_i, th_L, pi_n


def psi50(a):
    """Eq. (39): exact scaled dose for 50% suppression of active receptor."""
    return 1 + a


def p65_nuc(C_V, KD_P, L=None):
    """Eq. (41): nuclear p65 [mol/cm^3] from active receptor density."""
    L = from_uM(P.md2["L_uM"]) if L is None else L
    Ttot = from_uM(P.md2["TLR4_tot_nM"])
    KNF = from_uM(P.signaling["K_NF_nM"])
    h = P.signaling["h"]
    p65bas = from_uM(P.signaling["p65_bas_nM"])
    p65max = from_uM(P.signaling["p65_max_nM"])
    Tstar = Ttot * (L / P.md2["KD_L"]) / (1 + L / P.md2["KD_L"] + C_V / KD_P)
    return p65bas + (p65max - p65bas) * Tstar ** h / (KNF ** h + Tstar ** h)


def TLR4_star_target():
    """Eq. (42): admissible active-receptor density for the p65 target."""
    KNF = from_uM(P.signaling["K_NF_nM"])
    h = P.signaling["h"]
    p65bas = from_uM(P.signaling["p65_bas_nM"])
    p65max = from_uM(P.signaling["p65_max_nM"])
    p65tgt = from_uM(P.signaling["p65_target_nM"])
    return KNF * ((p65tgt - p65bas) / (p65max - p65tgt)) ** (1 / h)


def required_dose(L=None, KD_P=1.0, Ttgt=None):
    """Eq. (44) corrected required dose, in units of KD_P (set KD_P=1 to get
    the dimensionless u). Returns (P_req, feasible)."""
    L = from_uM(P.md2["L_uM"]) if L is None else L
    Ttot = from_uM(P.md2["TLR4_tot_nM"])
    Ttgt = TLR4_star_target() if Ttgt is None else Ttgt
    a = L / P.md2["KD_L"]
    u = a * (Ttot / Ttgt - 1) - 1
    return KD_P * u, u > 0


# ======================================================================
# Model 5 — iNOS / NO / nitrite
# ======================================================================

def mac_density_volumetric(band_cm=None):
    """cells/mm2 infiltration over a band depth -> cells/cm3.
    N [cells/mm2] / band [mm] = cells/mm3; x1000 -> cells/cm3."""
    band = P.no_chain["mac_band_cm"] if band_cm is None else band_cm
    return P.no_chain["N_mac_per_mm2"] / (band * 10.0) * 1000.0


def no_baseline(q=None, band_cm=None, k_scav=None):
    """Quasi-steady wound free NO [mol/cm^3], [NO] = N_mac*q/k_scav."""
    q = P.no_chain["q_chronic"] if q is None else q
    k = P.no_chain["k_scav"] if k_scav is None else k_scav
    return mac_density_volumetric(band_cm) * q / k


def nitrite_trajectory(C_V_of_t, t_grid, KD_P, q=None, k_scav=None):
    """Full V14 arm: C_V(t) -> p65 -> R_rel(t) -> cumulative nitrite (Eq. 53).
    C_V_of_t: callable(t) -> mol/cm^3 at the macrophage band.
    Returns t, NO_free_nM(t), NO2_cum_uM(t)."""
    q = P.no_chain["q_chronic"] if q is None else q
    k = P.no_chain["k_scav"] if k_scav is None else k_scav
    Nvol = mac_density_volumetric()
    K_iNOS = P.no_chain["K_iNOS_frac"] * from_uM(P.signaling["p65_max_nM"])
    p = P.no_chain["p_iNOS"]
    kturn = P.no_chain["k_turn"]
    phi = P.no_chain["phi"]

    def rhs(t, y):
        p65 = p65_nuc(C_V_of_t(t), KD_P)
        drive = Nvol * q * p65 ** p / (K_iNOS ** p + p65 ** p)
        return [kturn * (drive - y[0]), phi * y[0]]

    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), [0.0, 0.0],
                    t_eval=t_grid, method="BDF", rtol=1e-8, atol=1e-24)
    NO_free_nM = nM(sol.y[0] / k)
    NO2_uM = uM(sol.y[1])
    return sol.t, NO_free_nM, NO2_uM
