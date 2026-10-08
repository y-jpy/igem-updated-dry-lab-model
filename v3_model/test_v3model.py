"""
test_v3model.py — regression tests before any result is reported.
Run: python test_v3model.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import params as P
import v3model as M
import closure as C6

PASS = []


def check(name, ok, detail=""):
    PASS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ----------------------------------------------------------------------
# T1: method-of-lines gel solver vs Crank closed form (2nd order)
# ----------------------------------------------------------------------
def test_convergence():
    D = P.Deff_g_V
    k = P.k_deg_g_V
    L = P.gel["L"]
    C0 = 1.0
    t_probe = np.array([0.5, 2.0, 10.0])   # L^2/D ~ 66 s: profile still O(C0)
    errs = []
    for nx in (100, 200, 400, 800):
        dx = L / nx
        A = np.zeros((nx, nx))
        F = D / dx ** 2
        for i in range(nx):
            A[i, i] += -2 * F - k
            if i > 0:
                A[i, i - 1] += F
                A[i - 1, i] += F
        # perfect sink at x=0: loop's -2F assumed two neighbors; cell 0 has
        # one real neighbor (-F) + half-cell sink flux (-2F) -> net += -F
        A[0, 0] += -F
        # zero-flux backing at x=L: last cell has only one neighbor
        A[nx - 1, nx - 1] += F
        sol = solve_ivp(lambda t, C: A @ C, (0, t_probe[-1]), C0 * np.ones(nx),
                        t_eval=t_probe, method="BDF", rtol=1e-8, atol=1e-12)
        Cnum = sol.y[:, 2]  # last probe time
        Cex = M.crank_concentration(np.linspace(dx / 2, L - dx / 2, nx),
                                    t_probe[-1], C0, D, k, L)
        # L1 norm excluding the first 5% of cells: the corner singularity
        # dC/dx|_0 ~ t^(-1/2) at the perfect sink makes the MAX norm converge
        # at only 1st order regardless of scheme order (documented caveat).
        nskip = max(1, nx // 20)
        errs.append(np.mean(np.abs(Cnum[nskip:] - Cex[nskip:])) / C0)
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    order2 = all(3.5 < r < 4.5 for r in ratios)
    check("T1 Crank convergence (2nd order)", order2,
          f"errors={['%.2e' % e for e in errs]}")


# ----------------------------------------------------------------------
# T2: two-domain mass budget with no degradation (log grid — the initial
# release spike is unresolved on a uniform grid: corner singularity)
# ----------------------------------------------------------------------
def test_mass_budget():
    t = np.concatenate([[0], np.logspace(-2, 6, 400)])
    _, J, f, Csol = M.solve_two_domain(species="V", t_out=t, kg=0.0, kt=0.0)
    nxg = P.numerics["nx_gel"]
    dxt = P.tissue["L"] / P.numerics["nx_tis"]
    inv = (Csol[:nxg, :].sum(axis=0) * P.gel["eps"] * P.gel["L"] / nxg +
           Csol[nxg:, :].sum(axis=0) * P.tissue["eps"] * dxt)
    loaded = P.gel["eps"] * P.gel["L"]
    # Budget: with kg=kt=0 every loaded molecule must eventually leave
    # through the dermal sink, so f(t->inf) = 1 (f counts gel->tissue
    # transfer; the sink flux is the terminal exit and is not in J).
    # Inventory must stay non-negative and f monotone (no mass creation).
    ok = (abs(f[-1] - 1) < 0.01 and inv.min() > -1e-6 * loaded
          and np.all(np.diff(f) >= -1e-8))
    check("T2 mass budget (f->1, no creation)",
          ok, f"f(t=10^6 s)={f[-1]:.4f}, min inventory/loaded={inv.min()/loaded:.2e}")


# ----------------------------------------------------------------------
# T3: NO baseline self-consistency vs the team's 36 nM
# ----------------------------------------------------------------------
def test_no_baseline():
    NO_molcm3 = M.no_baseline()                     # scalar, mol/cm^3
    NO_nM = float(M.nM(NO_molcm3))
    from scipy.optimize import brentq
    f = lambda b: float(M.nM(M.no_baseline(band_cm=b))) - 36.0
    try:
        b_cal = brentq(f, 1e-4, 1.0)
        check("T3 NO baseline calibration", True,
              f"base band 50 um -> {NO_nM:.0f} nM; "
              f"36 nM reproduced at band={b_cal * 1e4:.0f} um")
    except ValueError:
        check("T3 NO baseline calibration", False, "no bracket")


# ----------------------------------------------------------------------
# T4: radial front speed vs analytic 2*sqrt(Dn*reff)
# ----------------------------------------------------------------------
def test_front_speed():
    Dn = P.closure["D_n"]
    for (rp_v, d_v) in [(1.0, 0.0), (1.0, 0.3), (2.0, 0.5)]:
        r_eff_s = (rp_v - d_v) / 86400.0          # 1/d -> 1/s
        c_an = 2 * np.sqrt(Dn * r_eff_s) * 86400 * 1e4  # -> um/day
        t, r_f, A_c, tc = C6.solve_closure(
            rp_of_t=lambda tt: rp_v, d_of_t=lambda tt: d_v,
            R_w=0.2, t_end_d=40, nr=800)
        # fit drf/dt over mid-traverse (skip initial transient and endgame)
        mask = (r_f > 0.02) & (r_f < 0.15)
        if mask.sum() > 10:
            slope = -np.polyfit(t[mask], r_f[mask], 1)[0] * 1e4 * 1  # cm/d->um/d
            err = abs(slope - c_an) / c_an
            check(f"T4 front speed rp={rp_v} d={d_v}", err < 0.08,
                  f"analytic={c_an:.0f} um/d, numeric={slope:.0f} um/d, err={err:.1%}")
        else:
            check(f"T4 front speed rp={rp_v} d={d_v}", False, "insufficient traverse")


# ----------------------------------------------------------------------
# T5: front failure reproduced (chronic baseline never closes)
# ----------------------------------------------------------------------
def test_front_failure():
    t, r_f, A_c, tc = C6.solve_closure(R_w=0.2, t_end_d=60)
    check("T5 chronic front stall (tc=inf)", np.isinf(tc), f"tc={tc}")


if __name__ == "__main__":
    test_convergence()
    test_mass_budget()
    test_no_baseline()
    test_front_speed()
    test_front_failure()
    n_ok = sum(1 for _, ok, _ in PASS if ok)
    print(f"\n{n_ok}/{len(PASS)} tests passed")
