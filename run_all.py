"""
run_all.py — end-to-end V3 model run. Reads ONLY params.py; edit params and
rerun this file to regenerate every figure and table.

Outputs (written to /workspace/v3_run, then copied to /mnt/results/v3_results):
  figures/  fig1_release.png        Model 1 cumulative release f(t), both species
            fig2_tissue.png         Model 2 tissue concentrations vs time
            fig3_no_nitrite.png     Model 5 NO / cumulative nitrite trajectories
            fig4_closure.png        Model 6 front trajectories and pulse table
            fig5_scaled_arm.png     Models 3-4 scaled dose-response
  tables/   table_release_check.csv doc-baseline 95% release validation
            table_closure_pulse.csv minimum pulse vs wound radius
            table_loading.csv       Model 7 loading bound + days above target
            table_no_check.csv      NO baseline calibration
            summary.json            headline numbers
"""

import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

import params as P
import v3model as M
import closure as C6

matplotlib.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

OUT = "/workspace/v3_run"
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

summary = {}

# ----------------------------------------------------------------------
# 1. Model 1 — release curves f(t) (log grid; the initial spike is a
#    corner singularity and aliases on a uniform grid)
# ----------------------------------------------------------------------
t_log = np.concatenate([[0], np.logspace(-2, 6, 600)])       # 0.01 s .. 11.6 d
_, J_V, f_V, _ = M.solve_two_domain(species="V", t_out=t_log)
_, J_F, f_F, _ = M.solve_two_domain(species="F", t_out=t_log)

i95 = np.searchsorted(f_V, 0.95)
t95_V = t_log[min(i95, len(t_log) - 1)]
i95 = np.searchsorted(f_F, 0.95)
t95_F = t_log[min(i95, len(t_log) - 1)]
summary["t95_release_s"] = {"V14": float(t95_V), "FGF2G3": float(t95_F)}
summary["tau_late_s"] = {
    "V14": 4 * P.gel["L"] ** 2 / (np.pi ** 2 * P.Deff_g_V),
    "FGF2G3": 4 * P.gel["L"] ** 2 / (np.pi ** 2 * P.Deff_g_F),
}

fig, ax = plt.subplots(figsize=(4.6, 3.4))
# gel-only release into a perfect sink (Model 1, Crank Eq. 19) and the
# coupled gel->tissue transfer fraction from the two-domain solver
f_gV = M.crank_fraction_released(t_log[1:], P.Deff_g_V, P.gel["L"])
f_gF = M.crank_fraction_released(t_log[1:], P.Deff_g_F, P.gel["L"])
ax.semilogx(t_log[1:], f_gV, color="#0279EE", label="V14 (gel only)")
ax.semilogx(t_log[1:], f_gF, color="#FF9400", label="FGF2-G3 (gel only)")
ax.semilogx(t_log[1:], np.maximum.accumulate(f_V[1:]), color="#0279EE",
            ls="--", label="V14 (into tissue)")
ax.semilogx(t_log[1:], np.maximum.accumulate(f_F[1:]), color="#FF9400",
            ls="--", label="FGF2-G3 (into tissue)")
ax.axhline(0.95, ls=":", lw=0.8, color="grey")
ax.set_xlabel("time (s)")
ax.set_ylabel("cumulative fraction released, f(t)")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_release.png"), dpi=200)
plt.close(fig)

# Validation against the document baseline (10% GelMA, 500 um, doc Deff):
# doc reports 95% release in 14.5 min (V14) / 40.9 min (FGF2-G3). Metric is
# gel-only release into a perfect sink (Crank closed form, Eq. 19).
rows = []
for sp, D_doc, t95_doc in [("V14", 3.25e-6, 14.5), ("FGF2-G3", 1.15e-6, 40.9)]:
    tt = np.logspace(0, 4, 400)
    f = M.crank_fraction_released(tt, D_doc, 500e-4)
    t95 = tt[np.searchsorted(f, 0.95)] / 60.0
    rows.append((sp, D_doc, t95_doc, round(t95, 1)))
with open(os.path.join(TAB, "table_release_check.csv"), "w") as fh:
    fh.write("species,Deff_doc_cm2_s,t95_doc_min,t95_ours_min\n")
    for r in rows:
        fh.write(",".join(map(str, r)) + "\n")
summary["release_check_doc_baseline"] = [
    {"species": r[0], "t95_doc_min": r[2], "t95_ours_min": r[3]} for r in rows]

# ----------------------------------------------------------------------
# 2. Model 2 — tissue concentrations (per 100 uM gel loading; linear in C0)
# ----------------------------------------------------------------------
t_tis = np.concatenate([[0], np.logspace(-1, 6.7, 700)])     # 0.1 s .. ~14 d
_, _, _, Cs_V = M.solve_two_domain(species="V", t_out=t_tis)
_, _, _, Cs_F = M.solve_two_domain(species="F", t_out=t_tis)

nxg = P.numerics["nx_gel"]
dxt = P.tissue["L"] / P.numerics["nx_tis"]
i_band = int(round(P.no_chain["mac_band_cm"] / dxt))          # 50 um band
n_1mm = int(round(0.10 / dxt))                                # top 1 mm
n_delta = int(round(P.tissue["delta"] / dxt))                 # fibroblast zone

cV_band = Cs_V[nxg + i_band, :]                # x C0_gel[uM] -> uM
cV_1mm = Cs_V[nxg: nxg + n_1mm, :].mean(axis=0)
cF_delta = Cs_F[nxg: nxg + n_delta, :].mean(axis=0)

fig, ax = plt.subplots(figsize=(4.6, 3.4))
floor = 1e-8                                # mask numerical noise (~1e-16)
mV = 100 * cV_band[1:] > floor
mV1 = 100 * cV_1mm[1:] > floor
mF = 100 * cF_delta[1:] > floor
ax.loglog(t_tis[1:][mV], 100 * cV_band[1:][mV], color="#0279EE",
          label="V14 at 50 $\\mu$m band")
ax.loglog(t_tis[1:][mV1], 100 * cV_1mm[1:][mV1], color="#0279EE", ls="--",
          label="V14 depth-avg top 1 mm")
ax.loglog(t_tis[1:][mF], 100 * cF_delta[1:][mF], color="#FF9400",
          label="FGF2-G3 depth-avg $\\delta$=500 $\\mu$m")
ax.set_xlabel("time (s)")
ax.set_ylabel("tissue concentration (uM per 100 uM loading)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_tissue.png"), dpi=200)
plt.close(fig)

# Days above the 25 uM design target vs loading (V14, depth-avg top 1 mm)
tgt = P.design["C_V_target_uM"]
loadings = [100, 200, 400, 738]
above_rows = []
for C0 in loadings:
    conc = C0 * cV_1mm
    idx = np.where(conc >= tgt)[0]
    if len(idx):
        t_above = (t_tis[idx[-1]] - t_tis[idx[0]]) / 86400.0
        peak = conc.max()
    else:
        t_above, peak = 0.0, conc.max()
    above_rows.append((C0, round(t_above, 2), round(peak, 1)))
with open(os.path.join(TAB, "table_loading.csv"), "w") as fh:
    fh.write("metric,value\n")
    fh.write("# days above 25 uM (V14 depth-avg top 1 mm) vs loading\n")
    fh.write("loading_uM,days_above_25uM,peak_uM\n")
    for r in above_rows:
        fh.write(",".join(map(str, r)) + "\n")
    # Model 7 bound for a delta grid
    fh.write("\n# Model 7 loading bound C_V0 >= C_tgt*eps_t*delta/(f_rel*eps_g*L_g)\n")
    fh.write("delta_um,loading_bound_uM\n")
    for d_um in (300, 500, 1000):
        b = C6.loading_bound(delta=d_um * 1e-4)
        fh.write(f"{d_um},{b:.0f}\n")
summary["days_above_25uM"] = [
    {"loading_uM": r[0], "days": r[1], "peak_uM": r[2]} for r in above_rows]
summary["loading_bound_uM"] = {
    str(d): round(C6.loading_bound(delta=d * 1e-4))
    for d in (300, 500, 1000)}

# ----------------------------------------------------------------------
# 3. Models 3-4 — scaled arm (scenario: KD_P unknown until flow cytometry)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.4))
psi = np.logspace(-2, 4, 300)
a = 1.0
for gam, col in [(0.1, "#0279EE"), (1.0, "#75A025"), (10.0, "#FF9400")]:
    _, _, pi_n = M.scaled_arm(psi, a, gamma=gam)
    ax.semilogx(psi, pi_n, color=col,
                label=f"$\\gamma$ = TLR4$_{{tot}}$/K$_{{NF}}$ = {gam}")
ax.axvline(M.psi50(a), color="grey", ls=":", lw=0.8)
ax.text(M.psi50(a) * 1.15, 0.5, "$\\psi_{50} = 1+a$", fontsize=8)
ax.set_xlabel("$\\psi$ = C/K$_{D,P}$   (a = [L]/K$_{D,L}$ = 1)")
ax.set_ylabel("normalised p65 response, $\\pi$")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_scaled_arm.png"), dpi=200)
plt.close(fig)

# Corrected required-dose inversion (Eq. 44) across the [L] knob
req_rows = []
for L_uM in (0.01, 0.1, 1.0, 10.0):
    Preq, feas = M.required_dose(L=L_uM * 1e-9 * 1e-3, KD_P=1.0)
    req_rows.append((L_uM, round(Preq, 2), bool(feas)))
summary["required_dose_xKDP"] = [
    {"L_uM": r[0], "P_req_in_KDP": r[1], "feasible": r[2]} for r in req_rows]
# The inversion is infeasible at every [L] because the p65 target (5 nM)
# requires TLR4_tot > T_tgt = 3.16 nM, outside the 0.17-1.7 nM prior:
# the absolute p65 target is never binding at these priors.
Ttgt_nM = M.TLR4_star_target() * 1e9 * 1e3   # mol/cm3 -> nM
summary["p65_target_TLR4tot_min_nM"] = round(float(Ttgt_nM), 2)

# ----------------------------------------------------------------------
# 4. Model 5 — NO / nitrite. Absolute baseline from the q_chronic QSSA;
#    trajectories show the treatment-modulated (p65-driven) delta.
# ----------------------------------------------------------------------
NO_base = float(M.nM(M.no_baseline()))
f_cal = lambda b: float(M.nM(M.no_baseline(band_cm=b))) - P.no_chain["C_w_target_nM"]
from scipy.optimize import brentq
band_cal_um = brentq(f_cal, 1e-4, 1.0) * 1e4
summary["NO_baseline_nM"] = {"band_50um": NO_base,
                             "band_for_36nM_um": round(band_cal_um)}

KD_P_scen = P.uM_to_mol_per_cm3(1.0)          # scenario: 1 uM (T4 knob)
ke0_d = P.closure["ke0_per_h"] * 24.0          # 1/d

def cV_band_of_t(t_s, C0_uM):
    """V14 at the macrophage band [mol/cm3], log-t interpolation."""
    c = C0_uM * P.uM_to_mol_per_cm3(1.0) * np.interp(
        np.log(np.maximum(t_s, t_tis[1])), np.log(t_tis[1:]), cV_band[1:])
    return c

t_days = np.concatenate([np.logspace(-3, np.log10(14), 1200)])
t_s = t_days * 86400.0
C0_load = 738.0                                # uM, solubility ceiling
cv = lambda tt: cV_band_of_t(tt, C0_load)

# untreated vs treated p65-driven trajectories (relative suppression)
t_u, NO_u, NO2_u = M.nitrite_trajectory(lambda tt: 0.0 * tt, t_s, KD_P_scen)
t_t, NO_t, NO2_t = M.nitrite_trajectory(cv, t_s, KD_P_scen)
supp = 1 - NO2_t[-1] / NO2_u[-1] if NO2_u[-1] > 0 else np.nan
summary["nitrite_suppression_14d"] = round(float(supp), 3)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
axes[0].semilogx(t_days[1:], M.nM(cv(t_s))[1:], color="#0279EE")
axes[0].set_xlabel("time (d)")
axes[0].set_ylabel("V14 at band (nM)")
axes[1].semilogx(t_days[1:], NO_u[1:], color="grey", label="untreated")
axes[1].semilogx(t_days[1:], NO_t[1:], color="#0279EE", label="treated")
axes[1].set_xlabel("time (d)")
axes[1].set_ylabel("free NO (nM, p65-driven)")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_no_nitrite.png"), dpi=200)
plt.close(fig)

with open(os.path.join(TAB, "table_no_check.csv"), "w") as fh:
    fh.write("metric,value\n")
    fh.write(f"NO_baseline_nM_at_50um_band,{NO_base:.0f}\n")
    fh.write(f"band_depth_um_reproducing_36nM,{band_cal_um:.0f}\n")
    fh.write(f"nitrite_suppression_14d_fraction,{supp:.3f}\n")

# ----------------------------------------------------------------------
# 5. Model 6 — closure scenarios
# ----------------------------------------------------------------------
# (a) doc 9.4 protocol: square therapeutic pulse (rp=2.0, d=0.4) for T,
#     then effect relaxes with a 0.5 d time constant onto the chronic base.
def pulse_rates(Tp, rp_p=2.0, d_p=0.4, tau_rel=0.5):
    r0, d0 = P.closure["r0_chronic"], P.closure["d0_chronic"]
    def rp_of_t(t):
        if t < Tp:
            return rp_p
        return r0 + (rp_p - r0) * np.exp(-(t - Tp) / tau_rel)
    def d_of_t(t):
        if t < Tp:
            return d_p
        return d0 + (d_p - d0) * np.exp(-(t - Tp) / tau_rel)
    return rp_of_t, d_of_t

Rw_grid = [0.1, 0.2, 0.3, 0.5]                 # 1, 2, 3, 5 mm
T_grid = [2, 3, 5, 8, 12, 16, 20, 25, 30, 40]
pulse_rows = []
traj_store = {}
for Rw in Rw_grid:
    min_T = None
    for Tp in T_grid:
        rp_of_t, d_of_t = pulse_rates(Tp)
        t, r_f, A_c, tc = C6.solve_closure(
            rp_of_t=rp_of_t, d_of_t=d_of_t, R_w=Rw, t_end_d=60, nr=600)
        reached = r_f.min() <= 3 * 2e-4        # within 3 cells of centre
        if reached and min_T is None:
            min_T = Tp
        if abs(Rw - 0.2) < 1e-9:
            traj_store[Tp] = (t, r_f)
    # outcome of a 5 d pulse
    rp_of_t, d_of_t = pulse_rates(5.0)
    t, r_f, A_c, tc = C6.solve_closure(
        rp_of_t=rp_of_t, d_of_t=d_of_t, R_w=Rw, t_end_d=60, nr=600)
    r5mm = r_f.min() * 1e4
    pulse_rows.append((Rw * 1e4, min_T, round(r5mm, 2)))
    print(f"R_w={Rw*1e4:.0f} um: min pulse={min_T} d, 5-d pulse min front={r5mm:.2f} um")

with open(os.path.join(TAB, "table_closure_pulse.csv"), "w") as fh:
    fh.write("R_w_um,min_pulse_d_for_closure,front_min_after_5d_pulse_um\n")
    for r in pulse_rows:
        fh.write(",".join(map(str, r)) + "\n")
summary["pulse_table"] = [
    {"R_w_um": r[0], "min_pulse_d": r[1], "front_min_5d_um": r[2]}
    for r in pulse_rows]

# chronic stall + representative trajectories at R_w = 2 mm
t_ch, r_ch, _, tc_ch = C6.solve_closure(R_w=0.2, t_end_d=60, nr=600)
summary["chronic_stall_tc_d"] = float(tc_ch)

# (b) coupled scenario demo: solubility-ceiling loading, KD_P = 1 uM scenario.
#     Full chain: release -> band concentration -> theta_LPS -> Ie; FGF
#     depth-avg -> Ce; Eqs. 58-59 rates -> front. Operator splitting is
#     legitimate: release (minutes) vs closure (weeks).
def Ce_of_t(t_d):
    cf = C0_load * P.uM_to_mol_per_cm3(1.0) * np.interp(
        np.log(np.maximum(t_d * 86400, t_tis[1])), np.log(t_tis[1:]), cF_delta[1:])
    return cf

def I_of_t(t_d):
    return M.theta_LPS(cv(t_d * 86400), KD_P_scen)

def Ie_of_t(t_d):
    # ke0 filter of I over [0, t] on a fine grid (quasi-steady check + filter)
    tt = np.linspace(0, max(t_d, 1e-3), 400)
    Ii = I_of_t(tt)
    ie = np.zeros_like(tt)
    for i in range(1, len(tt)):
        dt = tt[i] - tt[i - 1]
        ie[i] = ie[i - 1] + dt * ke0_d * (Ii[i - 1] - ie[i - 1])
    return float(np.interp(t_d, tt, ie))

def Ce_filt(t_d):
    tt = np.linspace(0, max(t_d, 1e-3), 400)
    cc = Ce_of_t(tt)
    ce = np.zeros_like(tt)
    for i in range(1, len(tt)):
        dt = tt[i] - tt[i - 1]
        ce[i] = ce[i - 1] + dt * ke0_d * (cc[i - 1] - ce[i - 1])
    return float(np.interp(t_d, tt, ce))

t_cp, r_cp, _, tc_cp = C6.solve_closure(
    Ce_of_t=Ce_filt, Ie_of_t=Ie_of_t, R_w=0.2, t_end_d=60, nr=600)
summary["coupled_scenario_tc_d"] = float(tc_cp)
print(f"coupled scenario (738 uM load, KD_P=1 uM): tc={tc_cp:.2f} d")


fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
axes[0].plot(t_ch, r_ch * 1e4, color="grey", label="chronic untreated (stall)")
for Tp, col in [(5, "#FF9400"), (12, "#75A025"), (20, "#0279EE")]:
    if Tp in traj_store:
        tt, rr = traj_store[Tp]
        axes[0].plot(tt, rr * 1e4, color=col, label=f"{Tp} d pulse")
axes[0].plot(t_cp, r_cp * 1e4, color="#FD9BED",
             label="coupled bolus (738 uM, KD$_P$=1 uM)")
axes[0].set_xlabel("time (d)")
axes[0].set_ylabel("front position r_f (um)")
axes[0].set_xlim(0, 40)
axes[0].set_ylim(0, 2100)          # clip the post-closure jump to the reservoir
axes[0].legend(frameon=False, fontsize=8)
axes[1].plot([r[0] for r in pulse_rows], [r[1] or 45 for r in pulse_rows], "o-",
             color="#0279EE")
axes[1].set_xlabel("wound radius R_w (um)")
axes[1].set_ylabel("minimum pulse for closure (d)")
axes[1].set_xticks([1000, 2000, 3000, 5000])
axes[1].set_xticklabels(["1000", "2000", "3000", "5000"])
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_closure.png"), dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------
# 6. Copy deliverables and save summary
# ----------------------------------------------------------------------
with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1, default=str)

dest = "/mnt/results/v3_results"
os.system(f"rm -rf {dest} && cp -r {OUT} {dest}")
print("DONE — outputs in /mnt/results/v3_results")
print(json.dumps(summary, indent=1, default=str))
