"""
params.py — SINGLE SOURCE OF TRUTH for the iGEM V3 bio-patch model.
====================================================================
Edit any value here and rerun run_all.py; every solver reads only from
this file. No parameter is hard-coded anywhere else.

Units: cm, s, mol, mol/cm^3 (interstitial / pore-fluid basis).
Concentration conversions:
    1 M = 1 mol/L = 1e-3 mol/cm^3
    1 uM = 1e-6 mol/L = 1e-9 mol/cm^3
    V14: 1 uM = 1.356 ug/mL   (MW 1355.59 Da)
    FGF2-G3: 1 uM = 16.49 ug/mL (MW 16487.87 Da)

Provenance tiers (document Appendix D convention):
    T1  measured in house
    T2  mechanistic correlation
    T3  named literature analogue / cited value
    T4  order-of-magnitude prior (log-uniform over >= 1 decade)

Each entry: VALUE, unit, tier, source, note.
Locked assumptions (user decision, 2026-10):
    - NO heparin anywhere: S_max = 0, R_F = 1 (both peptides linear Fickian).
    - NO MTT / qPCR data for V14: Model 6 couplings stay T4 scenario priors;
      no fitted pharmacology.
"""

import math

# ----------------------------------------------------------------------
# 0. Physical constants
# ----------------------------------------------------------------------
T_body = 310.0            # K, 37 C
k_B = 1.380649e-23        # J/K
eta_water_37 = 6.9e-4     # Pa s, water viscosity at 37 C [table, Anton-Paar]

def stokes_einstein(rh_nm):
    """D [cm^2/s] from hydrodynamic radius [nm] at 37 C."""
    r_m = rh_nm * 1e-9
    D_m2s = k_B * T_body / (6 * math.pi * eta_water_37 * r_m)
    return D_m2s * 1e4  # m^2/s -> cm^2/s

# ----------------------------------------------------------------------
# 1. Species properties
# ----------------------------------------------------------------------
V14 = dict(
    name="V14 (alphaTI-14)",
    seq="SVQELLELLAAGL",
    MW=1355.59,           # Da [Expasy; table]
    Rh=0.73,              # nm, globular-volume equivalent radius [table]
                          #   (doc correlation gives 0.86-0.87 nm; carried as
                          #    sensitivity upper bound -> D0 range below)
    charge=-2.02,         # at pH 7.4, Henderson-Hasselbalch [doc App. A]
    pI=3.48,              # [doc App. A]
    GRAVY=1.16,           # strongly hydrophobic [doc App. A]
    t_half_gel_h=6.1,     # h. BASE = Cavaco serum regression (in-house notebook);
                          #   gel k_deg is T1 "to be measured" (doc App. D).
                          #   Expasy N-end rule (1.9 h) is intracellular, not used.
    t_half_tissue_h=6.1,  # h, proteolysis proxy = serum t_half (k_prot,V is the
                          #   highest-priority unmeasured parameter, doc App. D/A.3)
    solubility_uM=738.0,  # uM = 1 mg/mL assumed ceiling [doc 9.5]; MEASURE THIS.
    KD_P=None,            # mol/cm^3. NOT KNOWN — no sequence method (doc App. A).
                          #   Model 3-5 run in scaled form (psi, a, gamma) until
                          #   the flow-cytometry panel returns it.
)

FGF = dict(
    name="FGF2-G3",
    MW=16487.87,          # Da [Addgene/Expasy; table]
    Rh=1.68,              # nm, globular-volume equivalent radius [table]
                          #   (doc correlation 2.34 nm -> sensitivity upper bound)
    t_half_37_h=168.0,    # h, functional half-life at 37 C [Reprocell QK053; table]
                          #   doc bound: k_deg <= 4e-7/s from ~20 d no-loss claim
    KD_FGFR=None,         # not needed: FGF arm enters closure via EC50,F prior
)

# Free-solution diffusivities at 37 C [cm^2/s]
D0_V = stokes_einstein(V14["Rh"])          # ~4.5e-6
D0_F = stokes_einstein(FGF["Rh"])          # ~2.0e-6
# Document correlation values (T2, three routes): 3.6e-6 / 1.5e-6.
# Discrepancy recorded; Stokes-Einstein from table Rh used as BASE.
D0_V_doc, D0_F_doc = 3.6e-6, 1.5e-6

# ----------------------------------------------------------------------
# 2. Gel (Model 1) — sodium alginate / CaCl2, as-manufactured
#
# hydrogel is calcium-crosslinked sodium alginate. Alginate is a charged polysaccharide; its mesh
# size, porosity and tortuosity differ from GelMA and depend on the
# SA:CaCl2 ratio. Values below are a "possible" set consistent with
# 1-2% w/v SA crosslinked with 50-200 mM CaCl2. REPLACE with your own
# measured values when the swelling / release assay is done.
# ----------------------------------------------------------------------
gel = dict(
    L=500e-4,             # cm (500 um). Alginate gels are usually cast
                          #   0.5-2 mm thick. A thicker gel slows release
                          #   by L^2 (Eq. 22: tau_late = 4 L^2 / pi^2 D).
    eps=0.85,             # porosity: alginate hydrogels are highly
                          #   hydrated. Lower than GelMA 10% baseline?
                          #   No — alginate holds more water. 0.85 is a
                          #   mid-range value (literature: 0.75-0.95).
    tau=1.15,             # Bruggeman eps^-0.5 = 0.85^-0.5 = 1.085,
                          #   rounded up for the extra tortuosity of a
                          #   Ca-crosslinked network. Literature range
                          #   1.05-1.30.
    mesh_nm=8.0,          # nm, mesh size xi. Alginate mesh depends on
                          #   SA and CaCl2 concentration:
                          #     1% SA / 100 mM CaCl2  -> ~20 nm
                          #     2% SA / 200 mM CaCl2  -> ~4-5 nm
                          #   8 nm is a mid-range, moderately
                          #   crosslinked formulation. Smaller than this
                          #   (e.g. 5 nm) needs higher SA/CaCl2 and is
                          #   harder to justify for a first pass.
    a_f=0.6,              # nm, polymer chain radius. Alginate
                          #   backbone is thicker than gelatin; 0.6 nm
                          #   is a conservative lower bound.
    use_doc_baseline=False,
)
if gel["use_doc_baseline"]:
    gel.update(L=500e-4, eps=0.91, tau=0.91 ** -0.5)

def amsden_H(a_i_nm, xi_nm, a_f_nm=0.6):
    """Amsden obstruction factor H = exp(-pi*((a_i+a_f)/(xi+2*a_f))^2)."""
    return math.exp(-math.pi * ((a_i_nm + a_f_nm) / (xi_nm + 2 * a_f_nm)) ** 2)

H_V = amsden_H(V14["Rh"], gel["mesh_nm"])   # ~0.96 at xi=10 nm
H_F = amsden_H(FGF["Rh"], gel["mesh_nm"])   # ~0.88 at xi=10 nm

Deff_g_V = D0_V * H_V / gel["tau"]          # cm^2/s
Deff_g_F = D0_F * H_F / gel["tau"]

# In-gel degradation [1/s]
k_deg_g_V = math.log(2) / (V14["t_half_gel_h"] * 3600)
k_deg_g_F = math.log(2) / (FGF["t_half_37_h"] * 3600)   # ~1.15e-6

# ----------------------------------------------------------------------
# 3. Tissue (Model 2) — Dirichlet systemic sink, NO distributed kcl
#    (doc 3.4/9.7: never double-count clearance)
# ----------------------------------------------------------------------
tissue = dict(
    L=0.15,               # cm (1.5 mm skin) [table, NBK144027]
    eps=0.8,              # interstitial volume fraction [table]
    tau=0.8 ** -0.5,      # Bruggeman = 1.118
    Deff_t_V=1.2e-6,      # cm^2/s base (doc range 0.7-1.8e-6, T3)
    Deff_t_F=0.5e-6,      # cm^2/s (T4; larger molecule, anionic ECM retention)
    k_prot_V=math.log(2) / (6.1 * 3600),  # 1/s, serum t_half proxy (T4; sweep 1e-6..1e-3)
    k_prot_F=math.log(2) / (168.0 * 3600),# FGF2-G3 proteolysis ~ thermal t_half (T4)
    Kp=1.0,               # gel/tissue partition (T4 +-1 order; table's
                          #   "AI calculation" NOT verifiable -> neutral 1.0 base)
    delta=500e-4,         # cm, fibroblast-active zone depth (T4; doc sweeps 300-1000 um)
)

# Microvascular clearance [1/s] — NOT used in the baseline PDE (Dirichlet sink
# replaces it, doc 3.4). Kept for sensitivity runs only.
k_cl_F_sens = 1.15e-4   # [table 2026-10-08 rev; PMC2764268] (was 1.35e-4 in
                        #   the Oct-6 table — updated to the new table value)
k_cl_V_sens = 3.5e-3    # [table, "AI calculation" — UNVERIFIED, T4]

# ----------------------------------------------------------------------
# 4. Model 3 — MD2/TLR4 competition
# ----------------------------------------------------------------------
md2 = dict(
    KD_L=65e-9 * 1e-3,    # mol/cm^3. LPS-MD2 apparent KD ~ 65 nM
                          #   [Viriyakosol 2000, J Endotoxin Res 6:131]. T3.
                          #   LPS aggregation/CD14 caveats -> order-of-magnitude.
    L_uM=1.0,             # uM, wound endotoxin load (T4 KNOB — unconstrained;
                          #   sweep decades). a = [L]/KD_L is what enters the
                          #   model. NO measured wound-exudate LPS value exists
                          #   (paucity of clinical data — Rippon 2022, J Wound
                          #   Care 31:380, DOI 10.12968/jowc.2022.31.5.380);
                          #   1 uM is the worst-case anchor and sets the
                          #   untreated theta_LPS = 0.94 used to calibrate the
                          #   closure death rates (see Model 6 calibration).
    TLR4_tot_nM=1.0,      # nM, total TLR4/MD2 (doc range 0.17-1.7 nM, T4)
)

# ----------------------------------------------------------------------
# 5. Model 4 — NF-kB signaling (QSSA)
# ----------------------------------------------------------------------
signaling = dict(
    K_NF_nM=10.0,         # nM, p65 half-maximal activation [table, JBC]
    h=2.0,                # Hill coefficient, p65 dimer [table, JBC]
    p65_bas_nM=0.5,       # nM, basal nuclear p65 (T4)
    p65_max_nM=50.0,      # nM, maximal nuclear p65 (T4)
    p65_target_nM=5.0,    # nM, therapeutic target (T4 knob)
)

# ----------------------------------------------------------------------
# 6. Model 5 — iNOS / NO / nitrite
# ----------------------------------------------------------------------
no_chain = dict(
    q_chronic=5.0e-19,    # mol/cell/s, sustained iNOS rate, mid of 3.5-8.1e-19
                          #   [table, PNAS 94:11875]. T3.
    q_max_acute=8.17e-17, # mol/cell/s, acute ceiling [table, Am J Physiol Cell]
                          #   NOTE: V2's 1e-14 was ~1e4x too high (doc 6.2).
    N_mac_per_mm2=262.0,  # cells/mm2, chronic baseline [table, DOI 10.1177/1534734620945559]
    mac_band_cm=50e-4,    # cm (50 um) macrophage infiltration band depth.
                          #   CALIBRATION: band depth x q_chronic / k_scav sets
                          #   the baseline NO level (see C_w below).
    k_scav=0.01,          # 1/s, first-order NO clearance [table, SAGE/Redox].
                          #   Doc alternative: t_half 3-10 s -> k 0.07-0.23/s
                          #   (autoxidation is 2nd order; keep as sensitivity).
    # --- New entries from the 2026-10-08 parameter table ---
    q_iNOS_Griess=2.3e-19,# mol/cell/s, per-cell iNOS rate from Griess assay on
                          #   LPS/IFN-g macrophages (2.3 +- 0.6 pmol/s/1e6 cells)
                          #   [table; PMC2879621]. Independent anchor for
                          #   q_chronic (5e-19 sits between this and q_max_acute).
    N_mac_M1=230.0,       # cells/mm2, M1 macrophage density, chronic wound
                          #   (230 +- 42) [table; pubmed 32815405]. Sensitivity
                          #   variant of N_mac_per_mm2.
    N_mac_chronic_day30=241.0,  # cells/mm2, chronic day-30 [DOI 10.1177/
                          #   1534734620945559]; ~8% decay from day-0 262 ->
                          #   supports treating N_mac as quasi-stationary.
    t_half_NO_s=(0.09, 2.0),  # s, NO half-life range in normoxic tissue
                          #   [table; pubmed 11134509] (replaces doc's 3-10 s).
                          #   -> k_scav range 0.35-7.7 /s; the baseline
                          #   k_scav=0.01 /s is the SAGE/Redox wound value and
                          #   is retained as BASE (see note above).
    k_turn=math.log(2) / (4 * 3600),  # 1/s, iNOS turnover, tau ~ 2-6 h (T4)
    K_iNOS_frac=0.3,      # iNOS induction threshold as fraction of [p65]max (T4)
    p_iNOS=2.0,           # Hill coefficient at NOS2 locus (T4)
    phi=0.9,              # NO -> nitrite stoichiometric yield in medium (T4)
    C_w_target_nM=36.0,   # nM, team's calculated wound NO baseline [table] —
                          #   used as a self-consistency CHECK, not an input.
)

# ----------------------------------------------------------------------
# 7. Model 6 — Fisher-KPP closure (all coupling constants T4 priors;
#    NO MTT/qPCR data — scenario analysis only, never fitted)
#
# CALIBRATION TARGET (rev 2026-10-09): the untreated chronic wound must be
# STALLED, not degenerating. Fisher-KPP with rp <= d has extinction as its
# only steady state (n* = K(1 - d/rp) <= 0), so the original d0=1.0 +
# Delta_d=0.7 drove net rates of -0.1 to -1.1 /d: the model predicted the
# fibroblast population dies everywhere (front detection then tracks the
# pinned reservoir node — meaningless trajectories). Real chronic wounds
# hold viable, non-invading granulation tissue. d0 and Delta_d are set so
#   net_untreated = rp(Ce=0, Ie=0.94) - d(Ie=0.94) = 0.497 - 0.51 ~ -0.01/d
# (Ie=0.94 = untreated theta_LPS at [L]=1 uM). Normal-wound rates
# (r0=0.8, d0=0.24) then heal; chronicity enters ONLY through inflammation.
# ----------------------------------------------------------------------
closure = dict(
    D_n=1e-9,             # cm^2/s, fibroblast motility (T4; bounded by clinical
                          #   granulation rates 0.1-0.5 mm/d, doc 7.5)
    K_cells=5e6,          # cells/cm3, confluent carrying capacity (T4)
    r0_chronic=0.8,       # 1/d, basal proliferation in chronic wound (T4;
                          #   fibroblast generation time 32-39 h in repair
                          #   [Raff & Houck 1969] -> ~0.4-0.5/d naive; chronic
                          #   granulation tissue proliferative attempt higher)
    d0_chronic=0.24,      # 1/d, basal death rate (T4; CALIBRATED — see target
                          #   above; was 1.0, which double-counted chronicity
                          #   on top of the inflammation term and forced
                          #   population extinction)
    E_max=2.0,            # fold-maximum FGF efficacy on proliferation (T4, no
                          #   MTT; 1+E_max = 3x total, consistent with FGF2
                          #   mitogenesis dose-response [Benington 2024,
                          #   Pharmaceuticals 17:247; Zhu 2010, Cell Commun
                          #   Signal 8:14 — optimum ~0.3 ng/mL, bell-shaped])
    EC50_F_ngmL=0.5,      # ng/mL, FGF2 proliferative EC50 on 3T3 (T4, no MTT;
                          #   literature FGF2 EC50 ~0.1-1 ng/mL)
    beta_r=0.5,           # inflammation penalty on proliferation (T4)
    K_I_r=0.3,            # scaled Ie threshold for proliferation penalty (T4)
    Delta_d=0.30,         # 1/d, max inflammation-driven death increment (T4;
                          #   CALIBRATED — was 0.7, see target above)
    K_I_d=0.3,            # scaled Ie threshold for death increment (T4)
    s_I=2.0,              # Hill exponent of death response (T4)
    ke0_per_h=1.0 / 18,   # 1/h, effect-compartment rate (T4, 1/(12-24 h))
    R_w_cm=0.2,           # cm, wound radius (grid: 1-5 mm)
    w_smooth_cm=75e-4,    # cm, initial-condition smoothing width (50-100 um)
    n_thr=0.5,            # closure threshold as fraction of n_ss
    T_pulse_d=None,       # d, therapeutic pulse duration (None = permanent)
)

# ----------------------------------------------------------------------
# 8. Model 7 — design / loading bound
# ----------------------------------------------------------------------
design = dict(
    C_V_target_uM=25.0,   # uM, efficacious V14 concentration (flow-cytometry
                          #   anchor; consistent with competing [L]=1 uM at
                          #   KD_P ~ 1 uM scenario)
    f_rel=0.5,            # fraction of loaded dose released (T4)
    reapply_interval_d=1.0,  # d, dressing-change interval for the reapplication
                          #   protocol scenario (daily changes are standard
                          #   clinical practice for wound dressings)
)

# ----------------------------------------------------------------------
# 9. LEGACY living-chassis sensing module — NOT used in the acellular V3
#    model. Kept verbatim from the 2026-10-08 parameter table so the
#    constants survive if the engineered-sensor arm is revived.
# ----------------------------------------------------------------------
legacy_sensing = dict(
    # Cellulose protective layer (living chassis)
    L_c=50e-4,            # cm (50 um), cellulose layer thickness [table]
    eps_c=0.94,           # porosity of cellulose layer [table]
    tau_c=1.03,           # tortuosity of cellulose layer [table]
    # NorR NO sensor / P_norV promoter (E. coli)
    alpha_0=4e-4,         # dimensionless basal promoter activity (0.04%) [table]
    K_A=50e-9 * 1e-3,     # mol/cm^3, NorR activation constant (50 nM) [table]
    n_NorR=3.0,           # Hill coefficient of P_norV [table]
    k_deg_m=2.1e-3,       # 1/s, mRNA degradation [table]
    v_transcription=3.6,  # mRNA/cell/s, max transcription rate [table]
    k_translation=0.11,   # 1/s, translation rate [table]
    k_secretion=0.08,     # 1/s, protein secretion rate [table]
    k_deg_p=math.log(2) / (1.9 * 3600),  # 1/s, protein degradation; ExPASy
                          #   N-end rule half-life 1.9 h [table]
)

# ----------------------------------------------------------------------
# 10. Numerics
# ----------------------------------------------------------------------
numerics = dict(
    nx_gel=400, nx_tis=400,   # grid points (2nd-order solver; doc used up to 1600)
    t_end_release=3600.0,     # s, release simulation horizon
    n_t_release=400,
    dr=2e-4,                  # cm, radial grid for closure (0.5 mm wound -> 1000 pts)
    t_end_closure=60.0,       # days
    n_t_closure=6000,
)

# ----------------------------------------------------------------------
# Convenience: unit conversions
# ----------------------------------------------------------------------
def uM_to_mol_per_cm3(uM):
    return uM * 1e-9

def mol_per_cm3_to_nM(c):
    return c * 1e3 * 1e9 / 1e3  # mol/cm3 -> mol/L -> nM  (1 mol/cm3 = 1000 mol/L)

def ngmL_to_mol_per_cm3(ngmL, MW):
    return ngmL * 1e-9 * 1e-3 / MW  # ng/mL -> g/cm3 -> mol/cm3
