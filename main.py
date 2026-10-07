"""
BEMT for the Caradonna & Tung rotor in hover: complete analyses.

Order of the analyses:
    1. Baseline models at 8 deg (uniform inflow, BEMT without tip loss, BEMT with Prandtl)
    2. BEMT with Prandtl against the experiment at 8 deg (Cl distribution)
    3. BEMT with Prandtl at several pitch angles
    4. Correction for the preceding blade's tip vortex at 8 deg
    5. Grid convergence study (uniform versus half-cosine)
    6. Sensitivity to the vortex position (+-0.025R)
    7. Sensitivity tests: without Prandtl, without mean downwash, compressibility, viscous core
    8. Predictive validation at 5, 8 and 12 deg
    9. Figures (saved in the "figure" folder)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from modello import (R, OMEGA, A_SUONO, griglia_uniforme, griglia_semicoseno,
                     coefficiente_spinta, errore_pct, inflow_uniforme,
                     bemt_senza_perdite, bemt_prandtl, bemt_vortice, cl_prandtl)
from dati_sperimentali import X_STAZIONI, CASI, K_VORTICE_5

# Main grid: half-cosine with 200 stations (chosen from the study in section 5)
x = griglia_semicoseno(200)

# Baseline case: 8 deg
caso8 = CASI[8]
theta8 = np.radians(8)


def stampa_risultato(etichetta, CT, CT_exp, Cl_stazioni=None, Cl_exp=None):
    """Print C_T with its error and, if given, the Cl errors at the stations."""
    print(f"{etichetta}: C_T = {CT:.5f}  (error {errore_pct(CT, CT_exp):+.1f}%)")
    if Cl_stazioni is not None:
        err = errore_pct(Cl_stazioni, Cl_exp)
        print(f"    Cl error at the stations [%]: {np.round(err, 1)}"
              f"   mean absolute error: {np.mean(np.abs(err)):.1f}%")


# %% 1. Baseline models at 8 deg
print("\n=== 1. Baseline models at 8 deg ===")
lam_unif, CT_unif = inflow_uniforme(theta8)
print(f"Uniform inflow: lambda = {lam_unif:.4f}")
stampa_risultato("Uniform inflow", CT_unif, caso8["CT"])

lam_sp = bemt_senza_perdite(x, theta8)
stampa_risultato("BEMT without tip loss", coefficiente_spinta(x, lam_sp), caso8["CT"])

lam_pr, F_pr, it_pr = bemt_prandtl(x, theta8)
CT_pr = coefficiente_spinta(x, lam_pr, F_pr)
print(f"BEMT with Prandtl: converged in {it_pr} iterations")
stampa_risultato("BEMT with Prandtl", CT_pr, caso8["CT"])


# %% 2. BEMT with Prandtl against the experiment at 8 deg
# Without the vortex the annuli are independent: the model is evaluated directly
# at the experimental stations, without interpolation.
print("\n=== 2. BEMT with Prandtl against the experiment at 8 deg ===")
lam_st, F_st, _ = bemt_prandtl(X_STAZIONI, theta8)
err_Cl_pr = errore_pct(cl_prandtl(X_STAZIONI, theta8, lam_st), caso8["Cl"])
print(f"Cl error at the stations [%]: {np.round(err_Cl_pr, 1)}")


# %% 3. BEMT with Prandtl at several pitch angles
print("\n=== 3. BEMT with Prandtl at several pitch angles ===")
for th_deg, caso in CASI.items():
    lam_th, F_th, it_th = bemt_prandtl(x, np.radians(th_deg))
    stampa_risultato(f"{th_deg:>2} deg", coefficiente_spinta(x, lam_th, F_th), caso["CT"])

# Continuous C_T(theta) curve of the model, for the plot
theta_curva = np.arange(1, 14.01, 0.5)
CT_curva = np.array([coefficiente_spinta(x, *bemt_prandtl(x, np.radians(th))[:2])
                     for th in theta_curva])


# %% 4. Correction for the preceding blade's tip vortex at 8 deg
print("\n=== 4. BEMT with tip vortex at 8 deg ===")
lam_v, Cl_v, F_v, gamma_v, it_v = bemt_vortice(x, theta8, caso8["r_v"], caso8["z_v"])
CT_v = coefficiente_spinta(x, lam_v, F_v)
print(f"Converged in {it_v} iterations, Gamma_max = {gamma_v:.3f} m^2/s")
stampa_risultato("BEMT with vortex", CT_v, caso8["CT"],
                 np.interp(X_STAZIONI, x, Cl_v), caso8["Cl"])


# %% 5. Grid convergence study
# The reference is the most accurate grid available (half-cosine, 4000 points).
# Near the tip the loading behaves as sqrt(1-x): with a uniform grid the trapezoidal
# rule converges with order ~1.5, with the half-cosine grid much faster.
print("\n=== 5. Grid convergence study (8 deg, with vortex) ===")


def CT_vortice_8(xg):
    lam, _, F, _, _ = bemt_vortice(xg, theta8, caso8["r_v"], caso8["z_v"])
    return coefficiente_spinta(xg, lam, F)


CT_rif = CT_vortice_8(griglia_semicoseno(4000))
lista_N = np.array([50, 100, 200, 400, 800, 1000])
err_uni = np.array([abs(CT_vortice_8(griglia_uniforme(N)) - CT_rif) / CT_rif for N in lista_N])
err_cos = np.array([abs(CT_vortice_8(griglia_semicoseno(N)) - CT_rif) / CT_rif for N in lista_N])
print(f"Reference C_T (half-cosine, N = 4000) = {CT_rif:.7f}")
for N, eu, ec in zip(lista_N, err_uni, err_cos):
    print(f"N = {N:>4}   relative error uniform = {eu:.1e}   half-cosine = {ec:.1e}")


# %% 6. Sensitivity to the vortex position (+-0.025R, as in the report)
# Note: picking the best combination would be a calibration, not a validation.
print("\n=== 6. Sensitivity to the vortex position (8 deg) ===")
for r_v in [0.880, 0.905, 0.930]:
    for z_v in [0.065, 0.090, 0.115]:
        lam_s, Cl_s, F_s, _, _ = bemt_vortice(x, theta8, r_v, z_v)
        stampa_risultato(f"r_v = {r_v:.3f}, z_v = {z_v:.3f}", coefficiente_spinta(x, lam_s, F_s),
                         caso8["CT"], np.interp(X_STAZIONI, x, Cl_s), caso8["Cl"])


# %% 7. Sensitivity tests on the vortex model
print("\n=== 7. Sensitivity tests ===")

# Test A: vortex without Prandtl. The tip loading no longer vanishes and grows strongly:
# the Prandtl factor is still needed with the vortex (this does not prove that the two
# effects are independent: both partly describe the tip wake).
lam_A, Cl_A, F_A, _, _ = bemt_vortice(x, theta8, caso8["r_v"], caso8["z_v"], usa_prandtl=False)
stampa_risultato("Test A, without Prandtl (8 deg)", coefficiente_spinta(x, lam_A, F_A), caso8["CT"],
                 np.interp(X_STAZIONI, x, Cl_A), caso8["Cl"])

# Test B: vortex without its disk-averaged downwash (double counting with the actuator disk).
lam_B, Cl_B, F_B, _, _ = bemt_vortice(x, theta8, caso8["r_v"], caso8["z_v"], togli_media=True)
stampa_risultato("Test B, without mean downwash (8 deg)", coefficiente_spinta(x, lam_B, F_B), caso8["CT"],
                 np.interp(X_STAZIONI, x, Cl_B), caso8["Cl"])

# Compressibility: Prandtl-Glauert correction of the lift-curve slope.
lam_co, Cl_co, F_co, _, _ = bemt_vortice(x, theta8, caso8["r_v"], caso8["z_v"], comprimibile=True)
stampa_risultato("Prandtl-Glauert (8 deg)", coefficiente_spinta(x, lam_co, F_co), caso8["CT"],
                 np.interp(X_STAZIONI, x, Cl_co), caso8["Cl"])

# Viscous core (Scully) at 5 deg, where the vortex is closest to the blade.
caso5 = CASI[5]
for r_c in [0.0, 0.01, 0.02]:
    lam_rc, Cl_rc, F_rc, _, _ = bemt_vortice(x, np.radians(5), caso5["r_v"], caso5["z_v"], r_c=r_c)
    stampa_risultato(f"Viscous core r_c = {r_c:.2f} (5 deg)", coefficiente_spinta(x, lam_rc, F_rc),
                     caso5["CT"], np.interp(X_STAZIONI, x, Cl_rc), caso5["Cl"])


# %% 8. Predictive validation at 5, 8 and 12 deg
# Vortex positions read from the report and never adjusted; strength k = 1 at all angles.
print("\n=== 8. Predictive validation ===")
risultati = {}
for th_deg, caso in CASI.items():
    th = np.radians(th_deg)
    lam_p, F_p, _ = bemt_prandtl(x, th)
    lam_c, Cl_c, F_c, _, _ = bemt_vortice(x, th, caso["r_v"], caso["z_v"])
    risultati[th_deg] = {"Cl_pr": cl_prandtl(x, th, lam_p), "Cl_v": Cl_c,
                         "CT_v": coefficiente_spinta(x, lam_c, F_c)}

    print(f"--- {th_deg} deg ---")
    stampa_risultato("  Prandtl", coefficiente_spinta(x, lam_p, F_p), caso["CT"],
                     np.interp(X_STAZIONI, x, risultati[th_deg]["Cl_pr"]), caso["Cl"])
    stampa_risultato("  Vortex", risultati[th_deg]["CT_v"], caso["CT"],
                     np.interp(X_STAZIONI, x, Cl_c), caso["Cl"])

    if th_deg == 5:
        lam_k, _, F_k, _, _ = bemt_vortice(x, th, caso["r_v"], caso["z_v"], k=K_VORTICE_5)
        stampa_risultato(f"  Vortex, k = {K_VORTICE_5}", coefficiente_spinta(x, lam_k, F_k), caso["CT"])


# %% 9. Figures
os.makedirs("figure", exist_ok=True)

# Local Mach number along the blade
plt.figure()
plt.plot(x, OMEGA * x * R / A_SUONO)
plt.xlabel("r/R")
plt.ylabel("Local Mach number")
plt.title("Local Mach number along the blade")
plt.grid()
plt.tight_layout()
plt.savefig("figure/mach.png", dpi=200)

# Spanwise loading at 8 deg
plt.figure()
plt.plot(x, 4 * lam_sp**2 * x, label="BEMT without tip loss")
plt.plot(x, 4 * F_pr * lam_pr**2 * x, label="BEMT with Prandtl")
plt.plot(x, 4 * F_v * lam_v**2 * x, label="BEMT with tip vortex")
plt.xlabel("r/R")
plt.ylabel("dC_T/dx")
plt.title("Spanwise loading, θ = 8°")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figure/carico_8deg.png", dpi=200)

# Sectional Cl along the blade at 8 deg
plt.figure()
plt.plot(x, cl_prandtl(x, theta8, lam_sp), label="BEMT without tip loss")
plt.plot(x, cl_prandtl(x, theta8, lam_pr), label="BEMT with Prandtl")
plt.plot(x, Cl_v, label="BEMT with tip vortex")
plt.plot(X_STAZIONI, caso8["Cl"], "o", label="Experiment")
plt.xlabel("r/R")
plt.ylabel("Cl")
plt.title("Sectional lift coefficient, θ = 8°")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figure/cl_8deg.png", dpi=200)

# C_T versus collective pitch
plt.figure()
th_lista = list(CASI.keys())
plt.plot(theta_curva, CT_curva, label="BEMT with Prandtl")
plt.plot(th_lista, [risultati[t]["CT_v"] for t in th_lista], "s", label="BEMT with tip vortex")
plt.plot(th_lista, [CASI[t]["CT"] for t in th_lista], "x", ms=8, label="Experiment")
plt.xlabel("θ [°]")
plt.ylabel("C_T")
plt.title("Thrust coefficient versus collective pitch")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figure/ct_theta.png", dpi=200)

# Grid convergence study
plt.figure()
plt.loglog(lista_N, err_uni, "o-", label="Uniform grid")
plt.loglog(lista_N, err_cos, "s-", label="Half-cosine grid")
plt.loglog(lista_N, err_uni[0] * (lista_N / lista_N[0])**-1.5, "k--", lw=0.8, label="slope -1.5")
plt.loglog(lista_N, err_uni[0] * (lista_N / lista_N[0])**-2, "k:", lw=0.8, label="slope -2")
plt.xlabel("Number of stations N")
plt.ylabel("Relative error on C_T")
plt.title("Convergence of the C_T integral")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("figure/convergenza.png", dpi=200)

# Validation: Cl distribution at the three angles
fig, assi = plt.subplots(1, 3, figsize=(13, 4))
for ax, th_deg in zip(assi, CASI):
    ax.plot(x, risultati[th_deg]["Cl_pr"], label="BEMT with Prandtl")
    ax.plot(x, risultati[th_deg]["Cl_v"], label="BEMT with tip vortex")
    ax.plot(X_STAZIONI, CASI[th_deg]["Cl"], "o", label="Experiment")
    ax.set_title(f"θ = {th_deg}°")
    ax.set_xlabel("r/R")
    ax.grid()
assi[0].set_ylabel("Cl")
assi[0].legend()
fig.tight_layout()
fig.savefig("figure/validazione_cl.png", dpi=200)

plt.show()
