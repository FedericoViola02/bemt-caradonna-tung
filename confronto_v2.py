"""
Step 4: comparison with the experiment of the vortex models, at 5, 8 and 12 deg.

Models (all with the same BEMT + Prandtl code, see modello_v2.py):
    P   BEMT with Prandtl only
    V1  2D straight vortex at the measured position of age 180 deg (v1)
    E1  full 3D helix                      -> counts the axisymmetric wake twice
    E2  helix minus its azimuthal average  -> only the effect of the concentrated vortex
Additional declared checks (not used to pick a model):
    - E2 with the measured vortex strength A = 0.74 at 5 deg (TM-81232, Fig. 12)
    - E2 with another core: r_c = 0.0047R (older vortex, TM-81316 Fig. 17) and n = 1
    - V1 with its disk-averaged downwash removed (v1 "Test B")
No parameter is adjusted to the data.

Run:  python confronto_v2.py      (about 1 minute; prints tables, saves figures)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from dati_sperimentali import CASI, SCIA, R1_NUCLEO, X_STAZIONI
from modello import (R, OMEGA, griglia_semicoseno, coefficiente_spinta, errore_pct,
                     bemt_prandtl, bemt_vortice, cl_prandtl)
from stile_figure import applica_stile, stile_modello, stile_esperimento, GRIGIO_ASSI
from modello_v2 import (downwash_unitario_2d, downwash_unitario_elica,
                        downwash_medio_azimutale, bemt_downwash_vortice)

# ---------------------------------------------------------------
# Configuration (record of every numerical and model choice)
# ---------------------------------------------------------------
N_STAZIONI = 200            # half-cosine grid, as v1
N_PALE = 2
DPSI = 1.25                 # deg, step 2 study: error < 1e-3
PSI_MAX_E2 = 1440.0         # deg (4 turns), step 2 study: E2 converged to 2e-4
PSI_MAX_E1 = 5760.0         # deg (16 turns), step 2 study: E1 error ~1e-2 per unit Gamma
PSI_LINEARE = 180.0         # start of the linear descent (Fig. 10)
N_VATISTAS = 2              # Bagai-Leishman core shape
N_AZIMUT = 72               # azimuthal average: 72 -> 1152 changes C_T by 3e-4 relative
K_VORTICE = 1.0             # vortex strength / Gamma_max (TM-81232 p. 5, TM-81316 p. 5)
A_MISURATO_5 = 0.74         # measured at 5 deg, TM-81232 Fig. 12 (declared check only)
R_C_VECCHIO = 0.0047        # r1 of an older vortex (age 202 deg), TM-81316 Fig. 17

x = griglia_semicoseno(N_STAZIONI)


def risultato(lam, Cl, F):
    """C_T from the momentum balance and Cl interpolated at the experimental stations."""
    return coefficiente_spinta(x, lam, F), np.interp(X_STAZIONI, x, Cl)


def downwash_E(theta_deg, r_c, n_vat, psi_max, togli_media):
    w = downwash_unitario_elica(x, SCIA[theta_deg], N_PALE, r_c, n_vat, DPSI, psi_max,
                                PSI_LINEARE)
    if togli_media:
        w = w - downwash_medio_azimutale(x, SCIA[theta_deg], N_PALE, r_c, n_vat, DPSI,
                                         psi_max, PSI_LINEARE, N_AZIMUT)
    return w


def stampa(etichetta, CT, Cl, caso):
    e_cl = errore_pct(Cl, caso["Cl"])
    print(f"  {etichetta:28s} C_T = {CT:.5f} ({errore_pct(CT, caso['CT']):+6.1f}%)   "
          f"Cl err [%] = {np.array2string(np.round(e_cl, 0), separator=' ')}")


if __name__ == "__main__":
    tutti = {}
    for th_deg, caso in CASI.items():
        th = np.radians(th_deg)
        r_c = R1_NUCLEO[th_deg]
        print(f"\n=== {th_deg} deg   C_T exp = {caso['CT']}   Cl exp = {caso['Cl']} "
              f"at r/R = {X_STAZIONI} ===")
        ris, lam_v = {}, {}

        lam, F, _ = bemt_prandtl(x, th)
        Cl = cl_prandtl(x, th, lam)
        ris["P"] = (*risultato(lam, Cl, F), Cl)
        w = {
            "V1": downwash_unitario_2d(x, caso["r_v"], caso["z_v"], 0.0),
            "E1": downwash_E(th_deg, r_c, N_VATISTAS, PSI_MAX_E1, False),
            "E2": downwash_E(th_deg, r_c, N_VATISTAS, PSI_MAX_E2, True),
        }
        for nome, ww in w.items():
            try:
                lam, Cl, F, g, _ = bemt_downwash_vortice(x, th, ww, K_VORTICE)
            except ValueError as err:
                # only expected for E1: no physical solution, recorded as such
                assert nome == "E1", f"{nome}: {err}"
                print(f"  E1 full helix: NO SOLUTION ({err})")
                continue
            ris[nome] = (*risultato(lam, Cl, F), Cl)
            lam_v[nome] = K_VORTICE * g / (OMEGA * R**2) * ww

        etichette = {"P": "Prandtl only", "V1": "V1 2D vortex", "E1": "E1 full helix",
                     "E2": "E2 helix - azimuthal mean"}
        for nome, et in etichette.items():
            if nome in ris:
                stampa(et, ris[nome][0], ris[nome][1], caso)

        print("  -- declared checks --")
        w_alt = downwash_E(th_deg, R_C_VECCHIO, N_VATISTAS, PSI_MAX_E2, True)
        lam, Cl, F, _, _ = bemt_downwash_vortice(x, th, w_alt, K_VORTICE)
        stampa(f"E2, r_c = {R_C_VECCHIO}R", *risultato(lam, Cl, F), caso)
        w_alt = downwash_E(th_deg, r_c, 1, PSI_MAX_E2, True)
        lam, Cl, F, _, _ = bemt_downwash_vortice(x, th, w_alt, K_VORTICE)
        stampa("E2, Scully core (n = 1)", *risultato(lam, Cl, F), caso)
        lam, Cl, F, _, _ = bemt_vortice(x, th, caso["r_v"], caso["z_v"], togli_media=True)
        stampa("V1 minus disk mean (Test B)", *risultato(lam, Cl, F), caso)
        if th_deg == 5:
            lam, Cl, F, _, _ = bemt_downwash_vortice(x, th, w["E2"], A_MISURATO_5)
            stampa(f"E2, A = {A_MISURATO_5} (measured)", *risultato(lam, Cl, F), caso)
        tutti[th_deg] = (ris, lam_v)

    # figures (shared style: same color for the same model as in the v1 figures)
    applica_stile()
    os.makedirs("figure", exist_ok=True)
    chiave = {"P": "prandtl", "V1": "v1", "E2": "e2"}     # E1 has no solution: not plotted
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for j, th_deg in enumerate(CASI):
        ris, lam_v = tutti[th_deg]
        for nome, k_stile in chiave.items():
            ax[0, j].plot(x, ris[nome][2], **stile_modello(k_stile))
            if nome in lam_v:
                ax[1, j].plot(x, lam_v[nome], **stile_modello(k_stile))
        ax[0, j].plot(X_STAZIONI, CASI[th_deg]["Cl"], **stile_esperimento())
        ax[0, j].set_title(f"Cl, θ = {th_deg}°")
        ax[1, j].set_title(f"Vortex downwash λ_v, θ = {th_deg}°")
        ax[1, j].axhline(0, color=GRIGIO_ASSI, lw=0.8)
        for a in ax[:, j]:
            a.set_xlabel("r/R")
    ax[0, 0].set_ylabel("Cl"); ax[1, 0].set_ylabel("λ_v (positive down)")
    ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figure/confronto_v2.png", dpi=200)

    # thrust coefficient versus pitch: the three converged models and the experiment
    fig, ax = plt.subplots(figsize=(6, 4.2))
    th_lista = list(CASI)
    for nome, k_stile in chiave.items():
        ax.plot(th_lista, [tutti[t][0][nome][0] for t in th_lista], marker="o",
                **stile_modello(k_stile))
    ax.plot(th_lista, [CASI[t]["CT"] for t in th_lista],
            **{**stile_esperimento(), "marker": "s"})
    ax.set_xlabel("θ [°]"); ax.set_ylabel("C_T")
    ax.set_title("Thrust coefficient: v1 and v2 models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figure/ct_theta_v2.png", dpi=200)
    print("\nFigures saved: figure/confronto_v2.png, figure/ct_theta_v2.png")
