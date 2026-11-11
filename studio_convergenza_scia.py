"""
Step 2: convergence study of the downwash induced on the reference blade by the
preceding blade's helical tip vortex (8 deg).

1. Angular step dpsi (wake length fixed).
2. Wake length psi_max (dpsi fixed), for
     a) the full helix,
     b) the helix minus its azimuthal average (the part due to the vortex being discrete;
        the azimuthal average is what an axisymmetric actuator-disk wake already contains).
3. Size of the full-helix downwash compared with the BEMT inflow of v1 (double counting).

Prints tables and saves figure/convergenza_scia.png.
Run:  python studio_convergenza_scia.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from dati_sperimentali import SCIA, R1_NUCLEO, CASI
from geometria_scia import nodi_elica, stazioni_pala
from vortice_3d import velocita_segmenti
from modello import R, OMEGA, griglia_semicoseno, bemt_vortice
from stile_figure import (applica_stile, stile_modello, RAMPA_GRIGI, INCHIOSTRO,
                          GRIGIO_ASSI, GRIGLIA)

THETA = 8
N_PALE = 2
PSI_LINEARE = 180.0
N_VATISTAS = 2
N_AZIMUT_MEDIA = 72          # azimuths used for the azimuthal average (checked below)

x = griglia_semicoseno(60)


def downwash(punti, dpsi, psi_max):
    """Downwash (positive down) per unit Gamma/(Omega R^2) at the given points."""
    nodi, _ = nodi_elica(SCIA[THETA], N_PALE, dpsi, psi_max, PSI_LINEARE)
    S = len(nodi) - 1
    V = velocita_segmenti(punti, nodi[:-1], nodi[1:], np.ones(S),
                          np.full(S, R1_NUCLEO[THETA]), N_VATISTAS)
    return -V[:, 2]


def downwash_medio_azimutale(dpsi, psi_max, n_azimut):
    """Average over the azimuth of the downwash at radius x: equivalent to smearing the
    helix into an axisymmetric vortex sheet with the same contraction and descent."""
    somma = np.zeros_like(x)
    for phi in np.linspace(0.0, 2 * np.pi, n_azimut, endpoint=False):
        punti = np.column_stack([x * np.cos(phi), x * np.sin(phi), np.zeros_like(x)])
        somma += downwash(punti, dpsi, psi_max)
    return somma / n_azimut


if __name__ == "__main__":
    P = stazioni_pala(x)

    # 1. angular step
    print(f"=== 1. Angular step (theta = {THETA} deg, psi_max = 1440 deg) ===")
    rif = downwash(P, 0.3125, 1440.0)
    passi = np.array([20, 10, 5, 2.5, 1.25, 0.625])
    err_passo = np.array([np.max(np.abs(downwash(P, d, 1440.0) - rif)) / np.max(np.abs(rif))
                          for d in passi])
    for d, e in zip(passi, err_passo):
        print(f"dpsi = {d:6.3f} deg   max relative error (ref. 0.3125 deg) = {e:.2e}")

    # 2. wake length
    print("\n=== 2. Wake length (dpsi = 1.25 deg) ===")
    lunghezze = np.array([360, 720, 1440, 2880, 5760])
    w_pieno = {L: downwash(P, 1.25, float(L)) for L in lunghezze}
    w_rif = downwash(P, 1.25, 14400.0)
    d_pert = {L: w_pieno[L] - downwash_medio_azimutale(1.25, float(L), N_AZIMUT_MEDIA)
              for L in lunghezze}
    print("psi_max   full helix: max|w - w(14400)|    helix - average: max|d - d(5760)|")
    for L in lunghezze:
        e_p = np.max(np.abs(w_pieno[L] - w_rif))
        e_d = np.max(np.abs(d_pert[L] - d_pert[lunghezze[-1]]))
        print(f"{L:7d}   {e_p:10.2e}                        {e_d:10.2e}")
    # check of the azimuthal resolution of the average
    d144 = w_pieno[1440] - downwash_medio_azimutale(1.25, 1440.0, 2 * N_AZIMUT_MEDIA)
    print(f"azimuthal average: max change 72 -> 144 azimuths = "
          f"{np.max(np.abs(d144 - d_pert[1440])):.1e}")

    # 3. size of the full-helix downwash against the BEMT inflow of v1
    print("\n=== 3. Full helix against the BEMT inflow (v1, 8 deg) ===")
    xg = griglia_semicoseno(200)
    lam, _, _, gamma_max, _ = bemt_vortice(xg, np.radians(THETA), CASI[THETA]["r_v"],
                                           CASI[THETA]["z_v"])
    g = gamma_max / (OMEGA * R**2)            # Gamma_max / (Omega R^2)
    print(f"Gamma_max/(Omega R^2) = {g:.4f}")
    for xs in (0.5, 0.7, 0.85):
        print(f"x = {xs:.2f}: lambda BEMT = {np.interp(xs, xg, lam):.4f}   "
              f"full helix = {g * np.interp(xs, x, w_rif):+.4f}   "
              f"helix - average = {g * np.interp(xs, x, d_pert[5760]):+.4f}")

    # figure (shared style; wake lengths are an ordered parameter -> gray ramp)
    applica_stile()
    os.makedirs("figure", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].loglog(passi, err_passo, "o-", color=INCHIOSTRO, label="computed")
    ax[0].loglog(passi, err_passo[2] * (passi / passi[2])**2, "--", color=GRIGIO_ASSI, lw=0.8,
                 label="slope 2")
    ax[0].set_xlabel("dpsi [deg]"); ax[0].set_ylabel("max relative error")
    ax[0].set_title("Angular step"); ax[0].legend()
    ax[0].grid(True, which="minor", color=GRIGLIA, lw=0.4)
    for L, col in zip(lunghezze, RAMPA_GRIGI):
        ax[1].plot(x, g * w_pieno[L], color=col, label=f"psi_max = {L}°")
        ax[2].plot(x, g * d_pert[L], color=col, label=f"psi_max = {L}°")
    ax[1].plot(xg, lam, **{**stile_modello("v1"), "ls": "--",
                           "label": "λ of the BEMT, V1 model"})
    ax[1].set_title("Full helix, 8°"); ax[2].set_title("Helix − azimuthal mean, 8°")
    for a in ax[1:]:
        a.set_xlabel("r/R"); a.set_ylabel("downwash / (ΩR)"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figure/convergenza_scia.png", dpi=200)
    print("\nFigure saved: figure/convergenza_scia.png")
