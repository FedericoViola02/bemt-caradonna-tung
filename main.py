"""
BEMT per il rotore di Caradonna & Tung in hover: analisi complete.

Ordine delle analisi:
    1. Modelli di base a 8° (inflow uniforme, BEMT senza perdite, BEMT con Prandtl)
    2. BEMT con Prandtl contro l'esperimento a 8° (distribuzione di Cl)
    3. BEMT con Prandtl a più angoli di passo
    4. Correzione con il vortice della pala precedente a 8°
    5. Studio di convergenza della griglia (uniforme contro semi-coseno)
    6. Sensibilità alla posizione del vortice (+-0.025R)
    7. Test di sensibilità: senza Prandtl, senza downwash medio, comprimibilità, nucleo viscoso
    8. Validazione predittiva a 5°, 8° e 12°
    9. Figure (salvate nella cartella "figure")
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from modello import (R, OMEGA, A_SUONO, griglia_uniforme, griglia_semicoseno,
                     coefficiente_spinta, errore_pct, inflow_uniforme,
                     bemt_senza_perdite, bemt_prandtl, bemt_vortice, cl_prandtl)
from dati_sperimentali import X_STAZIONI, CASI, K_VORTICE_5

# Griglia principale: semi-coseno con 200 stazioni (scelta dallo studio della sezione 5)
x = griglia_semicoseno(200)

# Caso base: 8°
caso8 = CASI[8]
theta8 = np.radians(8)


def stampa_risultato(etichetta, CT, CT_exp, Cl_stazioni=None, Cl_exp=None):
    """Stampa C_T con il suo errore e, se dati, gli errori sul Cl nelle stazioni."""
    print(f"{etichetta}: C_T = {CT:.5f}  (errore {errore_pct(CT, CT_exp):+.1f}%)")
    if Cl_stazioni is not None:
        err = errore_pct(Cl_stazioni, Cl_exp)
        print(f"    errore Cl nelle stazioni [%]: {np.round(err, 1)}"
              f"   errore medio assoluto: {np.mean(np.abs(err)):.1f}%")


# %% 1. Modelli di base a 8°
print("\n=== 1. Modelli di base a 8° ===")
lam_unif, CT_unif = inflow_uniforme(theta8)
print(f"Inflow uniforme: lambda = {lam_unif:.4f}")
stampa_risultato("Inflow uniforme", CT_unif, caso8["CT"])

lam_sp = bemt_senza_perdite(x, theta8)
stampa_risultato("BEMT senza perdite", coefficiente_spinta(x, lam_sp), caso8["CT"])

lam_pr, F_pr, it_pr = bemt_prandtl(x, theta8)
CT_pr = coefficiente_spinta(x, lam_pr, F_pr)
print(f"BEMT con Prandtl: convergenza in {it_pr} iterazioni")
stampa_risultato("BEMT con Prandtl", CT_pr, caso8["CT"])


# %% 2. BEMT con Prandtl contro l'esperimento a 8°
# Senza vortice gli anelli sono indipendenti: il modello si valuta direttamente
# nelle stazioni sperimentali, senza interpolazione.
print("\n=== 2. BEMT con Prandtl contro l'esperimento a 8° ===")
lam_st, F_st, _ = bemt_prandtl(X_STAZIONI, theta8)
err_Cl_pr = errore_pct(cl_prandtl(X_STAZIONI, theta8, lam_st), caso8["Cl"])
print(f"Errore Cl nelle stazioni [%]: {np.round(err_Cl_pr, 1)}")


# %% 3. BEMT con Prandtl a più angoli di passo
print("\n=== 3. BEMT con Prandtl a più angoli ===")
for th_deg, caso in CASI.items():
    lam_th, F_th, it_th = bemt_prandtl(x, np.radians(th_deg))
    stampa_risultato(f"{th_deg:>2}°", coefficiente_spinta(x, lam_th, F_th), caso["CT"])

# Curva continua C_T(theta) del modello, per il grafico
theta_curva = np.arange(1, 14.01, 0.5)
CT_curva = np.array([coefficiente_spinta(x, *bemt_prandtl(x, np.radians(th))[:2])
                     for th in theta_curva])


# %% 4. Correzione con il vortice della pala precedente a 8°
print("\n=== 4. BEMT con vortice a 8° ===")
lam_v, Cl_v, F_v, gamma_v, it_v = bemt_vortice(x, theta8, caso8["r_v"], caso8["z_v"])
CT_v = coefficiente_spinta(x, lam_v, F_v)
print(f"Convergenza in {it_v} iterazioni, Gamma_max = {gamma_v:.3f} m^2/s")
stampa_risultato("BEMT con vortice", CT_v, caso8["CT"],
                 np.interp(X_STAZIONI, x, Cl_v), caso8["Cl"])


# %% 5. Studio di convergenza della griglia
# Il riferimento è la griglia più precisa disponibile (semi-coseno, 4000 punti).
# Vicino alla punta il carico va come sqrt(1-x): con griglia uniforme la regola dei
# trapezi converge con ordine circa 1.5, con la semi-coseno molto più velocemente.
print("\n=== 5. Studio di convergenza della griglia (8°, con vortice) ===")


def CT_vortice_8(xg):
    lam, _, F, _, _ = bemt_vortice(xg, theta8, caso8["r_v"], caso8["z_v"])
    return coefficiente_spinta(xg, lam, F)


CT_rif = CT_vortice_8(griglia_semicoseno(4000))
lista_N = np.array([50, 100, 200, 400, 800, 1000])
err_uni = np.array([abs(CT_vortice_8(griglia_uniforme(N)) - CT_rif) / CT_rif for N in lista_N])
err_cos = np.array([abs(CT_vortice_8(griglia_semicoseno(N)) - CT_rif) / CT_rif for N in lista_N])
print(f"C_T di riferimento (semi-coseno, N = 4000) = {CT_rif:.7f}")
for N, eu, ec in zip(lista_N, err_uni, err_cos):
    print(f"N = {N:>4}   errore relativo uniforme = {eu:.1e}   semi-coseno = {ec:.1e}")


# %% 6. Sensibilità alla posizione del vortice (+-0.025R, come nel report)
# Nota: scegliere la combinazione migliore sarebbe una taratura, non una validazione.
print("\n=== 6. Sensibilità alla posizione del vortice (8°) ===")
for r_v in [0.880, 0.905, 0.930]:
    for z_v in [0.065, 0.090, 0.115]:
        lam_s, Cl_s, F_s, _, _ = bemt_vortice(x, theta8, r_v, z_v)
        stampa_risultato(f"r_v = {r_v:.3f}, z_v = {z_v:.3f}", coefficiente_spinta(x, lam_s, F_s),
                         caso8["CT"], np.interp(X_STAZIONI, x, Cl_s), caso8["Cl"])


# %% 7. Test di sensibilità sul modello del vortice
print("\n=== 7. Test di sensibilità ===")

# Test A: vortice senza Prandtl. Il carico alla punta non va più a zero e cresce molto:
# il fattore di Prandtl serve anche con il vortice (non dimostra però che i due
# effetti siano indipendenti: entrambi descrivono in parte la scia d'estremità).
lam_A, Cl_A, F_A, _, _ = bemt_vortice(x, theta8, caso8["r_v"], caso8["z_v"], usa_prandtl=False)
stampa_risultato("Test A, senza Prandtl (8°)", coefficiente_spinta(x, lam_A, F_A), caso8["CT"],
                 np.interp(X_STAZIONI, x, Cl_A), caso8["Cl"])

# Test B: vortice senza il suo downwash medio sul disco (doppio conteggio con il disco attuatore).
lam_B, Cl_B, F_B, _, _ = bemt_vortice(x, theta8, caso8["r_v"], caso8["z_v"], togli_media=True)
stampa_risultato("Test B, senza downwash medio (8°)", coefficiente_spinta(x, lam_B, F_B), caso8["CT"],
                 np.interp(X_STAZIONI, x, Cl_B), caso8["Cl"])

# Comprimibilità: Prandtl-Glauert sulla pendenza della retta di portanza.
lam_co, Cl_co, F_co, _, _ = bemt_vortice(x, theta8, caso8["r_v"], caso8["z_v"], comprimibile=True)
stampa_risultato("Prandtl-Glauert (8°)", coefficiente_spinta(x, lam_co, F_co), caso8["CT"],
                 np.interp(X_STAZIONI, x, Cl_co), caso8["Cl"])

# Nucleo viscoso (Scully) a 5°, dove il vortice è più vicino alla pala.
caso5 = CASI[5]
for r_c in [0.0, 0.01, 0.02]:
    lam_rc, Cl_rc, F_rc, _, _ = bemt_vortice(x, np.radians(5), caso5["r_v"], caso5["z_v"], r_c=r_c)
    stampa_risultato(f"Nucleo r_c = {r_c:.2f} (5°)", coefficiente_spinta(x, lam_rc, F_rc), caso5["CT"],
                     np.interp(X_STAZIONI, x, Cl_rc), caso5["Cl"])


# %% 8. Validazione predittiva a 5°, 8° e 12°
# Posizioni del vortice lette dal report e mai ritoccate; intensità k = 1 per tutti gli angoli.
print("\n=== 8. Validazione predittiva ===")
risultati = {}
for th_deg, caso in CASI.items():
    th = np.radians(th_deg)
    lam_p, F_p, _ = bemt_prandtl(x, th)
    lam_c, Cl_c, F_c, _, _ = bemt_vortice(x, th, caso["r_v"], caso["z_v"])
    risultati[th_deg] = {"Cl_pr": cl_prandtl(x, th, lam_p), "Cl_v": Cl_c,
                         "CT_v": coefficiente_spinta(x, lam_c, F_c)}

    print(f"--- {th_deg}° ---")
    stampa_risultato("  Prandtl", coefficiente_spinta(x, lam_p, F_p), caso["CT"],
                     np.interp(X_STAZIONI, x, risultati[th_deg]["Cl_pr"]), caso["Cl"])
    stampa_risultato("  Vortice", risultati[th_deg]["CT_v"], caso["CT"],
                     np.interp(X_STAZIONI, x, Cl_c), caso["Cl"])

    if th_deg == 5:
        lam_k, _, F_k, _, _ = bemt_vortice(x, th, caso["r_v"], caso["z_v"], k=K_VORTICE_5)
        stampa_risultato(f"  Vortice, k = {K_VORTICE_5}", coefficiente_spinta(x, lam_k, F_k), caso["CT"])


# %% 9. Figure
os.makedirs("figure", exist_ok=True)

# Mach locale lungo la pala
plt.figure()
plt.plot(x, OMEGA * x * R / A_SUONO)
plt.xlabel("r/R")
plt.ylabel("Mach locale")
plt.title("Mach lungo la pala")
plt.grid()
plt.tight_layout()
plt.savefig("figure/mach.png", dpi=200)

# Carico lungo la pala a 8°
plt.figure()
plt.plot(x, 4 * lam_sp**2 * x, label="BEMT senza perdite")
plt.plot(x, 4 * F_pr * lam_pr**2 * x, label="BEMT con Prandtl")
plt.plot(x, 4 * F_v * lam_v**2 * x, label="BEMT con vortice")
plt.xlabel("r/R")
plt.ylabel("dC_T/dx")
plt.title("Carico lungo la pala, θ = 8°")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figure/carico_8deg.png", dpi=200)

# Cl lungo la pala a 8°
plt.figure()
plt.plot(x, cl_prandtl(x, theta8, lam_sp), label="BEMT senza perdite")
plt.plot(x, cl_prandtl(x, theta8, lam_pr), label="BEMT con Prandtl")
plt.plot(x, Cl_v, label="BEMT con vortice")
plt.plot(X_STAZIONI, caso8["Cl"], "o", label="Esperimento")
plt.xlabel("r/R")
plt.ylabel("Cl")
plt.title("Coefficiente di portanza lungo la pala, θ = 8°")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figure/cl_8deg.png", dpi=200)

# C_T al variare dell'angolo di passo
plt.figure()
th_lista = list(CASI.keys())
plt.plot(theta_curva, CT_curva, label="BEMT con Prandtl")
plt.plot(th_lista, [risultati[t]["CT_v"] for t in th_lista], "s", label="BEMT con vortice")
plt.plot(th_lista, [CASI[t]["CT"] for t in th_lista], "x", ms=8, label="Esperimento")
plt.xlabel("θ [°]")
plt.ylabel("C_T")
plt.title("C_T al variare dell'angolo di passo")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figure/ct_theta.png", dpi=200)

# Studio di convergenza
plt.figure()
plt.loglog(lista_N, err_uni, "o-", label="Griglia uniforme")
plt.loglog(lista_N, err_cos, "s-", label="Griglia semi-coseno")
plt.loglog(lista_N, err_uni[0] * (lista_N / lista_N[0])**-1.5, "k--", lw=0.8, label="pendenza -1.5")
plt.loglog(lista_N, err_uni[0] * (lista_N / lista_N[0])**-2, "k:", lw=0.8, label="pendenza -2")
plt.xlabel("Numero di stazioni N")
plt.ylabel("Errore relativo su C_T")
plt.title("Convergenza dell'integrale di C_T")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("figure/convergenza.png", dpi=200)

# Validazione: distribuzione di Cl ai tre angoli
fig, assi = plt.subplots(1, 3, figsize=(13, 4))
for ax, th_deg in zip(assi, CASI):
    ax.plot(x, risultati[th_deg]["Cl_pr"], label="BEMT con Prandtl")
    ax.plot(x, risultati[th_deg]["Cl_v"], label="BEMT con vortice")
    ax.plot(X_STAZIONI, CASI[th_deg]["Cl"], "o", label="Esperimento")
    ax.set_title(f"θ = {th_deg}°")
    ax.set_xlabel("r/R")
    ax.grid()
assi[0].set_ylabel("Cl")
assi[0].legend()
fig.tight_layout()
fig.savefig("figure/validazione_cl.png", dpi=200)

plt.show()
