"""
Modello BEMT per il rotore di Caradonna & Tung in hover.

Contiene solo i parametri del rotore e le funzioni del modello:
nessuna stampa e nessun grafico. Le analisi sono in main.py.

Convenzioni:
    x      = r/R, posizione adimensionale lungo la pala
    lambda = v/(Omega*R), velocità indotta adimensionale (positiva verso il basso)
    theta  = angolo di passo collettivo [rad]
"""
import numpy as np

# ---------------------------------------------------------------
# Parametri del rotore (Caradonna & Tung, NASA TM-81232)
# ---------------------------------------------------------------
R = 1.143          # raggio della pala [m]
CORDA = 0.1905     # corda della pala [m]
N_PALE = 2         # numero di pale
RPM = 1250         # velocità di rotazione [giri/min]
A_SUONO = 340.0    # velocità del suono [m/s]
X_RADICE = 0.2     # inizio della parte portante della pala (r/R)
A0 = 5.7           # pendenza della retta di portanza [1/rad]

OMEGA = RPM * 2 * np.pi / 60              # velocità angolare [rad/s]
SIGMA = N_PALE * CORDA / (np.pi * R)      # solidità del rotore

# Parametri dell'iterazione di punto fisso
TOL = 1e-8
MAX_IT = 100


# ---------------------------------------------------------------
# Griglie radiali
# ---------------------------------------------------------------
def griglia_uniforme(N, x_radice=X_RADICE):
    """N stazioni equispaziate tra la radice e la punta."""
    return np.linspace(x_radice, 1.0, N)


def griglia_semicoseno(N, x_radice=X_RADICE):
    """N stazioni a semi-coseno: più fitte verso la punta, dove il carico
    scende a zero come sqrt(1-x) per effetto del fattore di Prandtl."""
    beta = np.linspace(0, np.pi / 2, N)
    return x_radice + (1.0 - x_radice) * np.sin(beta)


# ---------------------------------------------------------------
# Funzioni di servizio
# ---------------------------------------------------------------
def coefficiente_spinta(x, lam, F=1.0):
    """C_T integrando il bilancio della quantità di moto: dC_T = 4 F lambda^2 x dx."""
    return np.trapezoid(4 * F * lam**2 * x, x)


def errore_pct(modello, esperimento):
    """Errore percentuale (positivo = il modello sovrastima)."""
    return (modello - esperimento) / esperimento * 100


def _radice_stabile(A, B, C):
    """Radice positiva di A*lam^2 + B*lam + C = 0 nella forma -2C/(B + sqrt(B^2 - 4AC)).
    È equivalente alla formula classica, ma resta definita anche per A = 0
    (alla punta, dove il fattore di Prandtl F vale zero)."""
    return -2 * C / (B + np.sqrt(B**2 - 4 * A * C))


def _fattore_prandtl(x, lam):
    """Fattore di perdita d'estremità di Prandtl (hover, phi ~ lambda/x)."""
    f = (N_PALE / 2) * (1 - x) / lam
    return (2 / np.pi) * np.arccos(np.exp(-f))


# ---------------------------------------------------------------
# Modelli
# ---------------------------------------------------------------
def inflow_uniforme(theta, a=A0):
    """Teoria dell'elemento di pala + disco attuatore con inflow uniforme.
    C_T = (sigma*a/2)(theta/3 - lambda/2) e C_T = 2 lambda^2.
    Nota: questa formula integra da x = 0 a x = 1, senza radice e senza Prandtl."""
    lam = _radice_stabile(2.0, SIGMA * a / 4, -SIGMA * a * theta / 6)
    return lam, 2 * lam**2


def bemt_senza_perdite(x, theta, a=A0):
    """BEMT per anelli senza perdite d'estremità (soluzione in forma chiusa):
    lambda^2 + (sigma*a/8) lambda - (sigma*a/8) theta x = 0."""
    return _radice_stabile(1.0, SIGMA * a / 8, -SIGMA * a / 8 * theta * x)


def bemt_prandtl(x, theta, a=A0, tol=TOL, max_it=MAX_IT):
    """BEMT con perdite d'estremità di Prandtl, risolto per punto fisso:
    F lambda^2 + (sigma*a/8) lambda - (sigma*a/8) theta x = 0.
    Gli anelli sono indipendenti: si può valutare anche su punti isolati.
    Restituisce lambda, F e il numero di iterazioni."""
    F = np.ones_like(x)
    lam_old = np.zeros_like(x)
    for it in range(1, max_it + 1):
        lam = _radice_stabile(F, SIGMA * a / 8, -SIGMA * a / 8 * theta * x)
        F = _fattore_prandtl(x, lam)
        if np.max(np.abs(lam - lam_old)) < tol:
            return lam, F, it
        lam_old = lam
    print("ATTENZIONE: bemt_prandtl non ha raggiunto la convergenza")
    return lam, F, it


def bemt_vortice(x, theta, r_v, z_v, k=1.0, r_c=0.0, usa_prandtl=True,
                 togli_media=False, comprimibile=False, tol=TOL, max_it=MAX_IT):
    """BEMT con Prandtl e correzione per il vortice d'estremità della pala precedente.

    Il vortice (età 180°) è modellato come vortice rettilineo 2D nel piano (r, z)
    della pala, posto in (r_v, z_v) sotto il disco, con intensità k*Gamma_max.
    La componente verticale indotta (Biot-Savart) entra solo nella parte delle pale:
        F lambda^2 + (sigma*a/8) lambda - (sigma*a/8)(theta x - lambda_v) = 0
        lambda_v = -k Gamma_max (x - r_v) / (2 pi (d^2 + r_c^2) Omega R^2)
    Gamma_max dipende dalla soluzione, quindi tutto è risolto per punto fisso.

    ATTENZIONE: Gamma_max è il massimo su tutta la pala, quindi la funzione va
    chiamata sulla griglia completa (per i punti sperimentali usare np.interp).

    Opzioni (tutte disattivate di default):
        k            intensità del vortice / circolazione massima della pala
        r_c          raggio del nucleo viscoso adimensionale (modello di Scully)
        usa_prandtl  False per il test senza fattore di Prandtl
        togli_media  True per togliere a lambda_v il suo valore medio sul disco
        comprimibile True per la correzione di Prandtl-Glauert sulla pendenza

    Restituisce lambda, Cl, F, Gamma_max [m^2/s] e il numero di iterazioni."""
    a = A0
    if comprimibile:
        M_loc = OMEGA * x * R / A_SUONO
        a = A0 / np.sqrt(1 - M_loc**2)

    d2 = (x - r_v)**2 + z_v**2 + r_c**2
    F = np.ones_like(x)
    lam_old = np.zeros_like(x)
    gamma_max = 0.0

    for it in range(1, max_it + 1):
        lam_v = -k * gamma_max * (x - r_v) / (2 * np.pi * d2 * OMEGA * R**2)
        if togli_media:
            lam_v = lam_v - np.trapezoid(lam_v * x, x) / np.trapezoid(x, x)

        lam = _radice_stabile(F, SIGMA * a / 8, -SIGMA * a / 8 * (theta * x - lam_v))
        if usa_prandtl:
            F = _fattore_prandtl(x, lam)

        Cl = a * (theta - (lam + lam_v) / x)
        gamma_max = np.max(0.5 * OMEGA * x * R * CORDA * Cl)   # Kutta-Joukowski

        if np.max(np.abs(lam - lam_old)) < tol:
            return lam, Cl, F, gamma_max, it
        lam_old = lam

    print("ATTENZIONE: bemt_vortice non ha raggiunto la convergenza")
    return lam, Cl, F, gamma_max, it


def cl_prandtl(x, theta, lam, a=A0):
    """Cl di sezione dei modelli senza vortice: Cl = a (theta - lambda/x)."""
    return a * (theta - lam / x)
