"""
BEMT model for the Caradonna & Tung rotor in hover.

Contains only the rotor parameters and the model functions:
no printing and no plotting. The analyses are in main.py.

Conventions:
    x      = r/R, non-dimensional radial position along the blade
    lambda = v/(Omega*R), non-dimensional induced velocity (positive downwards)
    theta  = collective pitch angle [rad]
"""
import numpy as np

# ---------------------------------------------------------------
# Rotor parameters (Caradonna & Tung, NASA TM-81232)
# ---------------------------------------------------------------
R = 1.143          # blade radius [m]
CORDA = 0.1905     # blade chord [m]
N_PALE = 2         # number of blades
RPM = 1250         # rotational speed [rpm]
A_SUONO = 340.0    # speed of sound [m/s]
X_RADICE = 0.2     # start of the lifting part of the blade (r/R)
A0 = 5.7           # lift-curve slope [1/rad]

OMEGA = RPM * 2 * np.pi / 60              # angular velocity [rad/s]
SIGMA = N_PALE * CORDA / (np.pi * R)      # rotor solidity

# Fixed-point iteration parameters
TOL = 1e-8
MAX_IT = 100


# ---------------------------------------------------------------
# Radial grids
# ---------------------------------------------------------------
def griglia_uniforme(N, x_radice=X_RADICE):
    """N equally spaced stations between the root and the tip."""
    return np.linspace(x_radice, 1.0, N)


def griglia_semicoseno(N, x_radice=X_RADICE):
    """N half-cosine stations, clustered towards the tip, where the loading
    drops to zero as sqrt(1-x) because of the Prandtl tip-loss factor."""
    beta = np.linspace(0, np.pi / 2, N)
    return x_radice + (1.0 - x_radice) * np.sin(beta)


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def coefficiente_spinta(x, lam, F=1.0):
    """Thrust coefficient from the momentum balance: dC_T = 4 F lambda^2 x dx."""
    return np.trapezoid(4 * F * lam**2 * x, x)


def errore_pct(modello, esperimento):
    """Percentage error (positive = the model overpredicts)."""
    return (modello - esperimento) / esperimento * 100


def _radice_stabile(A, B, C):
    """Positive root of A*lam^2 + B*lam + C = 0, written as -2C/(B + sqrt(B^2 - 4AC)).
    Algebraically equivalent to the standard formula, but still defined for A = 0
    (at the tip, where the Prandtl factor F vanishes)."""
    return -2 * C / (B + np.sqrt(B**2 - 4 * A * C))


def _fattore_prandtl(x, lam):
    """Prandtl tip-loss factor (hover, phi ~ lambda/x)."""
    f = (N_PALE / 2) * (1 - x) / lam
    return (2 / np.pi) * np.arccos(np.exp(-f))


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
def inflow_uniforme(theta, a=A0):
    """Blade element theory + actuator disk with uniform inflow.
    C_T = (sigma*a/2)(theta/3 - lambda/2) and C_T = 2 lambda^2.
    Note: this formula integrates from x = 0 to x = 1, without root cut-out or Prandtl."""
    lam = _radice_stabile(2.0, SIGMA * a / 4, -SIGMA * a * theta / 6)
    return lam, 2 * lam**2


def bemt_senza_perdite(x, theta, a=A0):
    """Annular BEMT without tip loss (closed-form solution):
    lambda^2 + (sigma*a/8) lambda - (sigma*a/8) theta x = 0."""
    return _radice_stabile(1.0, SIGMA * a / 8, -SIGMA * a / 8 * theta * x)


def bemt_prandtl(x, theta, a=A0, tol=TOL, max_it=MAX_IT):
    """BEMT with Prandtl tip loss, solved by fixed-point iteration:
    F lambda^2 + (sigma*a/8) lambda - (sigma*a/8) theta x = 0.
    Annuli are independent, so the model can also be evaluated at isolated points.
    Returns lambda, F and the number of iterations."""
    F = np.ones_like(x)
    lam_old = np.zeros_like(x)
    for it in range(1, max_it + 1):
        lam = _radice_stabile(F, SIGMA * a / 8, -SIGMA * a / 8 * theta * x)
        F = _fattore_prandtl(x, lam)
        if np.max(np.abs(lam - lam_old)) < tol:
            return lam, F, it
        lam_old = lam
    print("WARNING: bemt_prandtl did not converge")
    return lam, F, it


def bemt_vortice(x, theta, r_v, z_v, k=1.0, r_c=0.0, usa_prandtl=True,
                 togli_media=False, comprimibile=False, tol=TOL, max_it=MAX_IT):
    """BEMT with Prandtl tip loss and a correction for the tip vortex of the preceding blade.

    The vortex (age 180 deg) is modelled as a straight 2D line vortex in the (r, z) plane
    of the blade, located at (r_v, z_v) below the disk, with strength k*Gamma_max.
    Its vertical induced velocity (Biot-Savart) enters the blade-element side only:
        F lambda^2 + (sigma*a/8) lambda - (sigma*a/8)(theta x - lambda_v) = 0
        lambda_v = -k Gamma_max (x - r_v) / (2 pi (d^2 + r_c^2) Omega R^2)
    Gamma_max depends on the solution, so everything is solved by fixed-point iteration.

    WARNING: Gamma_max is the maximum over the whole blade, so the function must be
    called on the full grid (use np.interp to evaluate at the experimental stations).

    Options (all off by default):
        k            vortex strength / maximum bound circulation of the blade
        r_c          non-dimensional viscous core radius (Scully model)
        usa_prandtl  False for the test without the Prandtl factor
        togli_media  True to remove the disk-averaged value of lambda_v
        comprimibile True for the Prandtl-Glauert correction of the lift-curve slope

    Returns lambda, Cl, F, Gamma_max [m^2/s] and the number of iterations."""
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

    print("WARNING: bemt_vortice did not converge")
    return lam, Cl, F, gamma_max, it


def cl_prandtl(x, theta, lam, a=A0):
    """Sectional Cl of the models without vortex: Cl = a (theta - lambda/x)."""
    return a * (theta - lam / x)
