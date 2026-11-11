"""
BEMT with the 3D helical tip vortex of the preceding blade (v2.0, phase 1, step 3).

The validated v1 code (modello.py) is not modified: this module reuses its constants and
helper functions, so the blade-element / momentum / Prandtl part is literally the same code.

Idea: the wake geometry is prescribed (measured), so the vortex-induced downwash is
    lambda_v(x) = k * Gamma_max / (Omega R^2) * w(x)
where w(x) is the downwash per unit non-dimensional circulation, computed once.
Only Gamma_max changes during the fixed-point iteration. One solver serves every model:
    - 2D straight vortex of v1          -> downwash_unitario_2d
    - full helix                        -> downwash_unitario_elica
    - helix minus its azimuthal average -> downwash_unitario_perturbazione
The last one keeps only the effect of the vortex being concentrated in a line instead of
smeared over the azimuth: the axisymmetric part of the wake is already in the momentum
balance of the actuator disk, so adding it again would count it twice.

Conventions as in modello.py: x = r/R, lambda positive downwards, theta in rad.
"""
import numpy as np

from modello import (R, CORDA, OMEGA, SIGMA, A0, TOL, MAX_IT,
                     _radice_stabile, _fattore_prandtl)
from geometria_scia import nodi_elica, stazioni_pala
from vortice_3d import velocita_segmenti


# ---------------------------------------------------------------
# Downwash per unit circulation, w(x): [Omega R] per [Omega R^2] / R
# ---------------------------------------------------------------
def downwash_unitario_2d(x, r_v, z_v, r_c):
    """2D straight vortex of v1 (Scully core of radius r_c), per unit Gamma/(Omega R^2):
    exactly the expression inside modello.bemt_vortice."""
    d2 = (x - r_v)**2 + z_v**2 + r_c**2
    return -(x - r_v) / (2 * np.pi * d2)


def _downwash_punti(punti, nodi, r_c, n_vatistas):
    """Downwash (positive down) at arbitrary points, per unit Gamma, from a chain of nodes."""
    S = len(nodi) - 1
    V = velocita_segmenti(punti, nodi[:-1], nodi[1:], np.ones(S), np.full(S, r_c), n_vatistas)
    return -V[:, 2]


def downwash_unitario_elica(x, dati_scia, n_pale, r_c, n_vatistas, dpsi, psi_max, psi_lineare):
    """Full helical tip vortex of the preceding blade, on the reference blade."""
    nodi, _ = nodi_elica(dati_scia, n_pale, dpsi, psi_max, psi_lineare)
    return _downwash_punti(stazioni_pala(x), nodi, r_c, n_vatistas)


def downwash_medio_azimutale(x, dati_scia, n_pale, r_c, n_vatistas, dpsi, psi_max,
                             psi_lineare, n_azimut):
    """Azimuthal average of the helix downwash at radius x (trapezoidal rule on a periodic
    function, i.e. the helix smeared into an axisymmetric vortex sheet)."""
    nodi, _ = nodi_elica(dati_scia, n_pale, dpsi, psi_max, psi_lineare)
    somma = np.zeros_like(x)
    for phi in np.linspace(0.0, 2 * np.pi, n_azimut, endpoint=False):
        punti = np.column_stack([x * np.cos(phi), x * np.sin(phi), np.zeros_like(x)])
        somma += _downwash_punti(punti, nodi, r_c, n_vatistas)
    return somma / n_azimut


def downwash_unitario_perturbazione(x, dati_scia, n_pale, r_c, n_vatistas, dpsi, psi_max,
                                    psi_lineare, n_azimut):
    """Helix minus its azimuthal average (option 2 of NOTE.md)."""
    w = downwash_unitario_elica(x, dati_scia, n_pale, r_c, n_vatistas, dpsi, psi_max, psi_lineare)
    w_medio = downwash_medio_azimutale(x, dati_scia, n_pale, r_c, n_vatistas, dpsi, psi_max,
                                       psi_lineare, n_azimut)
    return w - w_medio


# ---------------------------------------------------------------
# Solver
# ---------------------------------------------------------------
def bemt_downwash_vortice(x, theta, w_unitario, k, tol=TOL, max_it=MAX_IT):
    """BEMT with Prandtl tip loss plus a prescribed vortex downwash, fixed-point iteration.
    Same equations as modello.bemt_vortice (default options), with the 2D expression
    replaced by an arbitrary w(x):
        F lambda^2 + (sigma a/8) lambda - (sigma a/8)(theta x - lambda_v) = 0
        lambda_v = k Gamma_max/(Omega R^2) w(x)
        Cl = a (theta - (lambda + lambda_v)/x),  Gamma_max = max(0.5 Omega x R c Cl)
    The downwash enters the blade-element side only, as in v1.
    Gamma_max is the maximum over the whole blade: call it on the full grid.

    Returns lambda, Cl, F, Gamma_max [m^2/s], number of iterations.
    Raises RuntimeError if it does not converge (v1 printed a warning instead).
    """
    a = A0
    assert w_unitario.shape == x.shape, "w_unitario must be given on the grid x"
    F = np.ones_like(x)
    lam_old = np.zeros_like(x)
    gamma_max = 0.0
    for it in range(1, max_it + 1):
        lam_v = k * gamma_max / (OMEGA * R**2) * w_unitario
        lam = _radice_stabile(F, SIGMA * a / 8, -SIGMA * a / 8 * (theta * x - lam_v))
        non_fisico = ~(lam > 0)            # also catches NaN (negative discriminant)
        if np.any(non_fisico):
            # The vortex downwash exceeds the geometric angle: the annuli would need a
            # negative thrust, which the momentum balance dC_T = 4 F lambda^2 x dx
            # cannot represent (and F(lambda <= 0) is undefined). No physical solution.
            raise ValueError(f"non-positive inflow at iteration {it} for x in "
                             f"[{x[non_fisico].min():.3f}, {x[non_fisico].max():.3f}]")
        F = _fattore_prandtl(x, lam)
        Cl = a * (theta - (lam + lam_v) / x)
        gamma_max = np.max(0.5 * OMEGA * x * R * CORDA * Cl)    # Kutta-Joukowski
        if np.max(np.abs(lam - lam_old)) < tol:
            return lam, Cl, F, gamma_max, it
        lam_old = lam
    raise RuntimeError(f"bemt_downwash_vortice did not converge in {max_it} iterations")
