"""
Geometry of the helical tip vortex of the preceding blade (v2.0, phase 1, step 2).

Reference frame (rotor-fixed, non-dimensional with R):
    X along the reference blade (azimuth 0), Z upwards along the rotation axis,
    Y = Z x X. The rotor turns counter-clockwise seen from above (Omega along +Z),
    so the blades move towards increasing azimuth psi.
    The reference blade occupies X in [x_root, 1], Y = 0, Z = 0.

Wake age and position:
    With N_b equally spaced blades, the preceding blade is now at azimuth 2 pi / N_b.
    The element of its tip vortex with age zeta was released zeta ago, when that blade
    was at azimuth 2 pi / N_b - zeta. Hence
        X = r(zeta) cos(2 pi/N_b - zeta),  Y = r(zeta) sin(2 pi/N_b - zeta),  Z = -z(zeta)
    with r(zeta), z(zeta) the measured contraction and descent (z positive downwards).
    At zeta = 2 pi / N_b (180 deg for two blades) the vortex is right below the reference
    blade: this is the point that v1 modelled as a 2D line vortex.

Vortex direction and sign (lifting-line reasoning):
    The relative wind seen by the blade is -U e_psi. Kutta-Joukowski, L = rho V x Gamma,
    gives lift along +Z when the bound-vortex vector is +Gamma e_r (root to tip).
    Vortex lines do not end, so at the tip the line turns into the trailed tip vortex and
    runs towards increasing age. The filament is therefore ordered by increasing age and
    carries +Gamma (Gamma = bound circulation at the tip, here Gamma_max, as in v1).
    Check (test_geometria_scia.py): it gives downwash inboard of the vortex and upwash
    outboard, the same signs as the 2D vortex of v1.

Not included (phase 1 models only the tip vortex of the preceding blade, as v1):
    - the bound vortex of the preceding blade: it lies on the line Y = 0, Z = 0, X < 0,
      which also contains the reference blade, so its induced velocity on the reference
      blade is exactly zero (points on the line of a straight vortex);
    - the root vortex and the inboard trailed sheet; the tip vortex of the reference blade.
"""
import numpy as np


def _pendenza_z_lontana(psi_z, z, psi_inizio_lineare):
    """Least-squares slope dz/dpsi [1/deg] of the measured descent for psi >= psi_inizio_lineare.
    Fig. 10 of TM-81232 shows z linear in age after 180 deg (vortex passage under the next
    blade), as in the Landgrebe / Kocurek-Tangler form; this slope is used to extend z."""
    sel = psi_z >= psi_inizio_lineare
    if np.count_nonzero(sel) < 3:
        raise ValueError("fewer than 3 data points in the linear-descent range")
    pendenza, _ = np.polyfit(psi_z[sel], z[sel], 1)
    return pendenza


def traiettoria(psi_deg, dati, psi_inizio_lineare):
    """Contraction r(psi)/R and descent z(psi)/R (positive down) of the tip vortex.

    Within the measured range: linear interpolation of the data points (the data are
    samples, every 20 deg, of the authors' smooth fitted curves).
    Beyond the last measured age (far-wake model 'b', chosen in NOTE.md):
        r = last measured value (the measured r is flat after ~260 deg),
        z = last measured value + slope * (psi - psi_last), slope from a least-squares
            fit of the data with psi >= psi_inizio_lineare.
    Ages below 0 are an error.

    dati: dict with arrays psi_r, r, psi_z, z (one entry of dati_sperimentali.SCIA).
    Returns r, z with the shape of psi_deg.
    """
    psi = np.asarray(psi_deg, dtype=float)
    if np.any(psi < 0):
        raise ValueError("negative wake age")
    psi_r, r_dati = dati["psi_r"], dati["r"]
    psi_z, z_dati = dati["psi_z"], dati["z"]
    assert psi_r[0] == 0 and psi_z[0] == 0, "data must start at age 0"
    assert np.all(np.diff(psi_r) > 0) and np.all(np.diff(psi_z) > 0)

    r = np.interp(psi, psi_r, r_dati)                      # np.interp holds the last value
    z = np.interp(psi, psi_z, z_dati)
    pendenza = _pendenza_z_lontana(psi_z, z_dati, psi_inizio_lineare)
    oltre = psi > psi_z[-1]
    z = np.where(oltre, z_dati[-1] + pendenza * (psi - psi_z[-1]), z)
    return r, z


def nodi_elica(dati, n_pale, dpsi_deg, psi_max_deg, psi_inizio_lineare):
    """Nodes of the preceding blade's tip vortex, from age 0 to psi_max_deg every dpsi_deg.

    Returns
    -------
    nodi : (N+1, 3) array   positions in units of R (frame in the module docstring)
    eta  : (N+1,) array     wake ages [deg]
    """
    n_seg = psi_max_deg / dpsi_deg
    if abs(n_seg - round(n_seg)) > 1e-9:
        raise ValueError("psi_max_deg must be a multiple of dpsi_deg")
    eta = np.linspace(0.0, psi_max_deg, int(round(n_seg)) + 1)
    r, z = traiettoria(eta, dati, psi_inizio_lineare)
    azimut = 2 * np.pi / n_pale - np.radians(eta)
    nodi = np.column_stack([r * np.cos(azimut), r * np.sin(azimut), -z])
    return nodi, eta


def stazioni_pala(x):
    """Points (M, 3) on the reference blade (X = x, Y = 0, Z = 0)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
