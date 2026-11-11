"""
Tests for geometria_scia.py (step 2).

Run with:  python -m pytest -v test_geometria_scia.py
"""
import numpy as np
import pytest

from dati_sperimentali import SCIA, R1_NUCLEO
from geometria_scia import traiettoria, nodi_elica, stazioni_pala
from vortice_3d import velocita_segmenti

PSI_LINEARE = 180.0     # start of the linear descent (Fig. 10 of TM-81232)
N_PALE = 2


def downwash_elica(x, theta, dpsi, psi_max, gamma=1.0):
    """Downwash (positive down) on the reference blade induced by the preceding blade's
    tip vortex, per unit Gamma/(Omega R^2): the result is directly a lambda."""
    nodi, _ = nodi_elica(SCIA[theta], N_PALE, dpsi, psi_max, PSI_LINEARE)
    S = len(nodi) - 1
    V = velocita_segmenti(stazioni_pala(x), nodi[:-1], nodi[1:],
                          np.full(S, gamma), np.full(S, R1_NUCLEO[theta]), 2)
    return -V[:, 2]


# ---------------------------------------------------------------
# Trajectory: data reproduced, far-wake extension as decided
# ---------------------------------------------------------------
@pytest.mark.parametrize("theta", [5, 8, 12])
def test_traiettoria_ridà_i_dati(theta):
    d = SCIA[theta]
    r, _ = traiettoria(d["psi_r"], d, PSI_LINEARE)
    _, z = traiettoria(d["psi_z"], d, PSI_LINEARE)
    np.testing.assert_allclose(r, d["r"], atol=1e-15)
    np.testing.assert_allclose(z, d["z"], atol=1e-15)


@pytest.mark.parametrize("theta", [5, 8, 12])
def test_estensione_scia_lontana(theta):
    """Beyond the data: r constant (last value), z continuous and linear with the
    least-squares slope of the data at psi >= 180 deg."""
    d = SCIA[theta]
    psi = np.array([d["psi_z"][-1], d["psi_z"][-1] + 100.0, d["psi_z"][-1] + 1000.0])
    r, z = traiettoria(psi, d, PSI_LINEARE)
    sel = d["psi_z"] >= PSI_LINEARE
    pendenza = np.polyfit(d["psi_z"][sel], d["z"][sel], 1)[0]
    assert z[0] == pytest.approx(d["z"][-1], abs=1e-15)
    np.testing.assert_allclose(np.diff(z) / np.diff(psi), pendenza, rtol=1e-12)
    psi_r = d["psi_r"][-1] + np.array([10.0, 500.0])
    np.testing.assert_allclose(traiettoria(psi_r, d, PSI_LINEARE)[0], d["r"][-1], atol=1e-15)
    # the slope must be a descent, of the order of the measured one (Fig. 10)
    assert 0.0005 < pendenza < 0.002


def test_età_negativa_errore():
    with pytest.raises(ValueError):
        traiettoria([-1.0], SCIA[8], PSI_LINEARE)


# ---------------------------------------------------------------
# Helix nodes: position of the key points
# ---------------------------------------------------------------
@pytest.mark.parametrize("theta", [5, 8, 12])
def test_nodi_punti_chiave(theta):
    """Age 0: tip of the preceding blade (X = -1). Age 180 deg: right below the reference
    blade, at the measured (r_v, z_v), i.e. the point of the 2D vortex of v1."""
    nodi, eta = nodi_elica(SCIA[theta], N_PALE, 2.5, 720.0, PSI_LINEARE)
    np.testing.assert_allclose(nodi[0], [-1.0, 0.0, 0.0], atol=1e-15)
    k = int(np.argmin(np.abs(eta - 180.0)))
    r180, z180 = traiettoria([180.0], SCIA[theta], PSI_LINEARE)
    np.testing.assert_allclose(nodi[k], [r180[0], 0.0, -z180[0]], atol=1e-14)
    # the wake only contracts and descends
    raggio = np.hypot(nodi[:, 0], nodi[:, 1])
    assert np.all(raggio <= 1.0 + 1e-15) and np.all(np.diff(nodi[:, 2]) <= 0)


def test_verso_rotazione():
    """Increasing age = moving backwards relative to the blade rotation (towards decreasing
    azimuth): the azimuth of the nodes decreases monotonically."""
    nodi, _ = nodi_elica(SCIA[8], N_PALE, 5.0, 720.0, PSI_LINEARE)
    azimut = np.unwrap(np.arctan2(nodi[:, 1], nodi[:, 0]))
    assert np.all(np.diff(azimut) < 0)


# ---------------------------------------------------------------
# Sign of the induced velocity
# ---------------------------------------------------------------
def test_segno_downwash_come_v1():
    """Only the stretch of helix passing under the blade (ages 120-240 deg): downwash inboard
    of the vortex, upwash outboard, as the 2D vortex of v1 (lam_v > 0 for x < r_v)."""
    nodi, eta = nodi_elica(SCIA[8], N_PALE, 1.0, 360.0, PSI_LINEARE)
    sel = (eta >= 120) & (eta <= 240)
    tratto = nodi[sel]
    S = len(tratto) - 1
    x = np.array([0.6, 0.85, 0.95, 0.99])          # r_v(180) = 0.904 at 8 deg
    w = -velocita_segmenti(stazioni_pala(x), tratto[:-1], tratto[1:],
                           np.ones(S), np.full(S, R1_NUCLEO[8]), 2)[:, 2]
    assert w[0] > 0 and w[1] > 0
    assert w[2] < 0 and w[3] < 0


# ---------------------------------------------------------------
# Convergence with the number of segments
# ---------------------------------------------------------------
def test_convergenza_passo_angolare_secondo_ordine():
    """Chords of a smooth curve: error O(dpsi^2). Nodes include the 20-deg data points,
    where the interpolated trajectory has slope discontinuities, so the order is kept.
    Reference: dpsi = 0.3125 deg. Wake length fixed (1440 deg) for this test."""
    x = np.linspace(0.2, 1.0, 41)
    rif = downwash_elica(x, 8, 0.3125, 1440.0)
    passi = (5.0, 2.5, 1.25)
    err = [np.max(np.abs(downwash_elica(x, 8, d, 1440.0) - rif)) for d in passi]
    ordini = np.log2(np.array(err[:-1]) / np.array(err[1:]))
    assert np.all((ordini > 1.8) & (ordini < 2.4)), ordini
    # chosen step: 1.25 deg, error below 1e-3 of the peak downwash
    assert err[-1] / np.max(np.abs(rif)) < 1e-3
