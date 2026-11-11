"""
Tests for modello_v2.py (step 3).

Run with:  python -m pytest -v test_modello_v2.py
"""
import numpy as np
import pytest

from dati_sperimentali import CASI, SCIA
from geometria_scia import nodi_elica
from modello import griglia_semicoseno, bemt_prandtl, bemt_vortice
from modello_v2 import downwash_unitario_2d, bemt_downwash_vortice, _downwash_punti
from vortice_3d import velocita_segmenti, velocita_vortice_2d

x = griglia_semicoseno(200)


# ---------------------------------------------------------------
# The new solver is the v1 solver when given the v1 downwash
# ---------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [5, 8, 12])
def test_regressione_vortice_2d_v1(theta_deg):
    """With the 2D-vortex downwash, bemt_downwash_vortice must reproduce modello.bemt_vortice
    (same iterations; values equal up to floating-point reordering). This guarantees that
    v1 and v2 differ only in the vortex model."""
    caso = CASI[theta_deg]
    th = np.radians(theta_deg)
    v1 = bemt_vortice(x, th, caso["r_v"], caso["z_v"])
    w = downwash_unitario_2d(x, caso["r_v"], caso["z_v"], 0.0)
    v2 = bemt_downwash_vortice(x, th, w, 1.0)
    for a, b in zip(v1[:4], v2[:4]):
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-15)
    assert v1[4] == v2[4]


def test_senza_vortice_uguale_prandtl():
    """With w = 0 the solver is the BEMT with Prandtl of v1."""
    th = np.radians(8)
    lam_p, F_p, _ = bemt_prandtl(x, th)
    lam, _, F, _, _ = bemt_downwash_vortice(x, th, np.zeros_like(x), 1.0)
    np.testing.assert_allclose(lam, lam_p, rtol=1e-12)
    np.testing.assert_allclose(F, F_p, rtol=1e-12)


def test_non_convergenza_solleva_errore():
    with pytest.raises(RuntimeError):
        bemt_downwash_vortice(x, np.radians(8), np.zeros_like(x), 1.0, max_it=2)


# ---------------------------------------------------------------
# Close to the vortex, the helix looks like the 2D vortex of v1
# ---------------------------------------------------------------
def test_vicino_al_vortice_elica_uguale_2d():
    """Points at distance d from the helix node of age 180 deg, in the plane normal to the
    local tangent. The 2D field Gamma/(2 pi d) grows like 1/d, while curvature and the rest
    of the helix add a bounded velocity: the relative difference must go to zero as O(d)."""
    nodi, eta = nodi_elica(SCIA[8], 2, 1.25, 1440.0, 180.0)
    k = int(np.argmin(np.abs(eta - 180.0)))
    C = nodi[k]
    t = nodi[k + 1] - nodi[k - 1]
    t /= np.linalg.norm(t)                                   # local tangent (age increasing)
    n1 = np.cross(t, [0.0, 0.0, 1.0]); n1 /= np.linalg.norm(n1)
    n2 = np.cross(t, n1)                                     # (t, n1, n2) orthonormal

    distanze = np.array([0.04, 0.02, 0.01, 0.005])
    rel = []
    for d in distanze:
        errs = []
        for alfa in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            off = d * (np.cos(alfa) * n1 + np.sin(alfa) * n2)
            S = len(nodi) - 1
            V3 = velocita_segmenti((C + off)[None, :], nodi[:-1], nodi[1:],
                                   np.ones(S), np.zeros(S), 1)[0]
            V2 = velocita_vortice_2d(d, 1.0, 0.0, 1) * np.cross(t, off / d)
            errs.append(np.linalg.norm(V3 - V2) / np.linalg.norm(V2))
        rel.append(max(errs))
    rel = np.array(rel)
    ordini = np.log2(rel[:-1] / rel[1:])
    assert np.all(ordini > 0.8), (rel, ordini)               # first order in d
    assert rel[-1] < 0.05, rel


def test_downwash_unitario_2d_formula_v1():
    """Sign and value of the 2D downwash: inboard of the vortex positive (downwash)."""
    w = downwash_unitario_2d(np.array([0.5, 1.0]), 0.9, 0.09, 0.0)
    assert w[0] > 0 and w[1] < 0
    assert w[0] == pytest.approx(0.4 / (2 * np.pi * (0.16 + 0.0081)), rel=1e-14)


def test_downwash_eccessivo_errore_esplicito():
    """A vortex downwash larger than the geometric angle (theta x) has no momentum solution:
    the solver must stop with an explicit error, not return NaN."""
    th = np.radians(5)
    w = np.full_like(x, 10.0)          # huge uniform downwash per unit Gamma
    with pytest.raises(ValueError, match="non-positive inflow"):
        bemt_downwash_vortice(x, th, w, 1.0)
