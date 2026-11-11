"""
Tests for vortice_3d.py against analytical solutions.

Run with:  python -m pytest -v test_vortice_3d.py
"""
import numpy as np
import pytest

from vortice_3d import velocita_segmenti, velocita_vortice_2d

GAMMA = 1.3     # arbitrary circulation [m^2/s]; the tests are linear in Gamma


def _segmento_asse_z(meta_lunghezza):
    """One segment on the z axis, from -l to +l (Gamma > 0 -> counter-clockwise seen from +z)."""
    A = np.array([[0.0, 0.0, -meta_lunghezza]])
    B = np.array([[0.0, 0.0, +meta_lunghezza]])
    return A, B


# ---------------------------------------------------------------
# Test 1: very long segment -> infinite 2D vortex (the v1 model)
# ---------------------------------------------------------------
@pytest.mark.parametrize("r_c, n", [(0.0, 1), (0.05, 1), (0.05, 2)])
def test_segmento_lungo_uguale_vortice_2d(r_c, n):
    """At mid-segment, a segment with l >> d must give Gamma d/(2 pi (d^2n + r_c^2n)^(1/n)).
    Finite-length error of the inviscid formula: 1 - l/sqrt(l^2+d^2) ~ d^2/(2 l^2) = 5e-9
    for l/d = 1e4, so rtol = 1e-7 tests the formula, not the truncation."""
    d = 0.1
    A, B = _segmento_asse_z(1e4 * d)
    P = np.array([[d, 0.0, 0.0]])
    V = velocita_segmenti(P, A, B, [GAMMA], [r_c], n)[0]

    atteso = velocita_vortice_2d(d, GAMMA, r_c, n)
    # direction: right-hand rule, vortex along +z, point on +x -> velocity along +y
    assert V[0] == pytest.approx(0.0, abs=1e-15)
    assert V[2] == pytest.approx(0.0, abs=1e-15)
    assert V[1] == pytest.approx(atteso, rel=1e-7)


def test_scully_uguale_modello_v1():
    """With n = 1 the 2D reference is exactly the formula used in bemt_vortice (v1):
    V = Gamma d / (2 pi (d^2 + r_c^2))."""
    d, r_c = 0.07, 0.03
    A, B = _segmento_asse_z(1e3)
    V = velocita_segmenti([[d, 0, 0]], A, B, [GAMMA], [r_c], 1)[0, 1]
    assert V == pytest.approx(GAMMA * d / (2 * np.pi * (d**2 + r_c**2)), rel=1e-7)


# ---------------------------------------------------------------
# Test 2: short segment vs analytical solution
# ---------------------------------------------------------------
@pytest.mark.parametrize("l_su_d", [0.1, 1.0, 3.0])
def test_segmento_corto_asse_perpendicolare(l_su_d):
    """Point on the perpendicular bisector at distance d, half-length l:
        V = Gamma/(4 pi d) * 2 l / sqrt(l^2 + d^2)
    For l = d this is sqrt(2)/2 of the infinite-line value."""
    d = 0.2
    l = l_su_d * d
    A, B = _segmento_asse_z(l)
    V = velocita_segmenti([[d, 0, 0]], A, B, [GAMMA], [0.0], 1)[0]
    atteso = GAMMA / (4 * np.pi * d) * 2 * l / np.sqrt(l**2 + d**2)
    assert V[1] == pytest.approx(atteso, rel=1e-12)
    assert abs(V[0]) < 1e-15 and abs(V[2]) < 1e-15


def test_segmento_corto_punto_fuori_centro():
    """General point (not on the bisector, not in a coordinate plane):
        V = Gamma/(4 pi h) (cos t1 - cos t2), direction e_theta around the segment.
    The symmetric test cannot catch a sign error in one of the two end terms; this one can."""
    A = np.array([[0.1, -0.2, 0.3]])
    B = np.array([[0.4, 0.5, -0.1]])
    P = np.array([[0.7, 0.1, 0.6]])

    e = (B - A)[0] / np.linalg.norm(B - A)
    r1, r2 = (P - A)[0], (P - B)[0]
    cos1 = r1 @ e / np.linalg.norm(r1)          # angle at A between segment and A->P
    cos2 = r2 @ e / np.linalg.norm(r2)          # angle at B between segment and B->P
    piede = A[0] + (r1 @ e) * e                 # foot of the perpendicular
    h_vec = P[0] - piede
    h = np.linalg.norm(h_vec)
    e_theta = np.cross(e, h_vec / h)
    atteso = GAMMA / (4 * np.pi * h) * (cos1 - cos2) * e_theta

    V = velocita_segmenti(P, A, B, [GAMMA], [0.0], 1)[0]
    np.testing.assert_allclose(V, atteso, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------
# Additional checks (core behaviour, superposition)
# ---------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2])
def test_nucleo_picco_e_limiti(n):
    """Vatistas core on a long segment: peak at h = r_c with value 2^(-1/n) Gamma/(2 pi r_c),
    V -> 0 on the axis (no singularity), and inviscid far field (h = 20 r_c, error < 1e-5 for n=2)."""
    r_c = 0.02
    A, B = _segmento_asse_z(1e3)
    h = np.array([0.0, 0.5, 1.0, 2.0, 20.0]) * r_c
    P = np.column_stack([h, np.zeros_like(h), np.zeros_like(h)])
    V = velocita_segmenti(P, A, B, [GAMMA], [r_c], n)[:, 1]

    assert V[0] == 0.0
    assert V[2] == pytest.approx(2**(-1 / n) * GAMMA / (2 * np.pi * r_c), rel=1e-6)
    assert V[2] > V[1] and V[2] > V[3]
    lontano = GAMMA / (2 * np.pi * h[4])
    assert V[4] == pytest.approx(lontano, rel=3e-3 if n == 1 else 1e-5)


def test_spezzare_segmento_non_cambia_risultato():
    """Biot-Savart is linear: a segment split into 7 collinear pieces gives the same velocity."""
    A, B = np.array([[0.0, 0, -0.3]]), np.array([[0.0, 0, 0.5]])
    nodi = A + np.linspace(0, 1, 8)[:, None] * (B - A)
    P = np.array([[0.2, 0.1, 0.05], [-0.1, 0.3, 0.9]])
    V1 = velocita_segmenti(P, A, B, [GAMMA], [0.01], 2)
    V7 = velocita_segmenti(P, nodi[:-1], nodi[1:], np.full(7, GAMMA), np.full(7, 0.01), 2)
    np.testing.assert_allclose(V7, V1, rtol=1e-12)


def test_anello_circolare_convergenza():
    """Ring of radius a discretised with N straight segments: velocity at the centre
    -> Gamma/(2a) along the axis. Chords lie inside the arc, so the error is O(1/N^2).
    Preview of the discretisation issue of step 2."""
    a = 1.0
    errori = []
    for N in (16, 32, 64, 128):
        psi = np.linspace(0, 2 * np.pi, N + 1)
        nodi = np.column_stack([a * np.cos(psi), a * np.sin(psi), np.zeros_like(psi)])
        V = velocita_segmenti([[0, 0, 0]], nodi[:-1], nodi[1:],
                              np.full(N, GAMMA), np.zeros(N), 1)[0]
        assert abs(V[0]) < 1e-12 and abs(V[1]) < 1e-12
        errori.append(abs(V[2] - GAMMA / (2 * a)) / (GAMMA / (2 * a)))
    ordini = np.log2(np.array(errori[:-1]) / np.array(errori[1:]))
    np.testing.assert_allclose(ordini, 2.0, atol=0.05)


def test_punto_sulla_linea_inviscido_solleva_errore():
    A, B = _segmento_asse_z(1.0)
    with pytest.raises(ValueError):
        velocita_segmenti([[0, 0, 0.3]], A, B, [GAMMA], [0.0], 1)
