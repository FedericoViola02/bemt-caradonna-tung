"""
Induced velocity of straight, finite vortex segments (Biot-Savart law).

Building block for the 3D helical tip vortex of the preceding blade (v2.0, phase 1).
A curved filament is approximated by a chain of straight segments; the velocity
induced by each segment has a closed form, so the only discretisation error is
the geometric one (chord vs arc), which is tested separately.

Units: the functions are unit-agnostic but must be used consistently.
If lengths are in [m] and Gamma in [m^2/s], velocities are in [m/s].
If lengths are in R and Gamma in Omega*R^2, velocities are in Omega*R (i.e. lambda).

Sign convention: Gamma > 0 means counter-clockwise rotation when looking from B
towards A, i.e. the right-hand rule with the thumb along A -> B.

Reference for the formula and the core model:
    Bhagwat, M.J. & Leishman, J.G., "Generalized Viscous Vortex Model for Application
    to Free-Vortex Wake and Aeroacoustic Calculations", AHS 58th Forum, 2002.
    Leishman, J.G., Principles of Helicopter Aerodynamics, 2nd ed., ch. 10.
"""
import numpy as np


def velocita_segmenti(P, A, B, gamma, r_c, n_vatistas):
    """Velocity induced at points P by straight vortex segments A_j -> B_j, summed over j.

    Inviscid Biot-Savart for a finite segment (Katz & Plotkin form):
        V = Gamma/(4 pi) * (r1 x r2)/|r1 x r2|^2 * r0 . (r1/|r1| - r2/|r2|)
    with r0 = B - A, r1 = P - A, r2 = P - B.
    The geometric factor r0 . (r1/|r1| - r2/|r2|) / |r0| equals cos(t1) - cos(t2),
    and |r1 x r2| / |r0| = h is the distance of P from the segment's line, so the
    magnitude is Gamma/(4 pi h) (cos t1 - cos t2), the textbook result.

    Viscous core (Vatistas family): the 1/h^2 factor is replaced by
        1 / (h^(2n) + r_c^(2n))^(1/n)
    so that the swirl velocity follows the Vatistas profile
        V_theta = Gamma/(2 pi) * h / (h^(2n) + r_c^(2n))^(1/n)
    in the infinite-line limit. n = 1 is the Scully core (the one in the 2D model of
    v1), n = 2 is the Bagai-Leishman core (close to measured rotor tip vortices).
    For every n the swirl velocity peaks at h = r_c, where it is 2^(-1/n) times the
    inviscid value Gamma/(2 pi r_c).

    Parameters
    ----------
    P : (M, 3) array      evaluation points
    A, B : (S, 3) arrays  start and end of each segment
    gamma : (S,) array    circulation of each segment (sign as above)
    r_c : (S,) array      core radius of each segment, >= 0 (same length unit as P).
                          Per segment, because the core grows with wake age.
    n_vatistas : float    core-shape exponent, >= 1 (no default: it is a model choice)

    Returns
    -------
    V : (M, 3) array      induced velocity at each point, summed over all segments

    Raises ValueError if r_c = 0 and a point lies on a segment's line (singularity),
    or if a segment has zero length.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
    r_c = np.atleast_1d(np.asarray(r_c, dtype=float))
    M, S = P.shape[0], A.shape[0]
    assert P.shape == (M, 3), f"P must be (M, 3), got {P.shape}"
    assert A.shape == (S, 3) and B.shape == (S, 3), f"A, B must be (S, 3): {A.shape}, {B.shape}"
    assert gamma.shape == (S,), f"gamma must be (S,), got {gamma.shape}"
    assert r_c.shape == (S,), f"r_c must be (S,), got {r_c.shape}"
    if n_vatistas < 1:
        raise ValueError(f"n_vatistas must be >= 1, got {n_vatistas}")
    if np.any(r_c < 0):
        raise ValueError("r_c must be >= 0")

    r0 = B - A                                     # (S, 3)
    L = np.linalg.norm(r0, axis=1)                 # (S,) segment lengths
    if np.any(L <= 0):
        raise ValueError("zero-length vortex segment")

    r1 = P[:, None, :] - A[None, :, :]             # (M, S, 3)
    r2 = P[:, None, :] - B[None, :, :]             # (M, S, 3)
    r1n = np.linalg.norm(r1, axis=2)               # (M, S)
    r2n = np.linalg.norm(r2, axis=2)
    cross = np.cross(r1, r2)                       # (M, S, 3), along e_theta * |r1||r2| sin
    h = np.linalg.norm(cross, axis=2) / L          # (M, S) distance from the line

    # Inviscid singularity: point on the vortex line. With a core the velocity is
    # simply zero there (cross = 0), which is the physical limit.
    tol_h = 1e-12 * L                              # relative to segment size
    sing = (h < tol_h) & (r_c[None, :] == 0.0)
    if np.any(sing):
        raise ValueError("evaluation point on an inviscid vortex line (r_c = 0)")

    # cos(t1) - cos(t2) times |r0|: projection of the unit vectors on r0.
    # When P is on the line, r1n or r2n may be 0 only if P coincides with A or B;
    # those points have h = 0 and cross = 0, so they contribute 0 (handled below).
    with np.errstate(invalid="ignore", divide="ignore"):
        proiez = (np.einsum("msk,sk->ms", r1, r0) / r1n
                  - np.einsum("msk,sk->ms", r2, r0) / r2n)        # (M, S)
    proiez = np.where((r1n > 0) & (r2n > 0), proiez, 0.0)

    # Core-regularised 1/|r1 x r2|^2  ->  1 / (L^2 (h^2n + r_c^2n)^(1/n))
    n = float(n_vatistas)
    denom = L[None, :]**2 * (h**(2 * n) + r_c[None, :]**(2 * n))**(1.0 / n)   # (M, S)
    with np.errstate(invalid="ignore", divide="ignore"):
        fattore = gamma[None, :] / (4 * np.pi) * proiez / denom              # (M, S)
    fattore = np.where(denom > 0, fattore, 0.0)

    V = np.einsum("ms,msk->mk", fattore, cross)   # (M, 3)
    assert np.all(np.isfinite(V)), "non-finite induced velocity"
    return V


def velocita_vortice_2d(d, gamma, r_c, n_vatistas):
    """Swirl speed of an infinite straight vortex with a Vatistas core at distance d:
        V_theta = Gamma/(2 pi) * d / (d^(2n) + r_c^(2n))^(1/n)
    For n = 1 this is the 2D model of v1 (Scully core). Used as reference in the tests."""
    d = np.asarray(d, dtype=float)
    n = float(n_vatistas)
    return gamma / (2 * np.pi) * d / (d**(2 * n) + r_c**(2 * n))**(1.0 / n)
