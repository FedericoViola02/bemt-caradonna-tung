# BEMT with Tip-Vortex Correction for a Hovering Rotor
### Validation against the Caradonna & Tung benchmark (NASA TM-81232)

A Blade Element Momentum Theory (BEMT) model for a rotor in hover, written from scratch in Python and validated against the experimental data of Caradonna & Tung (1981). The model is then extended with a correction for the tip vortex of the preceding blade, based on the measured wake geometry.

The aim is not only to match the experiment, but to understand **where and why** a low-order model deviates from it.

**Main result (v1).** Classical BEMT with Prandtl tip loss overpredicts the thrust by 22–37%. Adding the tip vortex of the preceding blade as a 2D line vortex, with the measured wake geometry and **no parameter tuning**, reduces the thrust error to 4–9% at three collective pitch angles. The spanwise loading improves on the inboard part of the blade but is strongly overpredicted near the tip, so part of the thrust agreement comes from compensating errors.

**Main result (v2.0, phase 1).** Replacing the 2D vortex with the real 3D helical filament, built from the measured wake trajectory, shows that **most of the v1 thrust correction came from double counting** the wake already contained in the momentum balance. The full helix has no physical solution in the BEMT, and the part of the helix that momentum theory does not contain (helix minus its azimuthal mean) makes the thrust error larger (+30–51%) and the tip overprediction worse. The curvature and finite length of the vortex do **not** explain the tip error. See [Version 2.0, phase 1](#version-20-phase-1-3d-helical-tip-vortex).

![Spanwise loading at three pitch angles](figure/validazione_cl.png)

---

## Test case

| Parameter | Value |
|---|---|
| Rotor radius R | 1.143 m |
| Chord c | 0.1905 m |
| Number of blades | 2 |
| Airfoil | NACA 0012, untwisted, untapered |
| Solidity σ | 0.106 |
| Rotational speed | 1250 rpm (tip Mach number 0.44) |
| Collective pitch | 5°, 8°, 12° |

---

## Method

Notation: x = r/R, inflow ratio λ = v/(ΩR), collective pitch θ.

**1. Uniform inflow.** Blade element theory, C_T = (σa/2)(θ/3 − λ/2), combined with momentum theory, C_T = 2λ².

**2. BEMT.** The two balances are applied to each annulus of width dx:

- momentum: dC_T = 4Fλ²x dx
- blade element: dC_T = (σa/2)(θx² − λx) dx

which gives, at each radial station, Fλ² + (σa/8)λ − (σa/8)θx = 0.

**3. Prandtl tip loss.** F = (2/π) arccos(e^(−f)), with f = (N/2)(1 − x)/λ. Since F depends on λ, the system is solved by fixed-point iteration (tolerance 1e-8 on max|Δλ|, convergence in about 10 iterations). The root is computed as λ = −2C/(B + √(B² − 4AC)), which is algebraically equivalent to the standard formula but remains defined at the tip, where F = 0.

**4. Tip-vortex correction.** The tip vortex shed by the preceding blade (vortex age 180°) passes below the blade. It runs in the azimuthal direction, so in the radial–vertical plane of the blade it appears as a point vortex at (r_v, z_v), taken from the measured wake geometry (Figs. 9–10 of the report). Its vertical induced velocity (Biot–Savart, 2D) is added on the blade-element side only:

- λ_v = −Γ_max (x − r_v) / (2π d² ΩR²), with d² = (x − r_v)² + z_v²
- Fλ² + (σa/8)λ − (σa/8)(θx − λ_v) = 0
- Cl = a(θ − (λ + λ_v)/x)

The vortex strength is the maximum bound circulation of the blade, Γ = ½ ΩxR c Cl (Kutta–Joukowski), consistent with the hot-wire measurements of the same authors. Since Γ_max depends on the solution, it is updated within the same fixed-point loop.

**Numerics.** Near the tip the loading behaves as √(1 − x), so with uniform spacing the trapezoidal rule converges with order ~1.5 instead of 2. A half-cosine grid clusters the stations towards the tip and reaches a relative error below 2·10⁻⁴ on C_T with only 50 stations. All results use a half-cosine grid with 200 stations.

![Grid convergence](figure/convergenza.png)

### Assumptions

- Steady hover, fully developed wake
- Linear lift curve, a = 5.7 /rad (effective value, see the compressibility test); small angles (φ ≈ λ/x)
- Blade root at 0.2R; profile drag neglected in the thrust
- Vortex modelled as a straight 2D line vortex: its curvature (ring of radius ~0.9R, at a distance of ~0.1R from the blade), its finite length, the older turns of the helix and the inner vortex sheet are neglected

---

## Results

### Baseline: BEMT with Prandtl tip loss

The model overpredicts the thrust at all pitch angles, and the sectional lift at all stations (at 8°: +24%, +18%, +22%, +11%, +12% at r/R = 0.50, 0.68, 0.80, 0.89, 0.96). Reading the model C_T–θ curve at the measured thrust values, the model matches the experiment at an angle about 1–1.5° lower: the missing physics behaves like a nearly constant excess of angle of attack, i.e. a missing downwash.

### Tip-vortex correction: predictive validation

Vortex positions read from the report and never adjusted; vortex strength equal to the maximum bound circulation at all angles.

| θ | C_T error, Prandtl | C_T error, with vortex | Mean \|Cl error\|, Prandtl | Mean \|Cl error\|, with vortex | Same, without r/R = 0.96 |
|---|---|---|---|---|---|
| 5° | +37.0% | +9.4% | 33% | 26% | 39% → 15% |
| 8° | +21.6% | +3.6% | 17% | 15% | 19% → 9% |
| 12° | +21.7% | +6.2% | 20% | 21% | 20% → 14% |

![Thrust coefficient versus collective pitch](figure/ct_theta.png)

- **The thrust improves at all three angles**, by a factor of 3–6, without any tuning.
- **On the inboard part of the blade the error is reduced on average**, but not uniformly: at 8° it drops from +24% to −5% at r/R = 0.50, but only from +18% to −14% and from +22% to −17% at 0.68 and 0.80. The systematic overprediction turns into a moderate underprediction.
- **Near the tip the loading is strongly overpredicted** (+39% to +68% at r/R = 0.96). Over all five stations the mean error is roughly unchanged, so the very good thrust agreement is **partly due to compensating errors** between the inboard and tip regions.

### Sensitivity studies

| Test | Effect | Conclusion |
|---|---|---|
| Vortex position ±0.025R (8°) | Cl error at 0.80R ranges from −42% to −5% | The loading is very sensitive to the vortex position, as concluded in the original report. The best combination is a calibration, not a validation. |
| Without Prandtl tip loss (8°) | Tip error +90% | The tip loading must vanish: Prandtl is still needed. This does not prove that the two effects are independent, since F also models the helical wake of all blades. |
| Mean vortex downwash removed (8°) | C_T error +19%, inboard now overpredicted | The experiment lies between this case and the full vortex: partial double counting between the actuator-disk inflow and the discrete vortex. |
| Weaker vortex, k = 0.81 (5°) | C_T error from +9% to +16% | The residual error at 5° is not explained by the vortex strength. |
| Compressibility, Prandtl–Glauert (8°) | C_T error from +3.6% to +6.3%, tip worse | The incompressible assumption is not the cause of the overprediction. |
| Viscous core, r_c = 0.01–0.02 (5°) | Mean Cl error from 26% to 21%, C_T slightly worse | Secondary effect; it does not explain the tip error. |

---

## Version 2.0, phase 1: 3D helical tip vortex

**Question.** Do the curvature and finite length of the real tip vortex explain the tip overprediction of v1 (cause 1 in the discussion below)?

### Method

**1. Finite vortex segment.** Biot–Savart for a straight segment (Katz & Plotkin form), V = Γ/(4πh)(cos θ₁ − cos θ₂), with a Vatistas viscous core: the 1/h² factor becomes 1/(h²ⁿ + r_c²ⁿ)^(1/n) (n = 1 Scully, n = 2 Bagai–Leishman; Bhagwat & Leishman 2002).

**2. Helical wake of the preceding blade.** With two blades, the vortex element of age ζ lies at azimuth π − ζ, at the measured radius r(ζ) and depth z(ζ). The trajectory up to 380° is digitised from Fig. 10 of TM-81232; beyond it, r is held constant and z continues linearly, as the measurements up to 440° show (Fig. 9). The classical wake laws (Landgrebe, Kocurek–Tangler) contract the wake to ~0.78R and, according to the authors, overpredict the thrust by ~20% for this rotor, so they are not used. The filament is ordered by increasing age and carries +Γ_max: from Kutta–Joukowski the bound vortex runs from root to tip and continues into the trailed vortex. The core radius is the measured one (TM-81316: 0.0027R at 8°, 0.0081R at 12°, average 0.003R at 5°).

**3. Coupling with the BEMT.** The wake geometry is prescribed, so the vortex downwash is λ_v = k Γ_max/(ΩR²) · w(x), with w(x) computed once; only Γ_max is updated, in the same fixed-point loop as v1, and λ_v enters the blade-element side only. Three vortex models are compared:

| Model | Downwash w(x) | Meaning |
|---|---|---|
| V1 | 2D line vortex at the measured position of age 180° | v1 |
| E1 | full helix | contains the axisymmetric wake, already in the momentum balance |
| E2 | helix minus its azimuthal average | only the effect of the vortex being concentrated in a line |

### Verification

- The segment formula reproduces the infinite 2D vortex (relative error < 10⁻⁷) and the analytical finite-segment solution (10⁻¹²); a ring of N segments converges to Γ/(2a) with order 2.
- Close to the vortex the helix tends to the 2D vortex, with a relative difference proportional to the distance d (25% at d = 0.04R, 4% at 0.005R). At the blade d ≈ 0.05–0.11R, so there the two models differ substantially.
- Angular step: second-order convergence, Δψ = 1.25° (error 4.5·10⁻⁴). Wake length: the full helix converges slowly (~1/ψ_max², it behaves like a vortex cylinder), E2 converges within 4 turns.
- With the 2D-vortex downwash, the new solver reproduces v1 exactly (same iterations, 10⁻¹²): v1 and v2 differ only in the vortex model.

![Wake convergence](figure/convergenza_scia.png)

### Results

No parameter is adjusted to the data. Error on C_T, and on Cl at r/R = 0.96:

| Model | C_T 5° | C_T 8° | C_T 12° | Cl at 0.96R (5° / 8° / 12°) |
|---|---|---|---|---|
| BEMT + Prandtl | +37% | +22% | +22% | +8% / +12% / +22% |
| V1: 2D vortex | +9% | +4% | +6% | +68% / +39% / +49% |
| V1 without its disk-averaged downwash | +34% | +19% | +20% | +96% / +54% / +63% |
| E1: full helix | no solution | no solution | no solution | – |
| E2: helix − azimuthal mean | +51% | +31% | +30% | +138% / +76% / +64% |

Declared checks: changing the vortex core (r_c = 0.0047R, or n = 1) changes C_T by less than 1%; the vortex strength measured at 5° (A = 0.74, TM-81232 Fig. 12) moves E2 from +51% to +44%.

![Comparison of the vortex models](figure/confronto_v2.png)

![Thrust coefficient, v1 and v2 models](figure/ct_theta_v2.png)

### Conclusions

1. **The full helix counts the wake twice.** Far from the blade the helix is a vortex cylinder, i.e. the wake that the actuator disk already represents. Its downwash on the inner blade exceeds the geometric angle θx, so the inner annuli would need negative thrust; a direct search over all admissible Γ_max confirms that no self-consistent solution exists at any pitch.
2. **The v1 thrust agreement came mostly from the same double counting.** The 2D vortex adds a mean downwash on the inner blade (~0.01 at x = 0.5); without it the v1 error returns to +19–34%.
3. **The concentrated vortex itself makes the tip worse.** E2 gives upwash outboard of the vortex (r > r_v ≈ 0.9) and raises the tip loading, while the measured loading drops there. The curvature and finite length of the preceding blade's vortex are not the cause of the tip error.
4. **The remaining inconsistency is structural.** E2 assumes that the momentum balance contains the azimuthal mean of the wake, which holds only for a wake without contraction: outboard of r_v the contracted wake is not there, while the actuator disk and Prandtl (wake edge at x = 1) assume it is. A consistent model has to replace momentum theory with the prescribed wake (phase 2).

**Note on the experimental data.** The sectional C_L tabulated in Appendix A of TM-81232 (Tables 10 and 28, 5° and 12°) is printed in the wrong columns for the three outer stations; integrating the tabulated pressures (`verifica_cl_tabelle.py`) confirms the values of the figures, which are the ones used here. At 12°, r/R = 0.68, figure (0.492) and table (0.5455) disagree and the pressure integral cannot discriminate between them: that point carries an uncertainty of about ±0.05.

---

## Discussion and limitations

**The tip overprediction is systematic** at all angles and survives every sensitivity test. The most likely causes are:

1. ~~the straight 2D vortex overestimates the upwash outboard of the vortex compared with the real curved, finite filament~~: tested in v2.0 phase 1 and ruled out, the 3D filament gives more upwash, not less;
2. Prandtl's factor places the wake edge at x = 1, while the contracted wake has its edge at ~0.9R;
3. three-dimensional tip effects on a blade of aspect ratio 6;
4. uncertainty in the experimental data near the tip: at 12°, even the lifting-surface code of the original authors overpredicts the loading at r/R = 0.96.

**The hybrid model double counts part of the wake.** The actuator-disk inflow already represents the average effect of the whole wake, including the tip vortices, and the discrete vortex is added on top of it. The model cannot determine how much of the vortex is already contained in the momentum inflow: adding it in full counts too much, removing its mean counts too little.

**Experimental data** were read by eye from the figures of the report (uncertainty about ±0.005 on Cl), and the vortex positions from the wake-geometry plots. The full wake trajectory used in v2 was digitised at 400 dpi (uncertainty about ±0.003R).

### Next step

The consistent way to remove the double counting is to drop momentum theory and describe the whole wake with vortex filaments: a **prescribed-wake model**, with helical tip vortices following empirical wake laws (Landgrebe, Kocurek & Tangler) and Biot–Savart induction on a lifting line.

---

## Repository structure

| File | Content |
|---|---|
| `modello.py` | Rotor parameters and model functions (uniform inflow, BEMT, Prandtl, tip-vortex correction) |
| `dati_sperimentali.py` | Experimental data from NASA TM-81232, with the figure each value comes from |
| `main.py` | All v1 analyses in sequence; figures are saved in `figure/` |
| `vortice_3d.py` | v2: Biot–Savart for finite straight vortex segments with Vatistas core |
| `geometria_scia.py` | v2: helical tip vortex of the preceding blade from the measured trajectory |
| `modello_v2.py` | v2: BEMT with a prescribed vortex downwash (2D vortex, full helix, helix − azimuthal mean) |
| `studio_convergenza_scia.py` | v2: convergence with angular step and wake length |
| `confronto_v2.py` | v2: comparison of all models with the experiment at 5°, 8°, 12° |
| `test_*.py` | Tests against analytical solutions, limiting cases and the v1 solver (pytest) |
| `stile_figure.py` | Shared figure style: the same model has the same color in every figure |
| `digitalizza_fig10.py`, `verifica_cl_tabelle.py` | Data provenance: digitisation of the wake trajectory, check of the tabulated C_L |

Comments, printed output and figure labels are in English; function and variable names are in Italian.

### How to run

Requirements: Python 3, NumPy, Matplotlib, pytest (exact versions in `requirements.txt`).

```
pip install -r requirements.txt
python -m pytest                     # 34 tests
python main.py                       # v1 analyses
python studio_convergenza_scia.py    # v2 wake convergence
python confronto_v2.py               # v2 comparison (about 1 minute)
```

---

## References

1. F. X. Caradonna, C. Tung, *Experimental and Analytical Studies of a Model Helicopter Rotor in Hover*, NASA TM-81232, 1981.
2. C. Tung, S. L. Pucci, F. X. Caradonna, H. A. Morse, *The Structure of Trailing Vortices Generated by Model Rotor Blades*, NASA TM-81316, 1981.
3. J. G. Leishman, *Principles of Helicopter Aerodynamics*, Cambridge University Press, chapter 3.
4. L. N. Sankar, AE 6070 Rotorcraft Aerodynamics lecture notes, Georgia Institute of Technology.
5. M. J. Bhagwat, J. G. Leishman, *Generalized Viscous Vortex Model for Application to Free-Vortex Wake and Aeroacoustic Calculations*, AHS 58th Annual Forum, 2002.
6. J. Katz, A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001.
