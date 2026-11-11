"""
Experimental data from Caradonna & Tung (NASA TM-81232, 1981), 1250 rpm, M_tip = 0.439.

All Cl values and vortex positions were read by eye from the figures of the report
(uncertainty about +-0.005), not from numerical tables.
The tables of Appendix A (Tables 10, 17, 28) print the C_L of the outer three stations in
the wrong columns at 5 and 12 deg: the figure values used here are the ones consistent with
the tabulated pressures (check: verifica_cl_tabelle.py). 12 deg at r/R = 0.68 is uncertain:
Fig. 15 gives 0.492, Table 28 gives 0.5455 (see NOTE.md).
"""
import numpy as np

# Radial stations where the pressures were measured (r/R)
X_STAZIONI = np.array([0.50, 0.68, 0.80, 0.89, 0.96])

# For each collective pitch angle [deg]:
#   CT   measured thrust coefficient
#   Cl   sectional Cl at the 5 stations
#   r_v  radial position of the preceding blade's tip vortex (age 180 deg), Fig. 10
#   z_v  distance of the vortex below the rotor plane (age 180 deg), Fig. 10
CASI = {
    5: {
        "CT": 0.0021,                                             # Fig. 16
        "Cl": np.array([0.103, 0.123, 0.118, 0.150, 0.153]),      # Fig. 16
        "r_v": 0.915,
        "z_v": 0.05,
    },
    8: {
        "CT": 0.00459,                                            # Fig. 4
        "Cl": np.array([0.236, 0.284, 0.291, 0.319, 0.274]),      # Fig. 14
        "r_v": 0.905,
        "z_v": 0.09,
    },
    12: {
        "CT": 0.0079,                                             # Fig. 15
        "Cl": np.array([0.425, 0.492, 0.500, 0.510, 0.422]),      # Fig. 15
        "r_v": 0.885,
        "z_v": 0.11,
    },
}

# ---------------------------------------------------------------
# v2.0: tip-vortex trajectory versus wake age (helical wake of the preceding blade)
# ---------------------------------------------------------------
# Source: Caradonna & Tung, NASA TM-81232, Figure 10 "Wake geometry for various pitch
# settings" (PDF page 53, printed page 51). The curves are the authors' fits of the
# hot-wire measurements (Fig. 9 shows the 8 deg measured points behind the fit).
# Digitised from a 400 dpi render of the PDF with digitalizza_fig10.py (pixel columns
# every 20 deg). Reading uncertainty about +-0.003 (line thickness).
# psi = wake age [deg], r = r/R, z = distance below the rotor plane / R (positive down).
# Missing ages: curves overlapping or hidden by labels in the figure (not interpolated here).
SCIA = {
    5: {
        "psi_r": np.array([0, 20, 40, 60, 100, 120, 140, 160, 180, 200, 220, 240, 260,
                           280, 300, 320, 340, 360]),
        "r": np.array([1.000, 0.988, 0.984, 0.974, 0.955, 0.945, 0.935, 0.926, 0.917, 0.909,
                       0.901, 0.894, 0.888, 0.883, 0.881, 0.879, 0.879, 0.878]),
        "psi_z": np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260,
                           280, 300, 320, 340, 360, 380]),
        "z": np.array([0.000, 0.000, 0.000, 0.004, 0.010, 0.018, 0.027, 0.034, 0.043, 0.052,
                       0.067, 0.083, 0.099, 0.115, 0.131, 0.147, 0.162, 0.179, 0.196, 0.212]),
    },
    8: {
        "psi_r": np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 260, 280,
                           300, 320, 340, 360]),
        "r": np.array([1.000, 0.988, 0.978, 0.967, 0.956, 0.945, 0.934, 0.924, 0.913, 0.904,
                       0.896, 0.888, 0.877, 0.875, 0.874, 0.871, 0.872, 0.872]),
        "psi_z": np.array([0, 20, 40, 60, 80, 100, 120, 160, 180, 200, 220, 240, 260, 280,
                           300, 320, 340, 360, 380]),
        "z": np.array([0.000, 0.000, 0.004, 0.014, 0.027, 0.038, 0.051, 0.076, 0.088, 0.109,
                       0.129, 0.150, 0.169, 0.191, 0.213, 0.231, 0.252, 0.273, 0.292]),
    },
    12: {
        "psi_r": np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260,
                           280, 300, 320, 340, 360, 380]),
        "r": np.array([1.000, 0.988, 0.967, 0.953, 0.938, 0.925, 0.913, 0.900, 0.888, 0.878,
                       0.868, 0.861, 0.855, 0.852, 0.851, 0.850, 0.849, 0.850, 0.849, 0.849]),
        "psi_z": np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260,
                           280, 300, 320, 340, 360]),
        "z": np.array([0.000, 0.000, 0.009, 0.023, 0.038, 0.052, 0.067, 0.083, 0.097, 0.113,
                       0.143, 0.173, 0.205, 0.236, 0.267, 0.298, 0.329, 0.360, 0.391]),
    },
}

# Measured radius of maximum swirl velocity r1/R of the tip vortex, AR = 6 rotor (young vortex).
# Source: Tung, Pucci, Caradonna & Morse, NASA TM-81316, Fig. 15 (8 deg, age 50.5 deg,
# r1 = 0.0027R) and Fig. 16 (12 deg, age 52.2 deg, r1 = 0.0081R), PDF page 21.
# 5 deg was not measured on this rotor: 0.003R is the average over all blades tested
# (TM-81316, text on PDF page 9). Used only to regularise the filament (see NOTE.md).
R1_NUCLEO = {5: 0.003, 8: 0.0027, 12: 0.0081}

# Vortex strength used by the authors at 5 deg in their code (A = 0.81, Fig. 16),
# together with a 0.02R reduction of the wake contraction: it is a tuning value,
# not a direct measurement. Here it is used only as a sensitivity test.
K_VORTICE_5 = 0.81
