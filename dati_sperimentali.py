"""
Experimental data from Caradonna & Tung (NASA TM-81232, 1981), 1250 rpm, M_tip = 0.439.

All Cl values and vortex positions were read by eye from the figures of the report
(uncertainty about +-0.005), not from numerical tables.
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

# Vortex strength used by the authors at 5 deg in their code (A = 0.81, Fig. 16),
# together with a 0.02R reduction of the wake contraction: it is a tuning value,
# not a direct measurement. Here it is used only as a sensitivity test.
K_VORTICE_5 = 0.81
