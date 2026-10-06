"""
Dati sperimentali di Caradonna & Tung (NASA TM-81232, 1981), 1250 rpm, M_tip = 0.439.

Tutti i valori di Cl e di posizione del vortice sono letti a occhio dalle figure
del report (incertezza circa +-0.005), non da tabelle numeriche.
"""
import numpy as np

# Stazioni radiali in cui sono misurate le pressioni (r/R)
X_STAZIONI = np.array([0.50, 0.68, 0.80, 0.89, 0.96])

# Per ogni angolo di passo [gradi]:
#   CT   coefficiente di spinta misurato
#   Cl   Cl di sezione nelle 5 stazioni
#   r_v  posizione radiale del vortice della pala precedente (età 180°), fig. 10
#   z_v  distanza del vortice sotto il piano del rotore (età 180°), fig. 10
CASI = {
    5: {
        "CT": 0.0021,                                             # fig. 16
        "Cl": np.array([0.103, 0.123, 0.118, 0.150, 0.153]),      # fig. 16
        "r_v": 0.915,
        "z_v": 0.05,
    },
    8: {
        "CT": 0.00459,                                            # fig. 4
        "Cl": np.array([0.236, 0.284, 0.291, 0.319, 0.274]),      # fig. 14
        "r_v": 0.905,
        "z_v": 0.09,
    },
    12: {
        "CT": 0.0079,                                             # fig. 15
        "Cl": np.array([0.425, 0.492, 0.500, 0.510, 0.422]),      # fig. 15
        "r_v": 0.885,
        "z_v": 0.11,
    },
}

# Intensità del vortice usata dagli autori a 5° nel loro codice (A = 0.81, fig. 16),
# insieme a una riduzione della contrazione di 0.02R: è un valore di taratura,
# non una misura diretta. Qui serve solo come test di sensibilità.
K_VORTICE_5 = 0.81
