"""
Shared figure style: each model has the same color in every figure of the project.

Colors: first three slots of a colorblind-validated categorical palette (blue, orange,
aqua: distinguishable in pairs also for deuteranopia/protanopia/tritanopia).
They are assigned to the models, never to the plotting order:
    Prandtl  -> blue,  V1 2D vortex -> orange,  E2 helix minus azimuthal mean -> aqua.
References that are not models under comparison use ink/gray:
    experiment -> black markers, BEMT without tip loss -> gray dashed,
    numerical studies (grids, wake lengths) -> ink and a gray ramp.
Every figure with two or more series has a legend, so color is never the only cue.
"""
import matplotlib as mpl

# Models (identity colors)
COLORE = {
    "prandtl": "#2a78d6",          # BEMT with Prandtl tip loss
    "v1": "#eb6834",               # 2D straight vortex (v1)
    "e2": "#1baf7a",               # helix minus azimuthal mean (v2)
    "e1": "#eda100",               # full helix (v2), reserved: no physical solution
    "senza_perdite": "#898781",    # BEMT without tip loss (reference)
    "esperimento": "#0b0b0b",      # measured data
}
NOME = {
    "prandtl": "BEMT + Prandtl",
    "v1": "V1: 2D vortex",
    "e2": "V2: helix − azimuthal mean",
    "e1": "V2: full helix",
    "senza_perdite": "BEMT, no tip loss",
    "esperimento": "Experiment (NASA TM-81232)",
}

# Ink and chrome
INCHIOSTRO = "#0b0b0b"
INCHIOSTRO_2 = "#52514e"
GRIGIO_ASSI = "#898781"
GRIGLIA = "#e1e0d9"
SFONDO = "#fcfcfb"

# Ordered gray ramp, light to dark, for ordinal numerical parameters (e.g. wake length)
RAMPA_GRIGI = ["#c3c2b7", "#a3a29a", "#898781", "#52514e", "#0b0b0b"]

SPESSORE = 1.6        # line width [pt]
MARKER = 6.5          # marker size [pt]


def applica_stile():
    """Set the matplotlib defaults shared by all project figures."""
    mpl.rcParams.update({
        "figure.facecolor": SFONDO,
        "axes.facecolor": SFONDO,
        "savefig.facecolor": SFONDO,
        "axes.edgecolor": GRIGIO_ASSI,
        "axes.labelcolor": INCHIOSTRO,
        "axes.titlecolor": INCHIOSTRO,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIGLIA,
        "grid.linewidth": 0.8,
        "xtick.color": INCHIOSTRO_2,
        "ytick.color": INCHIOSTRO_2,
        "text.color": INCHIOSTRO,
        "legend.frameon": False,
        "lines.linewidth": SPESSORE,
        "lines.markersize": MARKER,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def stile_modello(nome):
    """Keyword arguments for plotting a model curve (color and label)."""
    tratto = "--" if nome == "senza_perdite" else "-"
    return {"color": COLORE[nome], "ls": tratto, "label": NOME[nome]}


def stile_esperimento():
    """Keyword arguments for plotting the measured points."""
    return {"color": COLORE["esperimento"], "marker": "o", "ls": "none",
            "markersize": MARKER, "label": NOME["esperimento"], "zorder": 5}
