from analysis_lib.funktionen import AnalysisIIVisualisierung
import sympy as sp
import numpy as np

x, y = sp.symbols('x y')

# Sattelpunkt visualisieren und als "sattelpunkt_3d.png" speichern
f_sattel = x**2 - y**2
AnalysisIIVisualisierung.plot_scalar_field_3d(
    f_sattel, x, y, 
    title="Sattelpunkt: $f(x,y) = x^2 - y^2$",
    save_fig="sattelpunkt_3d"
)

# Höhenlinien visualisieren und als "niveaumengen_gauss.png" speichern
f_gauss = sp.exp(-(x**2 + y**2))
AnalysisIIVisualisierung.plot_scalar_field_contour(
    f_gauss, x, y, x_range=(-3, 3), y_range=(-3, 3),
    title="Niveaumengen der Gauß-Glocke",
    save_fig="niveaumengen_gauss"
)