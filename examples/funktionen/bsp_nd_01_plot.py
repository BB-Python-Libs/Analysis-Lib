from analysis_lib.funktionen import AnalysisIIVisualisierung
import sympy as sp
import numpy as np

x, y = sp.symbols('x y')

# Beispiel: Ein Paraboloid
f_expr = sp.sqrt(1 - x**2 - y**2)

# Punkt, an dem wir die Ableitung betrachten
punkt = (1, 1)

'''
# 1. 3D-Ansicht (Schnitt parallel zur x-Achse)
AnalysisIIVisualisierung.plot_partial_derivative_3d(
    f_expr, x, y, point=punkt, wrt='x', 
    x_range=(-3, 3), y_range=(-3, 3),
    title="Partielle Ableitung nach x im 3D-Raum",
    save_fig="partielle_abl_3d"
)

# 2. 2D-Ansicht (Genau derselbe Schnitt, frontal betrachtet)
AnalysisIIVisualisierung.plot_partial_function_2d(
    f_expr, x, y, point=punkt, wrt='x', 
    plot_range=(-3, 3),
    title="Partielle Funktion als 2D-Profil",
    save_fig="partielle_abl_2d"
)
'''
# Wir schneiden bei y = 0 parallel zur x-Achse
AnalysisIIVisualisierung.plot_partial_function_3d(
    f_expr, x, y, point=(0, 0), wrt='x', 
    x_range=(-3, 3), y_range=(-3, 3),
    title="Schnitt durch die Fläche: $y=0$ wird festgehalten",
    save_fig="partielle_fkt_3d"
)

# Wir schneiden bei y = 0 parallel zur x-Achse
AnalysisIIVisualisierung.plot_partial_function_3d(
    f_expr, x, y, point=(0, 0), wrt='y', 
    x_range=(-3, 3), y_range=(-3, 3),
    title="Schnitt durch die Fläche: $x=0$ wird festgehalten",
    save_fig="partielle_fkt_3d_2"
)