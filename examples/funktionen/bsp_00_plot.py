from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (sp.cos(1/x), 'Cosinus-Funktion: $\\cos(1/x)$')
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-1, 1),
    y_range=(-1.5,1.5),
    title="Keine Konvergenz gegeben $x=0$",
    save_fig="bsp_00_funktionen_cos_1_over_x"
)

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (sp.tan(x), 'Tangens-Funktion: $\\tan(x)$')
]

x_links = float(sp.pi/2-0.0001)
# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-1, x_links),
    y_range=(-100, 100),
    title="Linksseitiger Grenzwert bei $x=\\pi/2$",
    save_fig="bsp_02_tan_01_funktion"
)

x_rechts = float(sp.pi/2+0.0001)
# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(x_rechts, 3),
    y_range=(-100, 100),
    title="Rechtsseitiger Grenzwert bei $x=\\pi/2$",
    save_fig="bsp_02_tan_02_funktion"
)