from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (2**x, '$2^x$'),
    (3**x, '$3^x$'),
    (4**x, '$4^x$')
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-4, 4),
    y_range=(0, 10),
    title="Verschiedene Exponentialfunktionen im Vergleich",
    save_fig="Exponentialfunktionen"
)
