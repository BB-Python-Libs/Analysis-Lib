from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    ((x+1)/(x**2-1), '$\\frac{x+1}{x^2-1}$')
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-4, 4),
    y_range=(-4, 4),
    title="Hebbare Singularität bei $x=-1$"
)
