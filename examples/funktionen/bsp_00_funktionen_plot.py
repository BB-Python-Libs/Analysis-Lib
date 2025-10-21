from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (x, 'Lineare Funktion: $x$'),
    (x**2, 'Quadratische Funktion: $x^2$'),
    (x**3, 'Kubische Funktion: $x^3$'),
    (sp.sin(x), 'Sinus-Funktion: $\\sin(x)$')
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-5, 5),
    y_range=(-2,2),
    title="Vergleich: Quadratisch, Kubisch und Trigonometrisch"
)
