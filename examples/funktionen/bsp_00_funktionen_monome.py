from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (x, '$x$'),
    (x**2, '$x^2$'),
    (x**3, '$x^3$'),
    (x**4, '$x^4$'),    (x**5, '$x^5$'), (x**6, '$x^6$'), (x**7, '$x^7$'), (x**8, '$x^8$')
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-2, 2),
    y_range=(-8, 8),
    title="Vergleich: Monome Funktionen bis zum Grad 8"
)
