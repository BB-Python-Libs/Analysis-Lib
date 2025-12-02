from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (sp.sinh(x), '$\\sinh(x)$'),
    (sp.cosh(x), '$\\cosh(x)$'),
    (sp.tanh(x), '$\\tanh(x)$'),
    (sp.coth(x), '$\\coth(x)$')
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-6, 6),
    y_range=(-6, 6),
    title="Hyperbolische Funktionen im Vergleich",
    save_fig="HyperbolicFunktionen"
)
