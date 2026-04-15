from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (x + sp.exp(-x**2)*x**2, 'Funktion: $x + e^(-x^2)x^2$'),
    (x, 'Lineare Approximation: $x$'),
    (x + x**2, 'Quadratische Approximation: $x + x^2$'),
    (x + x**2 - x**4, 'Approximation der Ordnung 4: $x + x^2 - x^4$'),
    (x + x**2 - x**4 + 1/2*x**6, 'Approximation der Ordnung 6: $x + x^2 - x^4 + \\frac{1}{2}x^6$'),
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-1.5, 1.5),
    y_range=(-3,3),
    title="Approximation durch Taylor-Polynom $x_0=0$",
    save_fig="bsp_Taylor_02" 
)

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (x + sp.exp(-x**2)*x**2, 'Funktion: $x + e^(-x^2)x^2$'),
    (x, 'Lineare Approximation: $x$'),
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-1.5, 1.5),
    y_range=(-3,3),
    title="Approximation durch Taylor-Polynom $x_0=0$",
    save_fig="bsp_Taylor_01" 
)
