from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')
expr = sp.sin(x) + x/10
FunktionenBibliothek.plot_with_extrema(expr, x, x_range=(-10, 10),
                                       title="Lokale/Globale Extrema für $\\sin x + x/10$")

upper = 1/2
lower = -1/2
expr = sp.sin(x)/(1+x**2)
FunktionenBibliothek.plot_with_extrema(expr, x, x_range=(-20, 20),
                                       title="Lokale/Globale Extrema für $\\frac{\\sin x}{1+x^2}$")
