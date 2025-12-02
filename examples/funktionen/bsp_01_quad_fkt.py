import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x, y = sp.symbols('x y')

print('--- Beispiel 1: $x^3$ und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(x**3, sp.root(y, 3), x_range=(0, 2), y_range=(0,4), plot_range=(0,4))