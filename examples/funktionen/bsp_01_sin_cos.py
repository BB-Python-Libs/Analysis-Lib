import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x, y = sp.symbols('x y')

print('--- Beispiel 1: Sinus-Funktion und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.sin(x), sp.asin(y), x_range=(-np.pi/2, np.pi/2), y_range=(-1,1), plot_range=(-2,2))

print('--- Beispiel 2: Kosinus-Funktion und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.cos(x), sp.acos(y), x_range=(0, np.pi), y_range=(-1,1), plot_range=(-1, np.pi))
