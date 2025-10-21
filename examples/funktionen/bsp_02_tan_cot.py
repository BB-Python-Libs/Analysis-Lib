import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x, y = sp.symbols('x y')

print('--- Beispiel 3: Tangens-Funktion und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.tan(x), sp.atan(y), x_range=(-np.pi/2, np.pi/2), y_range=(-10,10), plot_range=(-5,5))

print('--- Beispiel 4: Cotangens-Funktion und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.cot(x), sp.pi/2 - sp.atan(y), x_range=(0, np.pi), y_range=(-10,10), plot_range=(-5,5))
