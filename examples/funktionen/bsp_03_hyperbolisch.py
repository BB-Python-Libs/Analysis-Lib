import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x, y = sp.symbols('x y')

print('--- Beispiel 5: Hyperbolische Sinus-Funktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.sinh(x), sp.asinh(y), x_range=(-3,3), y_range=(-10,10), plot_range=(-5,5))

print('--- Beispiel 6: Hyperbolische Kosinus-Funktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.cosh(x), sp.acosh(y), x_range=(0.000001,5), y_range=(1,10), plot_range=(0,8))

print('--- Beispiel 7: Hyperbolische Tangens-Funktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.tanh(x), sp.atanh(y), x_range=(-3,3), y_range=(-0.99999,0.99999), plot_range=(-2,2))

print('--- Beispiel 8: Hyperbolische Cotangens-Funktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.coth(x), sp.acoth(y), x_range=(0.000001,5), y_range=(1.0000001,5), plot_range=(0,5))
