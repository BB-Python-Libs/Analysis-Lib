import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x, y = sp.symbols('x y')

print('--- Beispiel 9: Exponentialfunktion und Logarithmus ---')
FunktionenBibliothek.plot_function_and_inverse(sp.exp(x), sp.log(y), x_range=(-5,5), y_range=(0.00001,5), plot_range=(-5,5))

print('--- Beispiel 10: Monome x^n und inverse Wurzelfunktionen ---')
for n_val in [2,3,4]:
    print(f'Monom a) f(x) = x**{n_val}')
    FunktionenBibliothek.plot_function_and_inverse(x**n_val, y**(1/n_val), x_range=(0,5), y_range=(0,25), plot_range=(0,6))
