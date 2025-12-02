import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x, y = sp.symbols('x y')

print('--- Beispiel 1: Potenz-Funktion und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(2**x, sp.log(y, 2),
                                               x_range=(-4, 4), y_range=(0,4),
                                               plot_range=(-4,4), save_fig="Exponential_und_Umkehrfunktion")
