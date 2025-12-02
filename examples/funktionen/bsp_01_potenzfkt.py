import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x, y = sp.symbols('x y')

print('--- Beispiel 1: Potenz-Funktion und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(x**sp.pi, y**(1/sp.pi),
                                               x_range=(0, 4), y_range=(0,4),
                                               plot_range=(0,4), save_fig="Potenz_und_Umkehrfunktion")
