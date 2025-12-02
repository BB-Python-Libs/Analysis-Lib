import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x, y = sp.symbols('x y')

print('--- Beispiel 1: Hyperbolische Funktion sinh und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.sinh(x), sp.asinh(y),
                                               x_range=(-4, 4), y_range=(-4,4),
                                               plot_range=(-4,4), save_fig="Sinh_und_Umkehrfunktion")

print('--- Beispiel 1: Hyperbolische Funktion cosh und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.cosh(x), sp.acosh(y),
                                               x_range=(0, 4), y_range=(0,4),
                                               plot_range=(0,4), save_fig="Cosh_und_Umkehrfunktion")

print('--- Beispiel 1: Hyperbolische Funktion tanh und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.tanh(x), sp.atanh(y),
                                               x_range=(-4, 4), y_range=(-4,4),
                                               plot_range=(-4,4), save_fig="Tanh_und_Umkehrfunktion")

print('--- Beispiel 1: Hyperbolische Funktion coth und ihre Umkehrfunktion ---')
FunktionenBibliothek.plot_function_and_inverse(sp.coth(x), sp.acoth(y),
                                               x_range=(-4, 4), y_range=(-4,4),
                                               plot_range=(-4,4), save_fig="Coth_und_Umkehrfunktion")
