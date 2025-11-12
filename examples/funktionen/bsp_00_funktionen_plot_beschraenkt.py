from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')
funcs = [(sp.sin(x), r'$\sin x$'), (sp.cos(x)/2, r'$\frac{1}{2}\cos x$')]
FunktionenBibliothek.plot_boundedness(funcs, x, x_range=(-4*np.pi, 4*np.pi),
                                      upper_bound=1, lower_bound=-1,
                                      title="Beschränktheit mit konstanten Schranken")


upper = 1/2
lower = -1/2
funcs = [(sp.sin(x)/(1+x**2), r'$\frac{\sin x}{1+x^2}$')]
FunktionenBibliothek.plot_boundedness(funcs, x, x_range=(-6, 6),
                                      upper_bound=upper, lower_bound=lower,
                                      title="Beschränktheit mit x-abhängigen Schranken")

expr = sp.sin(x)/x
FunktionenBibliothek.plot_boundedness_with_tolerance(expr, x, x_range=(0.5, 30),
                                                     L=0.0, epsilon=0.1,
                                                     title="ε-Streifen um L=0 für sin(x)/x")