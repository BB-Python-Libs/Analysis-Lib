import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x = sp.Symbol('x')

# 1. Gedämpfte Schwingung gegen 0 für x -> +unendlich
# Hier sieht man schön, wie M immer weiter nach rechts wandert, wenn man Epsilon verkleinert.
f1 = 5 * sp.sin(x) / x
FunktionenBibliothek.plot_limit_at_infinity(f1, x, limit_val=0, epsilon=0.01, direction='plus', x_range=(100, 600),
                       title=r"Konvergenz von $\frac{5 \sin(x)}{x}$ gegen 0 ($x \to \infty$)", save_fig="bsp_02_limit_01")

FunktionenBibliothek.plot_limit_at_infinity(f1, x, limit_val=0, epsilon=0.03, direction='plus', x_range=(100, 600),
                       title=r"Konvergenz von $\frac{5 \sin(x)}{x}$ gegen 0 ($x \to \infty$)", save_fig="bsp_02_limit_02")

# 2. Rationale Funktion gegen 2 für x -> +unendlich
# (2x^2 - 1) / (x^2 + 1) konvergiert gegen 2
f2 = (2*x**2 - 1) / (x**2 + 1)
FunktionenBibliothek.plot_limit_at_infinity(f2, x, limit_val=2, epsilon=0.1, direction='plus', x_range=(0, 20),
                       title=r"Konvergenz von $\frac{2x^2-1}{x^2+1}$ gegen 2 ($x \to \infty$)", save_fig="bsp_02_limit_03")

# 3. e-Funktion gegen 0 für x -> -unendlich
f3 = sp.exp(x)
FunktionenBibliothek.plot_limit_at_infinity(f3, x, limit_val=0, epsilon=0.2, direction='minus', x_range=(-5, 2),
                       title=r"Konvergenz von $e^x$ gegen 0 ($x \to -\infty$)", save_fig="bsp_02_limit_04")