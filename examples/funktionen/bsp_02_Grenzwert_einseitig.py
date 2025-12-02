import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x = sp.Symbol('x')

# 1. Signum-Funktion (Klassischer Sprung)
# Links -1, Rechts +1
f_sign = sp.sign(x)
FunktionenBibliothek.plot_sided_limits(f_sign, x, x0=0, limit_left=-1, limit_right=1, epsilon=0.4, x_range=(-3, 3),
                  title=r"Links- und rechtsseitiger Grenzwert von $sgn(x)$", save_fig="bsp_02_signum_funktion")

# 2. Heaviside-ähnliche Funktion mit Polynom
# f(x) = x^2 für x<1, f(x) = 3-x für x>1
# Grenzwert links: 1, Grenzwert rechts: 2
f_piecewise = sp.Piecewise((x**2, x < 1), (3-x, x >= 1))
FunktionenBibliothek.plot_sided_limits(f_piecewise, x, x0=1, limit_left=1, limit_right=2, epsilon=0.3, x_range=(-1, 3),
                  title=r"Sprungstelle bei $x_0=1$", save_fig="bsp_02_piecewise_funktion")

# 3. Funktion 1/x (Nur rechtsseitig betrachten, z.B. Grenzwert ist unendlich - hier schwer darstellbar, 
# aber wir können einen endlichen Ausschnitt zeigen oder eine Funktion nehmen, die gegen eine Zahl geht)
# Beispiel: e^(-1/x)
# Für x->0+ geht es gegen 0. Für x->0- geht es gegen unendlich (lassen wir hier weg)
f_exp = sp.exp(-1/x)
FunktionenBibliothek.plot_sided_limits(f_exp, x, x0=0, limit_right=0, epsilon=0.15, x_range=(-1, 2),
                  title=r"Nur rechtsseitiger Grenzwert von $e^{-1/x}$ bei $x_0=0$", save_fig="bsp_02_exponential_funktion")
