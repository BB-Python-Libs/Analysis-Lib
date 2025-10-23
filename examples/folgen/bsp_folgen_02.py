import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from analysis_lib.folgen import FolgenBibliothek 

n = sp.Symbol('n')

# Definition der drei Folgen
untere_folge = -1/n
obere_folge = 1/n
mittlere_folge = sp.sin(n)/n

# Aufruf der Plot-Funktion mit detaillierten, flexiblen Labels
FolgenBibliothek.plot_squeeze_theorem(
    lower_seq=untere_folge,
    middle_seq=mittlere_folge,
    upper_seq=obere_folge,
    n_symbol=n,
    n_range=(1, 100),
    title="Einschließungskriterium für $c_n = \\frac{\\sin(n)}{n}$",
    lower_label=r'$a_n = -\frac{1}{n}$',  # LaTeX-Label
    middle_label=r'$c_n = \frac{\sin(n)}{n}$', # LaTeX-Label
    upper_label=r'$b_n = \frac{1}{n}$'   # LaTeX-Label
)


# Definition der drei Folgen
untere_folge = 0
obere_folge = 3/(3+n)
mittlere_folge = (3/4)**n

# Aufruf der Plot-Funktion mit detaillierten, flexiblen Labels
FolgenBibliothek.plot_squeeze_theorem(
    lower_seq=untere_folge,
    middle_seq=mittlere_folge,
    upper_seq=obere_folge,
    n_symbol=n,
    n_range=(1, 100),
    title="Einschließungskriterium für $c_n = \\left(\\frac{3}{4}\\right)^n$",
    lower_label=r'$a_n = 0$',  # LaTeX-Label
    middle_label=r'$c_n = \left(\frac{3}{4}\right)^n$', # LaTeX-Label
    upper_label=r'$b_n = \frac{1}{1 + \frac{n}{3}}$'   # LaTeX-Label
)

# Definition der drei Folgen
untere_folge = (1+1/n)**n
obere_folge = (1+1/n)**(n+1)
mittlere_folge = sp.E

# Aufruf der Plot-Funktion mit detaillierten, flexiblen Labels
FolgenBibliothek.plot_squeeze_theorem(
    lower_seq=untere_folge,
    middle_seq=mittlere_folge,
    upper_seq=obere_folge,
    n_symbol=n,
    n_range=(1, 100),
    title="Einschließungskriterium für $c_n = e = 2.7182818$",
    lower_label=r'$a_n = \left(1 + \frac{1}{n}\right)^n$',  # LaTeX-Label
    middle_label=r'$c_n = E = 2.7182818$', # LaTeX-Label
    upper_label=r'$b_n = \left(1 + \frac{1}{n}\right)^{n+1}$'   # LaTeX-Label
)
