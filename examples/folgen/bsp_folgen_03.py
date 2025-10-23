import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from analysis_lib.folgen import FolgenBibliothek 

n = sp.Symbol('n')
folge = 3 * sp.sin(n) / n

FolgenBibliothek.plot_supremum_infimum_visualization(
    folge, n, n_start=1, n_end=100,
    epsilon=0.05,
    title="Supremum, Infimum und Epsilon-Streifen für $a_n = \\frac{3 \\sin(n)}{n}$"
)