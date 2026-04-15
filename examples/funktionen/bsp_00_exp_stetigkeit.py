from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (abs(sp.exp(x) - 1), 'Exponential-Funktion: $|e^x - 1|$'),
    (2*abs(x), 'Betrags-Funktion: $2|x|$')
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-1, 1),
    y_range=(-1.5,1.5),
    title="Vergleich der Beträge",
    save_fig="bsp_00_exp_stetigkeit"
)

# Definiere die Funktionen zum Vergleich
functions_to_plot = [
    (abs(sp.sin(x)), 'Sinus-Funktion: $|sin(x)|$'),
    (abs(x), 'Betrags-Funktion: $|x|$')
]

# Rufe die Plot-Funktion auf
FunktionenBibliothek.plot_multiple_functions(
    functions=functions_to_plot,
    x_symbol=x,
    x_range=(-.25, .25),
    y_range=(0,0.5),
    title="Vergleich der Beträge",
    save_fig="bsp_00_sin_stetigkeit"
)