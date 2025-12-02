import sympy as sp
import numpy as np
from analysis_lib.funktionen import FunktionenBibliothek

x = sp.Symbol('x')

# Beispiel 1: Stetige Funktion x^2 an der Stelle x0=1
f_expr = x**2
FunktionenBibliothek.plot_epsilon_delta_auto(x**3, x, x0=1, epsilon=0.5, auto_delta=True, x_range=(0, 2), 
                        title_prefix="Automatische Delta-Bestimmung ($x^3$)")

# Beispiel 1: x^3 bei x0=1, Epsilon=0.5
# Hier berechnen wir Delta automatisch.
FunktionenBibliothek.plot_epsilon_delta_auto(x**3, x, x0=1, epsilon=0.5, auto_delta=True, x_range=(0, 2), 
                        title_prefix="Automatische Delta-Bestimmung ($x^3$)")

# Beispiel 2: Sinus Funktion
# Hier sieht man schön, wie das Delta durch die Krümmung limitiert wird.
FunktionenBibliothek.plot_epsilon_delta_auto(sp.sin(x), x, x0=0, epsilon=0.4, auto_delta=True, x_range=(-2, 2),
                        title_prefix="Automatische Delta-Bestimmung ($\sin(x)$)")

# Beispiel 3: Wurzel(x) bei x=4
# Hier ist die Funktion steiler links als rechts -> Delta wird durch die linke Seite limitiert.
FunktionenBibliothek.plot_epsilon_delta_auto(sp.sqrt(x), x, x0=4, epsilon=0.5, auto_delta=True, x_range=(0, 9),
                        title_prefix="Automatische Delta-Bestimmung ($\sqrt{x}$)")# Beispiel 3: Wurzel(x) bei x=4

# Hier ist die Funktion steiler links als rechts -> Delta wird durch die linke Seite limitiert.
FunktionenBibliothek.plot_epsilon_delta_auto(1/(1+x**2), x, x0=0, epsilon=0.1, auto_delta=True, x_range=(-2, 2),
                        title_prefix="Automatische Delta-Bestimmung ($\sqrt{x}$)")