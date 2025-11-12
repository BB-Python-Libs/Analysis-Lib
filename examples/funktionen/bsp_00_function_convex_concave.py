from analysis_lib.funktionen import FunktionenBibliothek
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


x = sp.Symbol('x')
expr = x**4 - 6*x**2 + 3
FunktionenBibliothek.plot_convex_concave(expr, x, x_range=(-3, 3), title="Konvexe und konkave Bereiche von $x^4 - 6x^2 + 3$")


# Definition der Funktion
def f(x):
    return x**3 - 3*x

# Plotbereich
x_vals = np.linspace(-2.5, 2.5, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$f(x)=x^3-3x$', color='blue', linewidth=2)

# Schar von Sekanten einzeichnen
x_points = np.linspace(-2.4, -0.1, 10)  # Sekanten-Stützpunkte
for x1 in x_points:
    for x2 in x_points:
        if abs(x1 - x2) > 1e-5:
            # Sekante zwischen x1, x2 berechnen
            y1, y2 = f(x1), f(x2)
            sec_x = np.array([x1, x2])
            sec_y = np.array([y1, y2])
            plt.plot(sec_x, sec_y, color='green', linewidth=1, alpha=0.3, zorder=1)

# Schar von Sekanten einzeichnen
x_points = np.linspace(0.1, 2.4, 10)  # Sekanten-Stützpunkte
for x1 in x_points:
    for x2 in x_points:
        if abs(x1 - x2) > 1e-5:
            # Sekante zwischen x1, x2 berechnen
            y1, y2 = f(x1), f(x2)
            sec_x = np.array([x1, x2])
            sec_y = np.array([y1, y2])
            plt.plot(sec_x, sec_y, color='red', linewidth=1, alpha=0.3, zorder=1)

plt.title(r'Schar von Sekanten für $f(x) = x^3 - 3x$')
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.grid(True, alpha=0.35)
plt.axhline(0, color='black', linewidth=0.7)
plt.axvline(0, color='black', linewidth=0.7)
plt.legend()
plt.tight_layout()
plt.show()
