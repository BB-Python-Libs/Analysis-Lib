import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def plot_implicit_with_tangent(F_expr, x_symbol, y_symbol,
                               point,
                               x_range=(-2, 2), y_range=(-2, 2),
                               resolution=800,
                               curve_color='darkred',
                               tangent_color='green',
                               title="Implizite Kurve mit Tangente",
                               save_fig=None):
    """
    Zeichnet die implizite Kurve F(x,y)=0 und die Tangente im Punkt point=(x0,y0).

    Parameter
    ---------
    F_expr : sympy expression
        Implizite Gleichung F(x,y), geplottet wird die Niveaulinie F(x,y)=0.
    x_symbol, y_symbol : sympy symbols
        Variablen.
    point : tuple
        Punkt (x0, y0) auf der Kurve.
    x_range, y_range : tuple
        Plotbereiche.
    resolution : int
        Anzahl der Gitterpunkte pro Richtung.
    save_fig : str oder None
        Falls gesetzt, wird als PNG gespeichert.
    """
    x0, y0 = point

    # Symbolische partielle Ableitungen
    Fx_expr = sp.diff(F_expr, x_symbol)
    Fy_expr = sp.diff(F_expr, y_symbol)

    # Numerische Funktionen
    F = sp.lambdify((x_symbol, y_symbol), F_expr, "numpy")
    Fx = sp.lambdify((x_symbol, y_symbol), Fx_expr, "numpy")
    Fy = sp.lambdify((x_symbol, y_symbol), Fy_expr, "numpy")

    # Prüfen, ob der Punkt auf der Kurve liegt
    F_val = float(sp.N(F_expr.subs({x_symbol: x0, y_symbol: y0})))
    if abs(F_val) > 1e-6:
        raise ValueError(f"Der Punkt ({x0}, {y0}) liegt nicht auf der Kurve F(x,y)=0. F(x0,y0)={F_val}")

    # Steigung der Tangente
    Fx0 = float(Fx(x0, y0))
    Fy0 = float(Fy(x0, y0))

    # Plotgitter
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    with np.errstate(all='ignore'):
        Z = F(X, Y)

    Z = np.array(Z, dtype=float)
    Z_masked = np.ma.masked_invalid(Z)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Implizite Kurve F(x,y)=0
    ax.contour(X, Y, Z_masked, levels=[0], colors=[curve_color], linewidths=1.5)

    # Punkt markieren
    ax.scatter([x0], [y0], color='black', s=40, zorder=5)

    # Tangente oder vertikale Tangente
    if abs(Fy0) > 1e-10:
        m = -Fx0 / Fy0
        tangent_y = y0 + m * (x_vals - x0)
        ax.plot(x_vals, tangent_y, color=tangent_color, linewidth=1.5,
                label=fr"Tangente: $y={y0:.3f}+({m:.3f})(x-{x0:.3f})$")
    elif abs(Fx0) > 1e-10:
        ax.axvline(x0, color=tangent_color, linewidth=1.5,
                   label=fr"Tangente: $x={x0:.3f}$")
    else:
        raise ValueError("In diesem Punkt verschwinden sowohl F_x als auch F_y; Tangente nicht eindeutig bestimmbar.")

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(f"${x_symbol}$")
    ax.set_ylabel(f"${y_symbol}$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')
    plt.show()

x, y = sp.symbols('x y')
F_expr = y**2 +  sp.sin(y*sp.pi) + x**3 - sp.cos(x*sp.pi)
plot_implicit_with_tangent(
    F_expr, x, y,
    point=(-1, 0),
    x_range=(-2, 2),
    y_range=(-2, 2),
    title=r"Implizite Kurve $F(x,y)=0$ mit Tangente",
    save_fig="ANA_II_implicit_Derivative_example1"
)

plot_implicit_with_tangent(
    F_expr, x, y,
    point=(0, 1),
    x_range=(-2, 2),
    y_range=(-2, 2),
    title=r"Implizite Kurve $F(x,y)=0$ mit Tangente",
    save_fig="ANA_II_implicit_Derivative_example2"
)

plot_implicit_with_tangent(
    F_expr, x, y,
    point=(0, -1),
    x_range=(-2, 2),
    y_range=(-2, 2),
    title=r"Implizite Kurve $F(x,y)=0$ mit Tangente",
    save_fig="ANA_II_implicit_Derivative_example3"
)