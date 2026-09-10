import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def taylor_polynomial_2d(f_expr, x_symbol, y_symbol, a=0, b=0, degree=3):
    """
    Berechnet das Taylorpolynom von f(x,y) um (a,b) bis zum Gesamtrad 'degree'.
    """
    t = sp.Symbol('t')

    # Verschiebung auf den Entwicklungspunkt und Skalierung mit t
    shifted = f_expr.subs({
        x_symbol: a + t * (x_symbol - a),
        y_symbol: b + t * (y_symbol - b)
    })

    series_t = sp.series(shifted, t, 0, degree + 1).removeO()
    T = sp.expand(series_t.subs(t, 1))

    return sp.expand(T)


def plot_taylor_surface_deg3(f_expr, x_symbol, y_symbol,
                             a=0, b=0, degree=3,
                             x_range=(-2, 2), y_range=(-2, 2),
                             resolution=150,
                             title="Taylorpolynom einer Funktion von zwei Variablen",
                             save_fig=None):
    """
    Visualisiert eine Funktion f(x,y) und ihr Taylorpolynom bis Grad 3
    um den Entwicklungspunkt (a,b) in einem gemeinsamen 3D-Plot.
    """
    if degree < 0 or degree > 3:
        raise ValueError("Bitte degree zwischen 0 und 3 wählen.")

    # Taylorpolynom berechnen
    T_expr = taylor_polynomial_2d(f_expr, x_symbol, y_symbol, a=a, b=b, degree=degree)

    # Numerische Funktionen
    f_num = sp.lambdify((x_symbol, y_symbol), f_expr, "numpy")
    T_num = sp.lambdify((x_symbol, y_symbol), T_expr, "numpy")

    # Gitter
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    with np.errstate(all='ignore'):
        Z_f = np.array(f_num(X, Y), dtype=float)
        Z_T = np.array(T_num(X, Y), dtype=float)

    # Entwicklungspunkt
    z0 = float(sp.N(f_expr.subs({x_symbol: a, y_symbol: b})))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Originalfunktion
    surf1 = ax.plot_surface(
        X, Y, Z_f,
        cmap='viridis',
        alpha=0.75,
        edgecolor='none'
    )

    # Taylorpolynom
    surf2 = ax.plot_surface(
        X, Y, Z_T,
        cmap='autumn',
        alpha=0.45,
        edgecolor='none'
    )

    # Entwicklungspunkt markieren
    ax.scatter([a], [b], [z0], color='black', s=60, zorder=5)

    ax.set_xlabel(f"${x_symbol}$")
    ax.set_ylabel(f"${y_symbol}$")
    ax.set_zlabel(r"$z$")
    ax.set_title(
        title + "\n"
        + fr"$f(x,y)={sp.latex(f_expr)}$, "
        + fr"$T_{{{degree}}}(x,y)={sp.latex(T_expr)}$"
    )

    # Zwei kleine Farbskalen wären zu viel; daher nur Legendentext manuell
    proxy1 = plt.Line2D([0], [0], linestyle="none", marker='s', markersize=10,
                        markerfacecolor='teal', alpha=0.8, label='Originalfunktion')
    proxy2 = plt.Line2D([0], [0], linestyle="none", marker='s', markersize=10,
                        markerfacecolor='orange', alpha=0.6, label=f'Taylorpolynom Grad {degree}')
    proxy3 = plt.Line2D([0], [0], linestyle="none", marker='o', markersize=8,
                        markerfacecolor='black', label='Entwicklungspunkt')
    ax.legend(handles=[proxy1, proxy2, proxy3], loc='upper left')

    plt.tight_layout()

    if save_fig is not None:
        plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')

    plt.show()

    return T_expr

x, y = sp.symbols('x y')

f_expr = x * y * sp.sin(x + y)

T3 = plot_taylor_surface_deg3(
    f_expr, x, y,
    a=1, b=1,
    degree=3,
    x_range=(-1, 3),
    y_range=(-1, 3),
    title=r"Taylorpolynom bis Grad 3 für $f(x,y)=xy\sin(x+y)$",
    save_fig="taylor_2d_xy_sin_4"
)

print("Taylorpolynom 3. Grades:")
print(T3)
