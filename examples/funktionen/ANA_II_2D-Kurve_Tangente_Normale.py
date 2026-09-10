import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def plot_curve_2d_with_vectors(gamma, t_symbol, t_range,
                               t0=None,
                               show_tangent=True,
                               show_normal=True,
                               vector_scale=1.0,
                               n_points=800,
                               title="2D-Kurve mit Tangenten- und Normalenvektor",
                               save_fig=None):
    """
    Plottet eine parametrisierte 2D-Kurve gamma(t) = (x(t), y(t))
    sowie optional Tangenten- und Normalenvektor im Punkt t0.
    """

    x_expr, y_expr = gamma

    # Ableitungen
    dx_expr = sp.diff(x_expr, t_symbol)
    dy_expr = sp.diff(y_expr, t_symbol)

    # Numerische Funktionen
    x_fun = sp.lambdify(t_symbol, x_expr, "numpy")
    y_fun = sp.lambdify(t_symbol, y_expr, "numpy")
    dx_fun = sp.lambdify(t_symbol, dx_expr, "numpy")
    dy_fun = sp.lambdify(t_symbol, dy_expr, "numpy")

    t_vals = np.linspace(t_range[0], t_range[1], n_points)
    x_vals = np.array(x_fun(t_vals), dtype=float)
    y_vals = np.array(y_fun(t_vals), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_vals, y_vals, color='blue', linewidth=2, label='Kurve')

    if t0 is not None:
        x0 = float(x_fun(t0))
        y0 = float(y_fun(t0))
        dx0 = float(dx_fun(t0))
        dy0 = float(dy_fun(t0))

        ax.scatter([x0], [y0], color='black', s=40, zorder=5, label=fr'Punkt bei $t_0={t0}$')

        # Tangentenvektor
        if show_tangent:
            ax.arrow(x0, y0,
                     vector_scale * dx0, vector_scale * dy0,
                     color='red', width=0.01, head_width=0.08,
                     length_includes_head=True, zorder=6,
                     label='Tangentenvektor')

        # Ein Normalenvektor in 2D: (-y', x')
        if show_normal:
            nx0 = -dy0
            ny0 = dx0
            ax.arrow(x0, y0,
                     vector_scale * nx0, vector_scale * ny0,
                     color='green', width=0.01, head_width=0.08,
                     length_includes_head=True, zorder=6,
                     label='Normalenvektor')

    ax.axhline(0, color='black', linewidth=0.7)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Doppelte Legendeneinträge vermeiden
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')
    plt.show()

t = sp.Symbol('t')

# Beispiel: gamma(t) = (t, t^2)
plot_curve_2d_with_vectors(
    gamma=(t, t**2),
    t_symbol=t,
    t_range=(-1, 2),
    t0=1,
    show_tangent=True,
    show_normal=True,
    vector_scale=0.4,
    title=r"2D-Kurve $\gamma(t)=(t,t^2)$ mit Tangente und Normalenvektor"
)

# Beispiel: gamma(t) = (t, t^2)
plot_curve_2d_with_vectors(
    gamma=((1+sp.cos(t))*sp.cos(t), (1+sp.cos(t))*sp.sin(t)),
    t_symbol=t,
    t_range=(0, 2*3.14159),
    t0=1,
    show_tangent=True,
    show_normal=True,
    vector_scale=0.4,
    title=r"2D-Kurve der Kardioide mit Tangente und Normalenvektor"
)