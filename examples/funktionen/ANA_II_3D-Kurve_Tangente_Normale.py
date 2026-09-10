import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def plot_curve_3d_with_vectors(gamma, t_symbol, t_range,
                               t0=None,
                               show_tangent=True,
                               show_normal=True,
                               vector_scale=1.0,
                               n_points=1000,
                               title="3D-Kurve mit Tangenten- und Normalenvektor",
                               save_fig=None):
    """
    Plottet eine parametrisierte 3D-Kurve gamma(t) = (x(t), y(t), z(t))
    sowie optional Tangenten- und Hauptnormalenvektor im Punkt t0.
    """

    x_expr, y_expr, z_expr = gamma

    # Erste und zweite Ableitungen
    dx_expr = sp.diff(x_expr, t_symbol)
    dy_expr = sp.diff(y_expr, t_symbol)
    dz_expr = sp.diff(z_expr, t_symbol)

    ddx_expr = sp.diff(dx_expr, t_symbol)
    ddy_expr = sp.diff(dy_expr, t_symbol)
    ddz_expr = sp.diff(dz_expr, t_symbol)

    # Numerische Funktionen
    x_fun = sp.lambdify(t_symbol, x_expr, "numpy")
    y_fun = sp.lambdify(t_symbol, y_expr, "numpy")
    z_fun = sp.lambdify(t_symbol, z_expr, "numpy")

    dx_fun = sp.lambdify(t_symbol, dx_expr, "numpy")
    dy_fun = sp.lambdify(t_symbol, dy_expr, "numpy")
    dz_fun = sp.lambdify(t_symbol, dz_expr, "numpy")

    ddx_fun = sp.lambdify(t_symbol, ddx_expr, "numpy")
    ddy_fun = sp.lambdify(t_symbol, ddy_expr, "numpy")
    ddz_fun = sp.lambdify(t_symbol, ddz_expr, "numpy")

    t_vals = np.linspace(t_range[0], t_range[1], n_points)
    x_vals = np.array(x_fun(t_vals), dtype=float)
    y_vals = np.array(y_fun(t_vals), dtype=float)
    z_vals = np.array(z_fun(t_vals), dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_vals, y_vals, z_vals, color='blue', linewidth=2, label='Kurve')

    if t0 is not None:
        x0 = float(x_fun(t0))
        y0 = float(y_fun(t0))
        z0 = float(z_fun(t0))

        v1 = np.array([
            float(dx_fun(t0)),
            float(dy_fun(t0)),
            float(dz_fun(t0))
        ], dtype=float)

        v2 = np.array([
            float(ddx_fun(t0)),
            float(ddy_fun(t0)),
            float(ddz_fun(t0))
        ], dtype=float)

        ax.scatter([x0], [y0], [z0], color='black', s=40, label=fr'Punkt bei $t_0={t0}$')

        # Tangentenvektor
        if show_tangent:
            norm_v1 = np.linalg.norm(v1)
            if norm_v1 > 1e-12:
                T = v1 / norm_v1
                ax.quiver(x0, y0, z0,
                      vector_scale * T[0], vector_scale * T[1], vector_scale * T[2],
                      color='red', arrow_length_ratio=0.15, linewidth=2,
                      label='Tangentenvektor')

        # Hauptnormalenvektor
        if show_normal:
            norm_v1 = np.linalg.norm(v1)
            if norm_v1 > 1e-12:
                T = v1 / norm_v1
                # Projektion von v2 auf Normalenanteil
                v2_normal = v2 - np.dot(v2, T) * T
                norm_v2_normal = np.linalg.norm(v2_normal)

                if norm_v2_normal > 1e-12:
                    N = v2_normal / norm_v2_normal
                    ax.quiver(x0, y0, z0,
                              vector_scale * N[0], vector_scale * N[1], vector_scale * N[2],
                              color='green', arrow_length_ratio=0.15, linewidth=2,
                              label='Normalenvektor')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title(title)
    ax.grid(True)

    # sinnvollere Box-Aspekte
    try:
        ax.set_box_aspect([
            np.ptp(x_vals) if np.ptp(x_vals) > 0 else 1,
            np.ptp(y_vals) if np.ptp(y_vals) > 0 else 1,
            np.ptp(z_vals) if np.ptp(z_vals) > 0 else 1
        ])
    except Exception:
        pass

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')
    plt.show()


t = sp.Symbol('t')
r = 6

plot_curve_3d_with_vectors(
    gamma=(r * sp.cos(t), r * sp.sin(t), t),
    t_symbol=t,
    t_range=(0, 2 * np.pi),
    t0=3,
    show_tangent=True,
    show_normal=True,
    vector_scale=0.8,
    title=r"3D-Kurve $\gamma(t)=(6\cos t, 6\sin t, t)$ mit Tangente und Normalenvektor"
)
