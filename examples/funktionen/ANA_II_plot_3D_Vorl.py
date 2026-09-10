import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def plot_scalar_field_3d(f_expr, x_symbol, y_symbol, x_range=(-5, 5), y_range=(-5, 5),
                         resolution=100, title="3D-Skalarfeld"):
    """
    Visualisiert ein 3D-Skalarfeld f(x, y) als Oberfläche inkl. projizierter Höhenlinien.
    Es werden nur Punkte geplottet, an denen die Funktion definiert und endlich ist.
    """
    f = sp.lambdify((x_symbol, y_symbol), f_expr, "numpy")

    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Numerische Auswertung mit Warnungsunterdrückung
    with np.errstate(all='ignore'):
        Z = f(X, Y)

    # In Array umwandeln und undefinierte / nichtendliche Werte maskieren
    Z = np.array(Z, dtype=float)
    Z_masked = np.ma.masked_invalid(Z)

    # Prüfen, ob überhaupt definierte Werte vorhanden sind
    if Z_masked.count() == 0:
        raise ValueError("Im gewählten Bereich gibt es keine definierten Funktionswerte.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Oberfläche nur auf definierten Punkten
    surf = ax.plot_surface(X, Y, Z_masked, cmap='viridis', edgecolor='none', alpha=0.85)

    # z-Offset für die projizierten Höhenlinien
    z_min = Z_masked.min()
    z_max = Z_masked.max()
    z_offset = z_min - (z_max - z_min) * 0.15 if z_max > z_min else z_min - 1

    # Höhenlinien nur dann zeichnen, wenn genug definierte Werte vorhanden sind
    try:
        ax.contour(X, Y, Z_masked, zdir='z', offset=z_offset, cmap='viridis', levels=15)
    except Exception:
        pass

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='$f(x, y)$')
    ax.set_title(title)
    ax.set_xlabel(f"${x_symbol.name}$")
    ax.set_ylabel(f"${y_symbol.name}$")
    ax.set_zlabel(r"$f(x, y)$")
    ax.set_zlim(z_offset, z_max)

    plt.tight_layout()
    plt.show()

def plot_scalar_field_contour(f_expr, x_symbol, y_symbol, x_range=(-5, 5), y_range=(-5, 5),
                              resolution=400, title="Höhenlinien (Niveaumengen)"):
    """
    Zeichnet ein 2D-Höhenliniendiagramm des Skalarfeldes.
    Es werden nur Punkte geplottet, an denen die Funktion definiert und endlich ist.
    """
    f = sp.lambdify((x_symbol, y_symbol), f_expr, "numpy")

    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    with np.errstate(all='ignore'):
        Z = f(X, Y)

    Z = np.array(Z, dtype=float)
    Z_masked = np.ma.masked_invalid(Z)

    if Z_masked.count() == 0:
        raise ValueError("Im gewählten Bereich gibt es keine definierten Funktionswerte.")

    fig, ax = plt.subplots(figsize=(8, 6))

    cf = ax.contourf(X, Y, Z_masked, levels=20, cmap='viridis', alpha=0.8)
    c = ax.contour(X, Y, Z_masked, levels=20, colors='black', linewidths=0.6)
    ax.clabel(c, inline=True, fontsize=9, fmt='%.1f')

    fig.colorbar(cf, ax=ax, label='$f(x, y)$')

    ax.set_title(title)
    ax.set_xlabel(f"${x_symbol.name}$")
    ax.set_ylabel(f"${y_symbol.name}$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

x, y = sp.symbols('x y')

# Vorlesung :
fexpr = x**2 + 2*x*y + y**2
plot_scalar_field_contour(
    fexpr, x, y, x_range=(-1.1, 1.1), y_range=(-1.1, 1.1), resolution=1600,
    title="Niveaumengen der Halbkugel: $f(x,y) = x^2 + 2xy + y^2$"
)

plot_scalar_field_3d(
    fexpr, x, y, x_range=(-1.1, 1.1), y_range=(-1.1, 1.1), resolution=800,
    title="Funktionsplot der Halbkugel: $f(x,y) = x^2 + 2xy + y^2$"
)


fexpr = 2*x**4 + y**4 - x**2 - 2*y**2
plot_scalar_field_contour(
    fexpr, x, y, x_range=(-1.1, 1.1), y_range=(-1.1, 1.1), resolution=1600,
    title="Niveaumengen der Halbkugel: $f(x,y) = 2x^4 + y^4 - x^2 - 2y^2$"
)

plot_scalar_field_3d(
    fexpr, x, y, x_range=(-1.1, 1.1), y_range=(-1.1, 1.1), resolution=800,
    title="Funktionsplot der Halbkugel: $f(x,y) = 2x^4 + y^4 - x^2 - 2y^2$"
)

