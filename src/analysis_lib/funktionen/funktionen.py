import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class FunktionenBibliothek:
    @staticmethod
    def plot_function_and_inverse(f_expr, f_inv_expr, x_range=(-10, 10), y_range=(-10, 10), plot_range=(-10, 10), save_fig=None):
        """
        Plottet eine gegebene Funktion f(x) und ihre Umkehrfunktion f⁻¹(x)
        im selben Diagramm, ergänzt durch die Winkelhalbierende y=x.
        """
        x, y = sp.symbols('x y')
        f_lambda = sp.lambdify(x, f_expr, 'numpy')
        f_inv_lambda = sp.lambdify(y, f_inv_expr, 'numpy')

        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = np.linspace(y_range[0], y_range[1], 5000)
        diag_vals = np.linspace(plot_range[0], plot_range[1], 2)

        y_vals_f = f_lambda(x_vals)
        x_vals_f_inv = f_inv_lambda(y_vals)

        plt.figure(figsize=(8, 8))
        plt.plot(x_vals, y_vals_f, label=f'$f(x) = {sp.latex(f_expr)}$', color='blue')
        plt.plot(y_vals, x_vals_f_inv, label=f'$f^{{-1}}(y) = {sp.latex(f_inv_expr)}$', color='green') 
        plt.plot(diag_vals, diag_vals, label='y = x', color='red', linestyle='--')

        plt.title('Funktion und Umkehrfunktion')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.grid(True)
        plt.legend()
        if save_fig is not None:
            plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')
        plt.show()


    @staticmethod
    def plot_multiple_functions(functions, x_symbol, x_range=(-10, 10), y_range=None, title="Vergleich mehrerer Funktionen", xlabel="x", ylabel="f(x)", save_fig=None):
        """
        Zeichnet mehrere Funktionen in ein gemeinsames Diagramm.

        Args:
            functions: Liste von Tupeln (Ausdruck, Label)
            x_symbol: Das Sympy-Symbol für die Variable x
            x_range: Tupel mit Start- und Endwert für die x-Achse
            y_range: Optional. Tupel mit Start- und Endwert für die y-Achse. Wenn None, wird automatisch skaliert.
            title: Titel des Plots
            xlabel: Beschriftung der x-Achse
            ylabel: Beschriftung der y-Achse
        """
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        plt.figure(figsize=(10, 6))
        
        for expr, label in functions:
            f = sp.lambdify(x_symbol, expr, "numpy")
            y_vals = f(x_vals)
            plt.plot(x_vals, y_vals, label=label)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        
        # Nur y_range setzen, wenn es angegeben ist
        if y_range is not None:
            plt.ylim(y_range)
            
        plt.legend()
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_boundedness(functions, x_symbol, x_range=(-10, 10), 
                          upper_bound=None, lower_bound=None,
                          shade_between=True, title="Beschränktheit von Funktionen",
                          xlabel="x", ylabel="f(x)"):
        """
        Visualisiert die Beschränktheit: Plottet Funktionen und optional obere/untere Schranken.
        
        Args:
            functions: Liste von Tupeln (sympy-Ausdruck, Label)
            x_symbol: SymPy-Symbol
            x_range: (xmin, xmax)
            upper_bound: float oder SymPy-Ausdruck für obere Schranke (konstant oder x-abhängig)
            lower_bound: float oder SymPy-Ausdruck für untere Schranke (konstant oder x-abhängig)
            shade_between: True -> Bereich zwischen Schranken schattieren, falls beide gesetzt
            title, xlabel, ylabel: Plot-Beschriftungen
        """
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        plt.figure(figsize=(10, 6))

        # Funktionen zeichnen
        for expr, label in functions:
            f = sp.lambdify(x_symbol, expr, 'numpy')
            y_vals = f(x_vals)
            plt.plot(x_vals, y_vals, label=label)

        # Schranken vorbereiten (konstant oder funktional)
        ub_vals = None
        lb_vals = None
        if upper_bound is not None:
            if isinstance(upper_bound, (int, float)):
                ub_vals = np.full_like(x_vals, float(upper_bound))
            else:
                ub = sp.lambdify(x_symbol, upper_bound, 'numpy')
                ub_vals = ub(x_vals)
            plt.plot(x_vals, ub_vals, color='red', linestyle='--', linewidth=1.8, label='Obere Schranke')
        if lower_bound is not None:
            if isinstance(lower_bound, (int, float)):
                lb_vals = np.full_like(x_vals, float(lower_bound))
            else:
                lb = sp.lambdify(x_symbol, lower_bound, 'numpy')
                lb_vals = lb(x_vals)
            plt.plot(x_vals, lb_vals, color='green', linestyle='--', linewidth=1.8, label='Untere Schranke')

        # Schattierung zwischen Schranken
        if shade_between and (ub_vals is not None) and (lb_vals is not None):
            # Nur dort füllen, wo ub >= lb, sonst kann fill_between invertiert sein
            mask = ub_vals >= lb_vals
            plt.fill_between(x_vals, lb_vals, ub_vals, where=mask, color='gray', alpha=0.15, label='Beschränkungsbereich')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_boundedness_with_tolerance(expr, x_symbol, x_range=(-10, 10), L=0.0, epsilon=0.5,
                                        title="Epsilon-Streifen um L", xlabel="x", ylabel="f(x)"):
        """
        Visualisiert einen Epsilon-Streifen um L und markiert Funktionswerte innerhalb/außerhalb.
        Nützlich, um lokal/globale Beschränktheit oder Konvergenz-gegen-L im Funktionskontext zu diskutieren.
        """
        x_vals = np.linspace(x_range[0], x_range[1], 1200)
        f = sp.lambdify(x_symbol, expr, 'numpy')
        y_vals = f(x_vals)

        plt.figure(figsize=(10, 5))
        # Punkte einfärben bzgl. Epsilon-Streifen
        inside = np.abs(y_vals - L) <= epsilon
        outside = ~inside
        plt.plot(x_vals[outside], y_vals[outside], 'b.', label='außerhalb ε-Streifen', alpha=0.7)
        plt.plot(x_vals[inside], y_vals[inside], 'g.', label='innerhalb ε-Streifen', alpha=0.7)

        # Epsilon-Streifen
        plt.axhline(L + epsilon, color='red', linestyle='--', linewidth=1.5)
        plt.axhline(L - epsilon, color='red', linestyle='--', linewidth=1.5, label='ε-Streifen')
        plt.axhline(L, color='black', linestyle=':', linewidth=1.2, label='L')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


    @staticmethod
    def find_and_classify_extrema(expr, x_symbol, x_range=(-10, 10), n_starts=50, tol_root=1e-8, tol_merge=1e-6):
        """
        Findet kritische Punkte (f'(x)=0) im Intervall und klassifiziert sie als
        'min', 'max' oder 'saddle'. Gibt Liste von Tupeln (x0, f(x0), kind) zurück.
        """
        f  = sp.lambdify(x_symbol, expr, 'numpy')
        fp = sp.lambdify(x_symbol, sp.diff(expr, x_symbol), 'numpy')
        fpp_expr = sp.diff(expr, x_symbol, 2)
        fpp = sp.lambdify(x_symbol, fpp_expr, 'numpy')

        a, b = x_range
        # Startpunkte-Grid für nsolve
        starts = np.linspace(a, b, n_starts)

        roots = []
        for s in starts:
            try:
                # Für reelle 1D-Fälle Secant/Newton mit 1 Startpunkt robust genug
                sol = sp.nsolve(sp.Eq(sp.diff(expr, x_symbol), 0), s, tol=tol_root, maxsteps=100)
                x0 = float(sol)
                if a - 1e-9 <= x0 <= b + 1e-9:
                    roots.append(x0)
            except Exception:
                pass

        # Deduplizieren nahe beieinander liegender Lösungen
        roots = sorted(roots)
        uniq = []
        for r in roots:
            if not uniq or abs(r - uniq[-1]) > tol_merge:
                uniq.append(r)

        extrema = []
        for x0 in uniq:
            try:
                fpp_val = float(fpp(x0))
                fx0 = float(f(x0))
                if np.isfinite(fpp_val):
                    if fpp_val > 0:
                        kind = 'min'
                    elif fpp_val < 0:
                        kind = 'max'
                    else:
                        kind = 'saddle'
                else:
                    # Fallback: Vorzeichenwechsel von f' prüfen
                    eps = 1e-4 * max(1.0, abs(x0))
                    left  = float(fp(x0 - eps))
                    right = float(fp(x0 + eps))
                    if left < 0 and right > 0:
                        kind = 'min'
                    elif left > 0 and right < 0:
                        kind = 'max'
                    else:
                        kind = 'saddle'
                extrema.append((x0, fx0, kind))
            except Exception:
                pass

        return extrema

    @staticmethod
    def plot_with_extrema(expr, x_symbol, x_range=(-10, 10), y_range=None,
                        title="Funktion mit lokalen/globalen Extrema",
                        xlabel="x", ylabel="f(x)",
                        mark_global=True):
        """
        Plottet f(x) mit markierten lokalen Extrema (min/max/saddle) und optional
        den globalen Extrema im sichtbaren Bereich.
        """
        f = sp.lambdify(x_symbol, expr, 'numpy')
        x_vals = np.linspace(x_range[0], x_range[1], 2000)
        y_vals = f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=fr'$f(x)={sp.latex(expr)}$', color='steelblue')

        # Lokale Extrema finden und markieren
        extrema = FunktionenBibliothek.find_and_classify_extrema(expr, x_symbol, x_range=x_range)
        xs_min = [x for x, y, k in extrema if k == 'min']
        ys_min = [f(x) for x in xs_min]
        xs_max = [x for x, y, k in extrema if k == 'max']
        ys_max = [f(x) for x in xs_max]
        xs_sad = [x for x, y, k in extrema if k == 'saddle']
        ys_sad = [f(x) for x in xs_sad]

        if xs_min:
            plt.scatter(xs_min, ys_min, c='green', s=60, marker='o', edgecolors='black', linewidths=0.6, label='Lokales Minimum')
            for x0, y0 in zip(xs_min, ys_min):
                plt.axhline(y0, color='green', linestyle=':', linewidth=0.8, alpha=0.6)
        if xs_max:
            plt.scatter(xs_max, ys_max, c='red', s=60, marker='o', edgecolors='black', linewidths=0.6, label='Lokales Maximum')
            for x0, y0 in zip(xs_max, ys_max):
                plt.axhline(y0, color='red', linestyle=':', linewidth=0.8, alpha=0.6)
        if xs_sad:
            plt.scatter(xs_sad, ys_sad, c='gold', s=60, marker='D', edgecolors='black', linewidths=0.6, label='Sattelpunkt')

        # Globale Extrema im Fenster
        if mark_global and len(y_vals) > 0:
            idx_min = int(np.nanargmin(y_vals))
            idx_max = int(np.nanargmax(y_vals))
            xgmin, ygmin = x_vals[idx_min], y_vals[idx_min]
            xgmax, ygmax = x_vals[idx_max], y_vals[idx_max]
            plt.scatter([xgmin], [ygmin], s=120, facecolors='none', edgecolors='green', linewidths=2.0, label='Globales Minimum (Fenster)')
            plt.scatter([xgmax], [ygmax], s=120, facecolors='none', edgecolors='red',   linewidths=2.0, label='Globales Maximum (Fenster)')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        if y_range is not None:
            plt.ylim(y_range)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_convex_concave(f_expr, x_symbol, x_range=(-5,5), n_points=1000, title="Konvexität und Konkavität"):
        # Funktions- und Ableitungs-Lambdas
        f = sp.lambdify(x_symbol, f_expr, "numpy")
        fpp_expr = sp.diff(f_expr, x_symbol, 2)
        fpp = sp.lambdify(x_symbol, fpp_expr, "numpy")
        
        x_vals = np.linspace(x_range[0], x_range[1], n_points)
        y_vals = f(x_vals)
        y_fpp  = fpp(x_vals)
        
        convex = y_fpp > 0
        concave = y_fpp < 0
        
        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, y_vals, label=fr'$f(x) = {sp.latex(f_expr)}$', c='black')
        plt.fill_between(x_vals, y_vals, where=convex, color='orange', alpha=0.3, label='konvex (\(f\'\'(x) > 0\))')
        plt.fill_between(x_vals, y_vals, where=concave, color='cyan', alpha=0.3, label='konkav (\(f\'\'(x) < 0\))')
        
        plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.show()