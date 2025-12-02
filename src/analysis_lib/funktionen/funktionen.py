import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon # Dieser Import fehlte

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
        x_vals = np.linspace(x_range[0], x_range[1], 4000)
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

    @staticmethod
    def calculate_max_delta(f_numpy, x0, epsilon, x_search_range=(-10, 10), resolution=10000):
        """
        Berechnet numerisch das maximal mögliche Delta für ein gegebenes Epsilon.
        """
        fx0 = f_numpy(x0)
        
        # Erzeuge sehr feines Gitter um x0
        x_vals = np.linspace(x_search_range[0], x_search_range[1], resolution)
        y_vals = f_numpy(x_vals)
        
        # Prüfe, wo die Funktion den Epsilon-Streifen verlässt
        # Bedingung: |f(x) - f(x0)| > epsilon
        outside_mask = np.abs(y_vals - fx0) > epsilon
        
        # Indizes der Werte außerhalb des Streifens
        outside_indices = np.where(outside_mask)[0]
        
        if len(outside_indices) == 0:
            # Die Funktion verlässt den Streifen im Suchbereich nie -> Delta durch Suchbereich begrenzt
            return min(abs(x_search_range[0] - x0), abs(x_search_range[1] - x0))
        
        # Finde die x-Werte, die x0 am nächsten liegen, aber außerhalb sind
        x_outside = x_vals[outside_indices]
        distances = np.abs(x_outside - x0)
        
        # Das kleinste dieser Distanzen ist unser Limit für Delta
        # Wir ziehen ein winziges Stückchen ab, um sicher "innerhalb" zu sein
        min_dist = np.min(distances)
        return min_dist * 0.99  # Sicherheitsfaktor, damit wir optisch "drin" bleiben

    @staticmethod
    def plot_epsilon_delta_auto(f_expr, x_symbol, x0, epsilon, delta=None, auto_delta=False, 
                                x_range=(-5, 5), title_prefix="Stetigkeit", save_fig=None):
        """
        Plottet das Epsilon-Delta-Kriterium. 
        Wenn auto_delta=True, wird delta automatisch berechnet.
        """
        # Funktion numerisch
        f = sp.lambdify(x_symbol, f_expr, "numpy")
        fx0 = float(f(x0))
        
        # Automatische Berechnung falls gewünscht
        if auto_delta:
            calc_delta = FunktionenBibliothek.calculate_max_delta(f, x0, epsilon, x_search_range=x_range)
            # Falls der User gar kein Delta angegeben hat oder auto_delta Vorrang hat:
            current_delta = calc_delta
            delta_label_text = fr"Auto $\delta \approx {current_delta:.3f}$"
        else:
            if delta is None: raise ValueError("Bitte delta angeben oder auto_delta=True setzen.")
            current_delta = delta
            delta_label_text = fr"Manuelles $\delta = {current_delta}$"

        # Plot-Daten generieren
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = f(x_vals)
        
        mask_delta = np.abs(x_vals - x0) <= current_delta
        x_in_delta = x_vals[mask_delta]
        y_in_delta = y_vals[mask_delta]
        
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. Funktion plotten
        ax.plot(x_vals, y_vals, label=fr'$f(x) = {sp.latex(f_expr)}$', color='steelblue', alpha=0.8)
        
        # 2. Epsilon-Delta-Box
        box = Rectangle(
            (x0 - current_delta, fx0 - epsilon),
            width=2 * current_delta,
            height=2 * epsilon,
            facecolor='orange',
            alpha=0.15,
            edgecolor='darkorange',
            linewidth=2,
            linestyle='--',
            label=r'Sicherer Bereich ($\epsilon$-$\delta$-Box)'
        )
        ax.add_patch(box)
        
        # 3. Relevanter Graph-Abschnitt (fett rot)
        ax.plot(x_in_delta, y_in_delta, color='red', linewidth=2.5, 
                label=r'Graph im $\delta$-Intervall')
        
        # 4. Hilfslinien für Epsilon (Grenzen)
        ax.axhline(fx0 + epsilon, color='green', linestyle=':', linewidth=1.5, label=r'$y = f(x_0) \pm \epsilon$')
        ax.axhline(fx0 - epsilon, color='green', linestyle=':', linewidth=1.5)
        
        # 5. Hilfslinien für Delta (Grenzen)
        ax.axvline(x0 - current_delta, color='purple', linestyle=':', linewidth=1.5, label=r'$x = x_0 \pm \delta$')
        ax.axvline(x0 + current_delta, color='purple', linestyle=':', linewidth=1.5)
        
        # 6. Punkt x0
        ax.scatter([x0], [fx0], color='black', zorder=10, s=80, label=f'$(x_0, f(x_0))$')

        # Titel und Achsen
        ax.set_title(f"{title_prefix}\nGegeben $\epsilon = {epsilon}$, {delta_label_text}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Zoom einstellen, damit man die Box gut sieht, aber auch etwas Umgebung
        plt.xlim(x_range)
        # y-Limit dynamisch anpassen
        y_min, y_max = min(y_vals), max(y_vals)
        plt.ylim(min(y_min, fx0 - 2*epsilon), max(y_max, fx0 + 2*epsilon))
        
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')
        plt.show()


    @staticmethod
    def find_threshold_M(f_numpy, L, epsilon, search_start=0, search_end=100, direction='plus'):
        """
        Findet numerisch die Schranke M, ab der f(x) im Epsilon-Schlauch bleibt.
        """
        # Wir suchen 'rückwärts' vom Ende des Bereichs, um den *letzten* Punkt zu finden,
        # an dem die Funktion den Schlauch verlässt.
        
        # Gitter erzeugen
        x_vals = np.linspace(search_start, search_end, 10000)
        y_vals = f_numpy(x_vals)
        
        # Wo ist die Funktion außerhalb des Epsilon-Streifens?
        outside = np.abs(y_vals - L) > epsilon
        outside_indices = np.where(outside)[0]
        
        if len(outside_indices) == 0:
            # Die Funktion ist im gesamten Bereich schon im Schlauch
            return search_start
        
        if direction == 'plus':
            # Bei x -> +inf suchen wir den größten x-Wert, der noch draußen ist.
            # Alles rechts davon ist "sicher".
            last_outside_idx = outside_indices[-1]
            M = x_vals[last_outside_idx]
            return M
        else: # direction == 'minus'
            # Bei x -> -inf suchen wir den kleinsten x-Wert, der noch draußen ist.
            # Alles links davon ist "sicher".
            first_outside_idx = outside_indices[0]
            M = x_vals[first_outside_idx]
            return M

    @staticmethod
    def plot_limit_at_infinity(f_expr, x_symbol, limit_val, epsilon=0.5, direction='plus', 
                            x_range=(-10, 50), auto_M=True, M_manual=None,
                            title="Grenzwert im Unendlichen", save_fig=None):
        """
        Visualisiert den Grenzwert für x -> +/- unendlich.
        
        Args:
            direction: 'plus' für x -> +oo, 'minus' für x -> -oo
        """
        f = sp.lambdify(x_symbol, f_expr, "numpy")
        
        # x-Werte für den Plot
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = f(x_vals)
        
        # M bestimmen
        if auto_M:
            if direction == 'plus':
                search_start, search_end = max(0, x_range[0]), x_range[1]
                M = FunktionenBibliothek.find_threshold_M(f, limit_val, epsilon, search_start, search_end, 'plus')
                # Kleiner Sicherheitszuschlag für die Optik
                M_plot = M
            else:
                search_start, search_end = x_range[0], min(0, x_range[1])
                M = FunktionenBibliothek.find_threshold_M(f, limit_val, epsilon, search_start, search_end, 'minus')
                M_plot = M
        else:
            M_plot = M_manual if M_manual is not None else (x_range[1] if direction=='plus' else x_range[0])
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Epsilon-Schlauch (Horizontaler Streifen über den gesamten Bereich)
        ax.axhspan(limit_val - epsilon, limit_val + epsilon, color='green', alpha=0.1, label=r'$\epsilon$-Schlauch')
        ax.axhline(limit_val, color='green', linestyle='--', linewidth=1, label=f'Grenzwert L={limit_val}')
        
        # 2. Funktion plotten
        ax.plot(x_vals, y_vals, label=fr'$f(x) = {sp.latex(f_expr)}$', color='blue', linewidth=1.5)
        
        # 3. "Sicherer" Bereich markieren (ab M)
        if direction == 'plus':
            # Bereich rechts von M
            valid_mask = x_vals >= M_plot
            ax.plot(x_vals[valid_mask], y_vals[valid_mask], color='red', linewidth=2.5, label=r'Konvergenter Teil ($x > M$)')
            
            # Vertikale Linie M
            ax.axvline(M_plot, color='red', linestyle='-', linewidth=1.5, label=f'Schranke M $\\approx {M_plot:.1f}$')
            
            # Markiere den Bereich rechts von M visuell
            # (Schattierung des "gültigen" Schlauchendes)
            poly_coords = [
                (M_plot, limit_val - epsilon), 
                (x_range[1], limit_val - epsilon), 
                (x_range[1], limit_val + epsilon), 
                (M_plot, limit_val + epsilon)
            ]
            
        else: # direction == 'minus'
            # Bereich links von M
            valid_mask = x_vals <= M_plot
            ax.plot(x_vals[valid_mask], y_vals[valid_mask], color='red', linewidth=2.5, label=r'Konvergenter Teil ($x < M$)')
            
            ax.axvline(M_plot, color='red', linestyle='-', linewidth=1.5, label=f'Schranke M $\\approx {M_plot:.1f}$')
            
            poly_coords = [
                (x_range[0], limit_val - epsilon), 
                (M_plot, limit_val - epsilon), 
                (M_plot, limit_val + epsilon), 
                (x_range[0], limit_val + epsilon)
            ]

        # Füge eine schraffierte Box für den "gültigen" Bereich hinzu
        poly = Polygon(poly_coords, facecolor='red', alpha=0.15, hatch='//', edgecolor='red')
        ax.add_patch(poly)
        
        # Beschriftungen
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        
        # Epsilon-Grenzen
        ax.text(x_range[0], limit_val + epsilon, r'$L + \epsilon$', va='bottom', color='green', fontsize=9)
        ax.text(x_range[0], limit_val - epsilon, r'$L - \epsilon$', va='top', color='green', fontsize=9)
        
        ax.legend(loc='best')
        
        # y-Achse auf sinnvollen Bereich beschränken
        # (Funktion kann am Anfang riesig sein, wir wollen den Grenzwert sehen)
        y_mid = limit_val
        y_span = max(2 * epsilon, 5) # Mindestspanne
        ax.set_ylim(y_mid - 3*epsilon, y_mid + 3*epsilon) # Fokus auf den Grenzwert
        
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')
        plt.show()


    @staticmethod
    def calculate_sided_delta(f_numpy, x0, limit_val, epsilon, side='left', x_search_dist=2.0, resolution=5000):
        """
        Berechnet das Delta für einen einseitigen Grenzwert.
        """
        if side == 'left':
            # Suche im Bereich [x0 - dist, x0]
            x_vals = np.linspace(x0 - x_search_dist, x0 - 1e-9, resolution)
        else:
            # Suche im Bereich [x0, x0 + dist]
            x_vals = np.linspace(x0 + 1e-9, x0 + x_search_dist, resolution)
            
        y_vals = f_numpy(x_vals)
        
        # Wo wird der Epsilon-Schlauch verlassen?
        outside = np.abs(y_vals - limit_val) > epsilon
        outside_indices = np.where(outside)[0]
        
        if len(outside_indices) == 0:
            return x_search_dist
        
        # Der relevante Punkt ist derjenige, der x0 am nächsten ist
        x_outside = x_vals[outside_indices]
        distances = np.abs(x_outside - x0)
        
        return np.min(distances) * 0.99

    @staticmethod
    def plot_sided_limits(f_expr, x_symbol, x0, limit_left=None, limit_right=None, 
                        epsilon=0.5, auto_delta=True, manual_delta=0.5,
                        x_range=(-2, 2), title="Einseitige Grenzwerte", save_fig=None):
        """
        Visualisiert links- und rechtsseitige Grenzwerte.
        Kann einen oder beide gleichzeitig darstellen (bei Sprungstellen).
        """
        f = sp.lambdify(x_symbol, f_expr, "numpy")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # x-Werte für den Plot (mit Lücke bei x0 für saubere Darstellung bei Polstellen/Sprüngen)
        x_vals_l = np.linspace(x_range[0], x0 - 1e-6, 1000)
        x_vals_r = np.linspace(x0 + 1e-6, x_range[1], 1000)
        y_vals_l = f(x_vals_l)
        y_vals_r = f(x_vals_r)
        
           # --- ROBUSTE LABEL-ERZEUGUNG ---
        latex_str = sp.latex(f_expr)
        if "cases" in latex_str:
            # Matplotlibs interner Parser scheitert an \begin{cases}, daher vereinfachtes Label
            label_text = r'$f(x)$ (stückweise definiert)'
        else:
            label_text = fr'$f(x)={latex_str}$'
            
        # Plot der Funktion (blau)
        ax.plot(x_vals_l, y_vals_l, color='steelblue', linewidth=1.5, label=label_text)
        ax.plot(x_vals_r, y_vals_r, color='steelblue', linewidth=1.5)
         
        # --- LINKSSEITIGER GRENZWERT ---
        if limit_left is not None:
            delta_l = FunktionenBibliothek.calculate_sided_delta(f, x0, limit_left, epsilon) if auto_delta else manual_delta
            
            # Epsilon-Delta-Box (nur links von x0)
            box_l = Rectangle(
                (x0 - delta_l, limit_left - epsilon), width=delta_l, height=2*epsilon,
                facecolor='green', alpha=0.15, edgecolor='green', hatch='//',
                label=fr'Linksseitig: $\lim_{{x \to x_0^-}} f(x) = {limit_left}$'
            )
            ax.add_patch(box_l)
            
            # Markiere den Grenzwert L-
            ax.scatter([x0], [limit_left], color='green', marker='<', s=80, zorder=5)
            # Graph im Delta-Bereich hervorheben
            mask_l = (x_vals_l >= x0 - delta_l)
            ax.plot(x_vals_l[mask_l], y_vals_l[mask_l], color='green', linewidth=2.5)

        # --- RECHTSSEITIGER GRENZWERT ---
        if limit_right is not None:
            delta_r = FunktionenBibliothek.calculate_sided_delta(f, x0, limit_right, epsilon, side='right') if auto_delta else manual_delta
            
            # Epsilon-Delta-Box (nur rechts von x0)
            box_r = Rectangle(
                (x0, limit_right - epsilon), width=delta_r, height=2*epsilon,
                facecolor='red', alpha=0.15, edgecolor='red', hatch='\\\\',
                label=fr'Rechtsseitig: $\lim_{{x \to x_0^+}} f(x) = {limit_right}$'
            )
            ax.add_patch(box_r)
            
            # Markiere den Grenzwert L+
            ax.scatter([x0], [limit_right], color='red', marker='>', s=80, zorder=5)
            # Graph im Delta-Bereich hervorheben
            mask_r = (x_vals_r <= x0 + delta_r)
            ax.plot(x_vals_r[mask_r], y_vals_r[mask_r], color='red', linewidth=2.5)

        # Vertikale Trennlinie bei x0
        ax.axvline(x0, color='black', linestyle='--', linewidth=1, label=f'$x_0={x0}$')
        
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Zoom einstellen
        all_limits = [l for l in [limit_left, limit_right] if l is not None]
        if all_limits:
            y_center = np.mean(all_limits)
            y_span = max([abs(l - y_center) for l in all_limits]) + 2*epsilon + 1
            ax.set_ylim(y_center - y_span, y_center + y_span)
        
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig + ".png", dpi=300, bbox_inches='tight')
        plt.show()