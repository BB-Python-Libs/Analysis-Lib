import sympy as sp
import numpy as np
import mpmath
import matplotlib.pyplot as plt

class FolgenBibliothek:

    @staticmethod
    def print_symbolic_sequence_table(sequence_expr, n_symbol, n_start=1, n_end=10):
        header_sym = "a_n (Symbolisch)"
        symbolic_values = [str(sequence_expr.subs(n_symbol, k)) for k in range(n_start, n_end + 1)]
        col_width_sym = max(len(header_sym), max(len(s) for s in symbolic_values))

        print(f"\nSymbolische Tabelle der Folge a_n =\n {sp.pretty(sequence_expr)}")
        print("-" * (col_width_sym + 10))
        print(f"{'n':<5} | {header_sym:<{col_width_sym}}")
        print("-" * (col_width_sym + 10))

        for k in range(n_start, n_end + 1):
            term_symbolic = sequence_expr.subs(n_symbol, k)
            print(f"{k:<5} | {str(term_symbolic):<{col_width_sym}}")

        print("-" * (col_width_sym + 10))

    @staticmethod
    def print_numeric_sequence_table(sequence_expr, n_symbol, n_start=1, n_end=10, precision=30):
        mpmath.mp.dps = precision + 5
        header_num = f"a_n (Numerisch, {precision} Stellen)"

        print(f"\nNumerische Tabelle der Folge a_n =\n {sp.pretty(sequence_expr)}")
        print("-" * (len(header_num) + 10))
        print(f"{'n':<5} | {header_num}")
        print("-" * (len(header_num) + 10))

        for k in range(n_start, n_end + 1):
            term_numeric = sequence_expr.subs(n_symbol, k).evalf(precision)
            numeric_str = f"{term_numeric:.{precision}f}"
            print(f"{k:<5} | {numeric_str}")

        print("-" * (len(header_num) + 10))

    @staticmethod
    def get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end):
        n_values = list(range(n_start, n_end + 1))
        y_values = np.array([float(sequence_expr.subs(n_symbol, k).evalf()) for k in n_values])
        return n_values, y_values

    @staticmethod
    def plot_xy(sequence_expr, n_symbol, n_start=1, n_end=20):
        _, y_values = FolgenBibliothek.get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end)
        x0 = [0.0] * len(y_values)

        plt.figure(figsize=(8, 4))
        plt.axhline(0, color='black', linewidth=1.0)
        plt.plot(y_values, x0, "o", label=f"Werte von $a_n = {sp.latex(sequence_expr)}$ für n={n_start}..{n_end}")
        plt.xlabel("Wert a_n")
        plt.yticks([])
        plt.title("Folge als Punktwolke auf der x-Achse")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_indexwert(sequence_expr, n_symbol, n_start=1, n_end=20):
        n_values, y_values = FolgenBibliothek.get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end)

        plt.figure(figsize=(8, 4))
        plt.plot(n_values, y_values, "o:", label=f"$a_n = {sp.latex(sequence_expr)}$")
        plt.xlabel("Index n")
        plt.ylabel("Wert a_n")
        plt.title(f"Folge a_n gegen Index n (für n={n_start}..{n_end})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_convergence_auto(sequence_expr, n_symbol, n_start=1, n_end=50, epsilon=0.1):
        n_values, a_vals = FolgenBibliothek.get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end)
        L_symbolic = None
        try:
            L_symbolic = sp.limit(sequence_expr, n_symbol, sp.oo)
            L = float(L_symbolic.evalf())
            limit_label = fr'Grenzwert L = ${sp.latex(L_symbolic)}$'
        except (NotImplementedError, ValueError, TypeError):
            print(f"Warnung: Symbolischer Grenzwert für {sequence_expr} konnte nicht bestimmt werden.")
            user_input = input('Bitte gib den Grenzwert L als Zahl ein (oder leer lassen zum Abbruch): ').strip()
            if user_input == '':
                print('Abbruch: Grenzwert wurde nicht angegeben.')
                exit(1)
            try:
                L = float(user_input)
                limit_label = f'Grenzwert L (Benutzereingabe) = {L}'
            except ValueError:
                print('Ungültige Eingabe. Programm wird beendet.')
                exit(1)

        plt.figure(figsize=(10, 5))
        plt.plot(n_values, a_vals, 'bo:', label=fr'Folge $a_n = {sp.latex(sequence_expr)}$')
        plt.axhline(L, color='black', linewidth=1.5, label=limit_label)
        plt.axhline(L + epsilon, color='red', linestyle='--', label=fr'$\varepsilon$-Band ($\varepsilon={epsilon}$)')
        plt.axhline(L - epsilon, color='red', linestyle='--')

        N_idx = -1
        for i in range(len(a_vals)):
            if np.all(np.abs(a_vals[i:] - L) < epsilon):
                N_idx = i
                break
        if N_idx != -1:
            n_start_conv = n_values[N_idx]
            plt.scatter(n_values[N_idx:], a_vals[N_idx:], color='green', zorder=5, label=fr'Alle $|a_n - L| < \varepsilon$ für $n \geq {n_start_conv}$')

        plt.xlabel('Index $n$')
        plt.ylabel('Wert $a_n$')
        plt.title(r'Visualisierung des $\varepsilon$-Konvergenzkriteriums')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multiple_sequences(sequences_to_plot, n_symbol, n_start=1, n_end=50, title="Vergleich mehrerer Folgen"):
        plt.figure(figsize=(12, 7))
        for seq_expr, label in sequences_to_plot:
            n_values, y_values = FolgenBibliothek.get_numeric_sequence_values(seq_expr, n_symbol, n_start, n_end)
            plt.plot(n_values, y_values, 'o:', markersize=4, label=label)
        plt.xlabel('Index $n$')
        plt.ylabel('Wert $a_n$')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_supremum_illustration(sequence_expr, n_symbol, sup_value, n_start=1, n_end=50, epsilon=0.1, title="Visualisierung des Supremums"):
        """
        Visualisiert die Konvergenz einer beschränkten, monoton wachsenden Folge
        gegen ihr Supremum.
        """
        n_values, y_values = FolgenBibliothek.get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end)

        plt.figure(figsize=(10, 6))

        # 1. Folgenglieder als Punkte und Linie
        plt.plot(n_values, y_values, 'bo:', label=f'$a_n = {sp.latex(sequence_expr)}$', markersize=5)

        # 2. Supremum als "Decke" einzeichnen
        plt.axhline(sup_value, color='black', linewidth=2, label=f'Supremum $\\sup(a_n) = {sup_value}$')

        # 3. Nur die untere Epsilon-Grenze einzeichnen.
        # Dies visualisiert die Bedingung: sup - ε ist KEINE obere Schranke mehr.
        plt.axhline(sup_value - epsilon, color='red', linestyle='--', alpha=0.7, 
                    label=f'Untere Grenze des $\\varepsilon$-Streifens ($L - \\varepsilon$)')

        # 4. Alle Folgenglieder finden, die zwischen sup - ε und sup liegen
        indices_in_strip = [i for i, y in enumerate(y_values) if (sup_value - epsilon) < y <= sup_value]
        
        if indices_in_strip:
            # Finde das erste Folgenglied, das in den Streifen "eindringt"
            first_idx_in_strip = indices_in_strip[0]
            n_for_strip = n_values[first_idx_in_strip]
            
            # Hebe alle Punkte im Streifen hervor
            strip_n_values = [n_values[i] for i in indices_in_strip]
            strip_y_values = [y_values[i] for i in indices_in_strip]
            
            plt.scatter(strip_n_values, strip_y_values, color='green', s=60, zorder=5, 
                        label=f'Mindestens ein $a_n$ für $n \\geq {n_for_strip}$ liegt im Streifen')

        # Plot-Anpassungen
        plt.title(title)
        plt.xlabel("Index $n$")
        plt.ylabel("Wert $a_n$")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_squeeze_theorem(
        lower_seq, middle_seq, upper_seq, n_symbol, 
        n_range=(1, 50),
        title="Visualisierung des Einschließungskriteriums",
        lower_label='Untere Folge',
        middle_label='Eingeschlossene Folge',
        upper_label='Obere Folge'
    ):
        """
        Visualisiert das Einschließungskriterium mit robuster Behandlung von konstanten Folgen.
        """
        n_start, n_end = n_range
        n_values = np.arange(n_start, n_end + 1, dtype=float)

        def evaluate_seq(seq_expr):
            """
            Wertet einen SymPy-Ausdruck aus. Wenn er konstant ist, wird ein Array
            der passenden Länge zurückgegeben.
            """
            # Prüfen, ob der Ausdruck konstant ist
            if sp.simplify(seq_expr).is_constant():
                # Konstanten Wert ermitteln
                const_val = float(seq_expr.evalf() if hasattr(seq_expr, 'evalf') else seq_expr)
                # Ein Array mit diesem Wert in der Länge von n_values zurückgeben
                return np.full_like(n_values, fill_value=const_val)
            else:
                # Ansonsten die Funktion normal auswerten
                f = sp.lambdify(n_symbol, seq_expr, 'numpy')
                return f(n_values)

        # Werte für alle drei Folgen robust berechnen
        y_lower = evaluate_seq(lower_seq)
        y_middle = evaluate_seq(middle_seq)
        y_upper = evaluate_seq(upper_seq)

        plt.figure(figsize=(12, 7))

        # Plotten der Folgen
        plt.plot(n_values, y_upper, 'r--', label=upper_label)
        plt.plot(n_values, y_lower, 'b--', label=lower_label)
        plt.plot(n_values, y_middle, 'go', markersize=4, label=middle_label)
        plt.fill_between(n_values, y_lower, y_upper, color='gray', alpha=0.2, label='Einschließungsbereich')
        
        # Grenzwertberechnung (bereits robust)
        try:
            if sp.simplify(lower_seq).is_constant():
                limit = float(lower_seq)
            else:
                limit = sp.limit(lower_seq, n_symbol, sp.oo)
            
            if limit is not None and limit.is_finite:
                plt.axhline(float(limit), color='black', linestyle=':', linewidth=2, 
                           label=f'Gemeinsamer Grenzwert L = {limit}')
        except Exception as e:
            print(f"Warnung: Der Grenzwert konnte nicht berechnet werden: {e}")

        # Plot-Anpassungen
        plt.title(title)
        plt.xlabel('Index n')
        plt.ylabel('Wert')
        plt.legend()
        plt.grid(True, linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()


