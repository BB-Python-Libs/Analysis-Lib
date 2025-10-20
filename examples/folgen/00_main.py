
import sympy as sp

# Importiert die 'FolgenBibliothek' aus dem Paket 'analysis_lib'
from analysis_lib.folgen import FolgenBibliothek

n = sp.symbols('n')
folge = (n**2 + 3*n -1)/(2*n**2+1)

print("Symbolische Tabelle der Folge ")
FolgenBibliothek.print_symbolic_sequence_table(folge, n, 1, 10)

print("Numerische Tabelle der Folge ")
FolgenBibliothek.print_numeric_sequence_table(folge, n, 1, 10)

print("Plot für die Folge ")
FolgenBibliothek.plot_indexwert(folge, n, n_start=1, n_end=200)

print("Plot für die Folge ")
FolgenBibliothek.plot_xy(folge, n, n_start=1, n_end=200)

print("Plot für die Folge mit Epsilon-Band-Kriterium")
FolgenBibliothek.plot_convergence_auto(folge, n, n_start=1, n_end=200, epsilon=0.02)

print("--- Beispiel: Intervallschachtelung für die Eulersche Zahl e ---")
e_sequences = [
    ((1 + 1/n)**n, r'Untere Schranke: $(1 + \frac{1}{n})^n$'),
    ((1 + 1/n)**(n+1), r'Obere Schranke: $(1 + \frac{1}{n})^{n+1}$'),
    (sp.E, r'Eulersche Zahl: $e \approx 2.71828$')
]

# Die Funktion erhält die Liste der Folgen, das Index-Symbol und den darzustellenden Bereich.
FolgenBibliothek.plot_multiple_sequences(
    sequences_to_plot=e_sequences,
    n_symbol=n,
    n_start=1,
    n_end=50,
    title="Intervallschachtelung für die Eulersche Zahl e"
)

print("\n--- Beispiel: Vergleich der Konvergenzgeschwindigkeit von Nullfolgen ---")
zero_sequences = [
        (1/n, r'Harmonische Folge: $\frac{1}{n}$'),
        ((5/6)**n, r'Geometrische Folge: $(\frac{5}{6})^n$'),
        ((1/2)**n, r'Geometrische Folge: $(\frac{1}{2})^n$')
    ]

# Aufruf der Funktion zum Plotten mehrerer Folgen
FolgenBibliothek.plot_multiple_sequences(
        sequences_to_plot=zero_sequences,
        n_symbol=n,
        n_start=1,
        n_end=25,
        title="Vergleich der Konvergenzgeschwindigkeit von Nullfolgen"
    )
