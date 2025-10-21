import sympy as sp
import matplotlib.pyplot as plt

# Importiert die 'FolgenBibliothek' aus dem Paket 'analysis_lib'
from analysis_lib.folgen import FolgenBibliothek

plt.ion()

n = sp.symbols('n')
folge = 1/n

print("Symbolische Tabelle der Folge ")
FolgenBibliothek.print_symbolic_sequence_table(folge, n, 1, 10)

input("Bitte drücken Sie Enter, um fortzufahren.")

print("Numerische Tabelle der Folge ")
FolgenBibliothek.print_numeric_sequence_table(folge, n, 1, 10)

input("Bitte drücken Sie Enter, um fortzufahren.")

print("Plot für die Folge ")
FolgenBibliothek.plot_indexwert(folge, n, n_start=1, n_end=30)

print("Plot für die Folge ")
FolgenBibliothek.plot_xy(folge, n, n_start=1, n_end=200)

print("Plot für die Folge mit Epsilon-Band-Kriterium")
FolgenBibliothek.plot_convergence_auto(folge, n, n_start=1, n_end=50, epsilon=0.05)

print("Vergleich der Konvergenzgeschwindigkeit von Nullfolgen")
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

print("Illustration des Supremums nach oben beschränkter Folgen")
folge = 2 - 1/n
FolgenBibliothek.plot_supremum_illustration(folge, n, sup_value=2, n_start=1, n_end=30, epsilon=0.1)

folge = (-1)**n

FolgenBibliothek.plot_supremum_illustration(folge, n, sup_value=1, n_start=1, n_end=30, epsilon=0.1)

folge = sp.sin(n)

FolgenBibliothek.plot_supremum_illustration(folge, n, sup_value=1, n_start=1, n_end=100, epsilon=0.1)

folge = 2 - 1/n + (-5/6)**n

# 1. Berechnung der Werte der ersten 10 Folgenglieder
werte_liste = [folge.subs(n, k).evalf() for k in range(1, 11)]

# 2. Bestimmung des Maximums und des zugehörigen Index
max_wert = max(werte_liste)
# Da der Listenindex bei 0 beginnt, n aber bei 1, addieren wir 1
max_index_n = werte_liste.index(max_wert) + 1

FolgenBibliothek.plot_supremum_illustration(folge, n, sup_value=max(2,max_wert), n_start=1, n_end=30, epsilon=0.1)

input("Bitte drücken Sie Enter, um fortzufahren.")
plt.close('all')


