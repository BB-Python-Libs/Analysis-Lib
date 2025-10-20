
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
