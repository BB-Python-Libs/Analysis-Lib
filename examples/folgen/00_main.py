
import sympy as sp
import matplotlib.pyplot as plt

import bsp_01_grundlagen_folgen as grundlagen
import bsp_02_konvergenz_visualisieren as epsilon_band

# Importiert die 'FolgenBibliothek' aus dem Paket 'analysis_lib'
from analysis_lib.folgen import FolgenBibliothek

#grundlagen.beispiel_alternierende_harmonische_folge()
#epsilon_band.beispiel_epsilon_band_kriterium()

plt.ioff()  # Deaktiviert den interaktiven Modus

n = sp.symbols('n')
folge = (n**2 + 3*n -1)/(2*n**2+1)

print("Plot für die Folge mit Epsilon-Band-Kriterium")
FolgenBibliothek.plot_convergence_auto(folge, n, n_start=1, n_end=50, epsilon=0.05)