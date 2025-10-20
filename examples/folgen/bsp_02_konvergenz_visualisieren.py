import sympy as sp
import matplotlib.pyplot as plt

# Importiert die 'FolgenBibliothek' aus dem Paket 'analysis_lib'
from analysis_lib.folgen import FolgenBibliothek

def beispiel_epsilon_band_kriterium():
    n = sp.symbols('n')
    folge = 1 / n

    plt.ion()

    print("Plot für die Folge 1/n mit Epsilon-Band-Kriterium")
    FolgenBibliothek.plot_convergence_auto(folge, n, n_start=1, n_end=100, epsilon=0.03)

    input("--> Alle Plots zum Beispiel sind offen. Bitte drücken Sie Enter, um fortzufahren.")
    plt.close('all') 

# --- Hauptteil des Skripts ---

# Der folgende Block wird nur ausgeführt, wenn das Skript direkt gestartet wird
# (und nicht, wenn es von einem anderen Skript importiert wird).
# Dies ist die Standardmethode, um ein Python-Skript 'ausführbar' zu machen.
if __name__ == "__main__":
    
    # Ruft das Beispiel "harmonische Folge" auf.
    beispiel_epsilon_band_kriterium()