# Importiert die notwendigen Bibliotheken:
# - sympy für symbolische Mathematik
# - matplotlib.pyplot zum Erstellen von Plots
import sympy as sp
import matplotlib.pyplot as plt

# Importiert die 'FolgenBibliothek' aus dem Paket 'analysis_lib'
from analysis_lib.folgen import FolgenBibliothek

# --- Definition der einzelnen Beispiele als gekapselte Funktionen ---

def beispiel_harmonische_folge():
    """
    Beispiel: Führt eine vollständige Analyse der harmonischen Folge a_n = 1/n durch.
    - Gibt symbolische und numerische Tabellen aus.
    - Erstellt zwei verschiedene Visualisierungen.
    """
    # Überschrift für das Beispiel in der Konsolenausgabe
    print("\n--- Beispiel: Analyse der harmonischen Folge a_n = 1/n ---")

    # 1. Vorbereitung: Definiert das Symbol 'n' und den symbolischen Ausdruck für die Folge.
    n = sp.Symbol('n')
    folge = 1 / n

    # 2. Tabellarische Analyse: Gibt die ersten Glieder der Folge aus.
    FolgenBibliothek.print_symbolic_sequence_table(folge, n, n_start=1, n_end=8)
    FolgenBibliothek.print_numeric_sequence_table(folge, n, n_start=1, n_end=8, precision=5)

    # 3. Visuelle Analyse: Erstellt und zeigt die Plots an.
    # 'plt.ion()' (interactive on) sorgt dafür, dass das Skript nach dem plt.show()
    # in den Plot-Funktionen weiterläuft und die Fenster offen bleiben.
    plt.ion()
    FolgenBibliothek.plot_indexwert(folge, n, n_start=1, n_end=25)
    FolgenBibliothek.plot_xy(folge, n, n_start=1, n_end=25)

    # 4. Nutzerinteraktion: Pausiert das Skript, damit die Plots in Ruhe betrachtet werden können.
    # Das Programm wartet hier auf eine Eingabe des Nutzers (Enter), bevor es fortfährt.
    input("--> Alle Plots zum Beispiel sind offen. Bitte drücken Sie Enter, um fortzufahren.")
    plt.close('all') # Schließt alle offenen Matplotlib-Fenster

def beispiel_alternierende_harmonische_folge():
    """
    Beispiel: Führt eine vollständige Analyse der harmonischen Folge a_n = 1/n durch.
    - Gibt symbolische und numerische Tabellen aus.
    - Erstellt zwei verschiedene Visualisierungen.
    """
    # Überschrift für das Beispiel in der Konsolenausgabe
    print("\n--- Beispiel: Analyse der harmonischen Folge a_n = (-1)^n/n ---")

    # 1. Vorbereitung: Definiert das Symbol 'n' und den symbolischen Ausdruck für die Folge.
    n = sp.Symbol('n')
    folge = (-1)**n / n

    # 2. Tabellarische Analyse: Gibt die ersten Glieder der Folge aus.
    FolgenBibliothek.print_symbolic_sequence_table(folge, n, n_start=1, n_end=8)
    FolgenBibliothek.print_numeric_sequence_table(folge, n, n_start=1, n_end=8, precision=5)

    # 3. Visuelle Analyse: Erstellt und zeigt die Plots an.
    # 'plt.ion()' (interactive on) sorgt dafür, dass das Skript nach dem plt.show()
    # in den Plot-Funktionen weiterläuft und die Fenster offen bleiben.
    plt.ion()
    FolgenBibliothek.plot_indexwert(folge, n, n_start=1, n_end=25)
    FolgenBibliothek.plot_xy(folge, n, n_start=1, n_end=25)

    # 4. Nutzerinteraktion: Pausiert das Skript, damit die Plots in Ruhe betrachtet werden können.
    # Das Programm wartet hier auf eine Eingabe des Nutzers (Enter), bevor es fortfährt.
    input("--> Alle Plots zum Beispiel sind offen. Bitte drücken Sie Enter, um fortzufahren.")
    plt.close('all') # Schließt alle offenen Matplotlib-Fenster

def beispiel_eulersche_folge():
    """
    Beispiel: Führt eine Analyse der Folge a_n = (1 + 1/n)^n durch,
    die gegen die Eulersche Zahl e konvergiert.
    """
    print("\n--- Beispiel: Analyse der Folge a_n = (1 + 1/n)^n (Konvergenz gegen e) ---")

    n = sp.Symbol('n')
    folge = (1 + 1/n) ** n

    FolgenBibliothek.print_symbolic_sequence_table(folge, n, n_start=1, n_end=8)
    FolgenBibliothek.print_numeric_sequence_table(folge, n, n_start=1, n_end=8, precision=15)

    plt.ion()
    FolgenBibliothek.plot_indexwert(folge, n, n_start=1, n_end=50) # längerer Bereich, um Konvergenz zu sehen
    FolgenBibliothek.plot_xy(folge, n, n_start=1, n_end=50)

    input("--> Alle Plots zum Beispiel sind offen. Bitte drücken Sie Enter, um fortzufahren.")
    plt.close('all')

# --- Hauptteil des Skripts ---

# Der folgende Block wird nur ausgeführt, wenn das Skript direkt gestartet wird
# (und nicht, wenn es von einem anderen Skript importiert wird).
# Dies ist die Standardmethode, um ein Python-Skript 'ausführbar' zu machen.
if __name__ == "__main__":
    
    # Ruft das Beispiel "harmonische Folge" auf.
    beispiel_harmonische_folge()
 
    # Ruft das Beispiel "alternierende harmonische Folge" auf.
    beispiel_alternierende_harmonische_folge()

    # Ruft die Beispiel "Eulersche Folge" auf.
    beispiel_eulersche_folge()
    
    print("\nAlle Beispiele wurden erfolgreich ausgeführt.")

