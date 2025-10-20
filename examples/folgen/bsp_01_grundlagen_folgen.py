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

def beispiel_intervallschachtelung_e():
    """
    Visualisiert die Intervallschachtelung für die Eulersche Zahl e durch die Folgen
    a_n = (1 + 1/n)^n (monoton steigend) und b_n = (1 + 1/n)^(n+1) (monoton fallend).
    """
    print("\n--- Beispiel: Intervallschachtelung für die Eulersche Zahl e ---")

    # 1. Symbol für den Index n definieren
    n = sp.Symbol('n')

    # 2. Definition der zu plottenden Folgen
    # Jede Folge wird als Tupel (SymPy-Ausdruck, LaTeX-Label für die Legende) in einer Liste gespeichert.
    # Das 'r' vor den Strings sorgt dafür, dass Backslashes in den LaTeX-Ausdrücken korrekt interpretiert werden.
    e_sequences = [
        ((1 + 1/n)**n, r'Untere Schranke: $(1 + \frac{1}{n})^n$'),
        ((1 + 1/n)**(n+1), r'Obere Schranke: $(1 + \frac{1}{n})^{n+1}$'),
        (sp.E, r'Eulersche Zahl: $e \approx 2.71828$')
    ]

    # 3. Interaktiven Modus für Matplotlib aktivieren
    # Dies stellt sicher, dass das Skript nach dem Anzeigen des Plots weiterläuft.
    plt.ion()

    # 4. Aufruf der Plot-Funktion aus der Bibliothek, die mehrere Folgen darstellen kann.
    # Die Funktion erhält die Liste der Folgen, das Index-Symbol und den darzustellenden Bereich.
    FolgenBibliothek.plot_multiple_sequences(
        sequences_to_plot=e_sequences,
        n_symbol=n,
        n_start=1,
        n_end=30,
        title="Intervallschachtelung für die Eulersche Zahl e"
    )

    # 5. Warten auf Benutzereingabe, um das Plot-Fenster in Ruhe betrachten zu können.
    input("--> Der Plot zur Intervallschachtelung ist offen. Bitte drücken Sie Enter, um das Programm zu beenden.")
    
    # 6. Schließt alle offenen Plot-Fenster, nachdem der Benutzer bestätigt hat.
    plt.close('all')

def beispiel_vergleich_nullfolgen():
    """
    Visualisiert und vergleicht die Konvergenzgeschwindigkeit von drei verschiedenen Nullfolgen:
    - Harmonische Folge: a_n = 1/n
    - Geometrische Folge: a_n = (5/6)^n
    - Geometrische Folge: a_n = (1/2)^n
    """
    print("\n--- Beispiel: Vergleich der Konvergenzgeschwindigkeit von Nullfolgen ---")

    # 1. Symbol für den Index n definieren
    n = sp.Symbol('n')

    # 2. Definition der zu plottenden Folgen als Liste von Tupeln.
    # Jedes Tupel besteht aus dem SymPy-Ausdruck der Folge und einem LaTeX-formatierten Label.
    zero_sequences = [
        (1/n, r'Harmonische Folge: $\frac{1}{n}$'),
        ((5/6)**n, r'Geometrische Folge: $(\frac{5}{6})^n$'),
        ((1/2)**n, r'Geometrische Folge: $(\frac{1}{2})^n$')
    ]

    # 3. Interaktiven Modus für Matplotlib aktivieren
    plt.ion()

    # 4. Aufruf der Funktion zum Plotten mehrerer Folgen aus unserer Bibliothek
    FolgenBibliothek.plot_multiple_sequences(
        sequences_to_plot=zero_sequences,
        n_symbol=n,
        n_start=1,
        n_end=25,
        title="Vergleich der Konvergenzgeschwindigkeit von Nullfolgen"
    )

    # 5. Warten auf Benutzereingabe, um das Plot-Fenster in Ruhe betrachten zu können.
    input("--> Der Plot zum Vergleich der Nullfolgen ist offen. Bitte schließen Sie das Fenster und drücken Sie Enter, um das Programm zu beenden.")
    
    # 6. Schließt alle offenen Plot-Fenster.
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

    beispiel_intervallschachtelung_e()

    beispiel_vergleich_nullfolgen()
    
    print("\nAlle Beispiele wurden erfolgreich ausgeführt.")

