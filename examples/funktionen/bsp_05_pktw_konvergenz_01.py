import numpy as np
import matplotlib.pyplot as plt

# Parameter
x = np.linspace(-1, 1, 1000)
n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Potenzen n
points = [-8/9, -2/3, -1/2, 1/2, 2/3, 8/9]

# Farben und Marker für die Punkte definieren
# Wir nutzen eine Colormap für die Kurven
colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_values)))

# Plot-Setup
plt.figure(figsize=(12, 8))

# Grenzfunktion visualisieren:
# f(x) = 0 für |x| < 1
# f(1) = 1
# f(-1) existiert nicht im klassischen Konvergenzsinne (oszilliert), 
# aber wir zeichnen die Nulllinie zur Orientierung.
plt.plot([-1, 1], [0, 0], 'k--', linewidth=2, label=r'$f(x) = 0$ für $|x|<1$')
plt.plot(1, 1, 'ko', markersize=8, label=r'$f(1)=1$')
# (Optional: Punkt bei -1 oszilliert zwischen -1 und 1, daher kein Grenzwert)

# Für jedes n die Kurve zeichnen
for i, n in enumerate(n_values):
    y = x**n
    color = colors[i]
    
    # 1. Funktionsgraph zeichnen
    plt.plot(x, y, color=color, linewidth=2, label=f'$n={n}$')
    
    # 2. Die spezifischen Punkte auf dieser Kurve markieren
    # Wir nutzen verschiedene Marker-Typen, um die x-Werte besser zu unterscheiden,
    # oder einfach einheitliche Punkte. Hier einheitlich, da die Farbe n zugeordnet ist.
    for px in points:
        py = px**n
        plt.plot(px, py, marker='o', color=color, markeredgecolor='black', markersize=6)

# Dummy-Plots für die Legende der Punkte (damit man weiß, wo die x-Werte liegen)
# Wir zeichnen sie ganz unten auf der x-Achse als "Ticks" ein oder beschriften sie direkt im Plot?
# Besser: Wir fügen vertikale Hilfslinien ein oder lassen es clean.
# Hier: Einfache Legende ist schwierig bei so vielen Punkten, daher plotten wir
# die Punkte im Graph, aber beschriften sie ggf. an der x-Achse.

plt.xlabel('$x$', fontsize=14)
plt.ylabel('$x^n$', fontsize=14)
plt.title(r'Punktweise Konvergenz von $x^n$ auf $[-1,1]$ mit markierten Punkten', fontsize=16)

# x-Achse so beschriften, dass unsere speziellen Punkte auftauchen
tick_vals = sorted(points + [-1, 0, 1])
tick_labels = [f"{val:.2f}" for val in tick_vals]
# Manuell schönere Brüche für die Labels (optional, aber didaktisch wertvoll):
tick_labels_frac = ['-1', '-8/9', '-2/3', '-1/2', '0', '1/2', '2/3', '8/9', '1']
plt.xticks([-1, -8/9, -2/3, -1/2, 0, 1/2, 2/3, 8/9, 1], tick_labels_frac, rotation=45)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.xlim(-1.05, 1.05)
plt.ylim(-1.05, 1.05)
plt.tight_layout()

plt.savefig("Pktweise_Konvergenz_01.png", dpi=300, bbox_inches='tight')

plt.show()
