import numpy as np
import matplotlib.pyplot as plt

# Parameter
x = np.linspace(0, 1, 1000)
n_values = [1, 2, 4, 8, 16, 32]  # Potenzen n
p1 = 0.5       # Erster Punkt x = 1/2
p2 = 2/3       # Zweiter Punkt x = 2/3
p3 = 0.9       # Dritter Punkt x = 0.9

# Plot-Setup
plt.figure(figsize=(10, 7))

# Grenzfunktion (gestrichelt)
plt.plot(x, np.zeros_like(x), 'k--', linewidth=2, label=r'$f(x) = 0$ für $x<1$')
plt.plot(1, 1, 'ko', markersize=8, label=r'$f(1) = 1$')

# Farben für die Kurven
colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_values)))

for i, n in enumerate(n_values):
    y = x**n
    color = colors[i]
    
    # 1. Funktionsgraph zeichnen
    plt.plot(x, y, color=color, linewidth=2, label=f'$n={n}$')
    
    # 2. Die Punkte (1/2)^n und (2/3)^n auf dieser Kurve markieren
    # Punkt bei x = 1/2
    plt.plot(p1, p1**n, marker='o', color=color, markeredgecolor='black', markersize=8)
    # Punkt bei x = 2/3
    plt.plot(p2, p2**n, marker='s', color=color, markeredgecolor='black', markersize=8)
    # Punkt bei x = 0.9
    plt.plot(p3, p3**n, marker='^', color=color, markeredgecolor='black', markersize=8)

# Dummy-Plots für die Legende der Punkte (damit sie nur einmal auftauchen)
plt.plot([], [], 'ko', markeredgecolor='black', label=r'Werte für $x=1/2$')
plt.plot([], [], 'ks', markeredgecolor='black', label=r'Werte für $x=2/3$')
plt.plot([], [], 'k^', markeredgecolor='black', label=r'Werte für $x=0.9$')

plt.xlabel('$x$', fontsize=14)
plt.ylabel('$x^n$', fontsize=14)
plt.title(r'Punktweise Konvergenz von $x^n$ mit Werten für $x=1/2$, $x=2/3$ und $x=0.9$', fontsize=16)
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1.02)
plt.ylim(-0.05, 1.05)
plt.tight_layout()

plt.savefig("Pktweise_Konvergenz.png", dpi=300, bbox_inches='tight')

plt.show()
