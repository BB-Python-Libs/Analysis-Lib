import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Parameter
x = np.linspace(-np.pi, np.pi, 2000)
n_start = 15
n_end = 20  # Wir zeichnen n von 5 bis 15
n_range = range(n_start, n_end + 1)

# Epsilon für den schlechtesten Fall (n_start)
# Alle nachfolgenden n haben ein kleineres Epsilon, passen also auch hier rein.
epsilon_max = 1 / 5

def f_limit(x):
    return np.cos(x)

def f_n(x, n):
    return np.cos(x) + np.sin(n*x)/n

# Plot Setup
fig, ax = plt.subplots(figsize=(12, 7))

# 1. Epsilon-Schlauch für das kleinste n (n_start) zeichnen
# Wenn f_n_start hier reinpasst, passen alle späteren erst recht rein.
ax.fill_between(x, f_limit(x) - epsilon_max, f_limit(x) + epsilon_max, 
                color='lightgray', alpha=0.4, 
                label=fr'$\epsilon$-Schlauch: $\quad\epsilon = 1/5$')

# 2. Grenzfunktion
ax.plot(x, f_limit(x), 'k--', linewidth=2.5, zorder=10, label=r'Grenzfunktion $f(x) = \cos(x)$')

# 3. Funktionsschar zeichnen mit Colormap
# Wir erstellen eine LineCollection für effizientes und hübsches Plotten
lines = []
colors = []

# Colormap wählen (viridis ist gut lesbar: dunkelblau -> gelb)
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(n_start, n_end)

for n in n_range:
    y = f_n(x, n)
    # Wir plotten jede Linie einzeln, um sie in der Legende referenzieren zu können?
    # Nein, bei vielen Linien ist LineCollection besser oder eine Schleife.
    # Hier Schleife für einfache Legende (nur start/end).
    color = cmap(norm(n))
    
    # Nur die erste und letzte Linie ins Label nehmen, um Legende nicht zu fluten
    label = None
    if n == n_start:
        label = f'$f_n(x)$ für $n={n_start}$ (Start)'
    elif n == n_end:
        label = f'$f_n(x)$ für $n={n_end}$ (Ende)'
        
    ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8, label=label)

# Colorbar hinzufügen, um zu zeigen welches n welche Farbe hat
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Parameter $n$', fontsize=12)

# Achsen und Labels
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$f_n(x)$', fontsize=14)
ax.set_title(fr'Gleichmäßige Konvergenz: Schar $f_n(x)$ für $n \in [{n_start}, {n_end}]$ im Schlauch von $n={n_start}$', fontsize=16)

# Pi-Ticks
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=12)

ax.legend(loc='lower center', framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-1.3, 1.3) # Etwas Platz lassen

plt.tight_layout()

plt.savefig("Glm_Konvergenz_03.png", dpi=300, bbox_inches='tight')

plt.show()
