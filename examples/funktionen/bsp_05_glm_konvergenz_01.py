import numpy as np
import matplotlib.pyplot as plt

# Parameter
x = np.linspace(-np.pi, np.pi, 1000)
n_example = 5  # Beispiel n für den Schlauch (gerne anpassen, z.B. 3, 5, 10)
epsilon = 1/n_example  # Das passende Epsilon für die Abschätzung

# Funktionen definieren
def f_limit(x):
    return np.cos(x)

def f_n(x, n):
    return np.cos(x) + np.sin(n*x)/n

# Plot Setup
plt.figure(figsize=(12, 7))

# 1. Epsilon-Schlauch um die Grenzfunktion zeichnen
# Der Schlauch ist f(x) +/- epsilon
y_upper = f_limit(x) + epsilon
y_lower = f_limit(x) - epsilon

plt.fill_between(x, y_lower, y_upper, color='lightgray', alpha=0.5, 
                 label=fr'$\epsilon$-Schlauch ($\epsilon = 1/{n_example}$)')

# 2. Grenzfunktion zeichnen
plt.plot(x, f_limit(x), 'k--', linewidth=2, label=r'Grenzfunktion $f(x) = \cos(x)$')

# 3. Funktionsfolgen-Glied f_n(x) zeichnen
y_n = f_n(x, n_example)
plt.plot(x, y_n, 'b-', linewidth=2, label=fr'$f_{{{n_example}}}(x) = \cos(x) + \frac{{\sin({n_example}x)}}{{{n_example}}}$')

# 4. (Optional) Weitere n zum Vergleich (dünner gezeichnet)
# n_compare = 10
# plt.plot(x, f_n(x, n_compare), 'g:', linewidth=1.5, label=fr'$f_{{{n_compare}}}(x)$')

# Achsen und Labels
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f_n(x)$', fontsize=14)
plt.title(fr'Gleichmäßige Konvergenz: $f_n(x)$ liegt komplett im Schlauch von $f(x) \pm \frac{{1}}{{n}}$', fontsize=16)

# Ticks auf x-Achse als Pi-Vielfache
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
plt.xticks(ticks, labels, fontsize=12)

plt.legend(loc='lower center', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("Glm_Konvergenz_01.png", dpi=300, bbox_inches='tight')

plt.show()
