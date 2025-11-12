import numpy as np
import matplotlib.pyplot as plt

# Beispiel-Funktion
def f(x):
    return (x - 1)**2 + 1

a, b = -0.5, 3
x_vals = np.linspace(-0.5, 3.5, 400)
y_vals = f(x_vals)

x1, x2 = 0.5, 2.5
lambda_ = 0.4
y1, y2 = f(x1), f(x2)

x = lambda_ * x1 + (1 - lambda_) * x2
fx = f(x)
sek_y = lambda_ * y1 + (1 - lambda_) * y2  # Punkt auf Sekante
sek_x = [x1, x2]
sek_yvals = [y1, y2]

plt.figure(figsize=(9, 6))
plt.plot(x_vals, y_vals, c='navy', label=r'$f(x)$')
plt.plot(sek_x, sek_yvals, 'r--', linewidth=2, label='Sekante')
plt.scatter([x1, x2], [y1, y2], c='red', zorder=5)
plt.scatter([x], [fx], c='navy', s=100, edgecolors='black', zorder=6, label=r'$f(\lambda x_1 + (1-\lambda) x_2)$')
plt.scatter([x], [sek_y], c='orange', s=100, edgecolors='black', zorder=6, label=r'$\lambda f(x_1) + (1-\lambda)f(x_2)$')

# Hilfslinien
plt.vlines([x1, x2, x], ymin=0, ymax=[y1, y2, sek_y], color='grey', linestyles='dotted')
plt.hlines([y1, y2, sek_y, fx], xmin=[0, 0, 0, 0], xmax=[x1, x2, x, x], color='grey', linestyles='dotted')

# Intervall [a,b] auf der x-Achse
plt.hlines(y=0, xmin=a, xmax=b, color='brown', linewidth=5, alpha=0.5, label='Intervall $[a,b]$')
plt.text(a, min(y_vals)-1, '$a$', ha='center', va='top', fontsize=12, color='brown')
plt.text(b, min(y_vals)-1, '$b$', ha='center', va='top', fontsize=12, color='brown')

# Koordinatenkreuz und Achsenpfeile
plt.axhline(0, color='black', linewidth=1.0)
plt.axvline(0, color='black', linewidth=1.0)
plt.annotate('', xy=(3.5, 0), xytext=(a, 0), arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
plt.annotate('', xy=(0, max(y_vals)+1), xytext=(0, min(y_vals)-1), arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

plt.text(x1 - 0.2, y1 - 0.35, '$f(x_1)$', color='red')
plt.text(x2 + 0.05, y2 - 0.1, '$f(x_2)$', color='red')
plt.text(x + 0.05, fx - 0.4, r'$f(\lambda x_1 + (1-\lambda) x_2)$', color='navy')
plt.text(x - 0.6, sek_y + 0.3, r'$\lambda f(x_1) + (1-\lambda)f(x_2)$', color='orange')
plt.text(x1 - 0.03, -0.3, '$x_1$', color='grey')
plt.text(x2 - 0.03, -0.3, '$x_2$', color='grey')
plt.text(x - 0.25, -0.3, r'$\lambda x_1 + (1-\lambda) x_2$', color='grey')

plt.xlim(a-0.7, b+0.7)
plt.ylim(min(y_vals)-2, max(y_vals)+1.5)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.grid(True, alpha=0.25)
plt.legend(loc='upper left')
plt.title("Koordinatenkreuz und Intervall $[a, b]$ für Konvexitäts-Definition")

plt.tight_layout()
plt.savefig("konvexitaet_schaubild.png", dpi=300, bbox_inches='tight')
plt.show()
