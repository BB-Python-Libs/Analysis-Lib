import numpy as np
import matplotlib.pyplot as plt

# 1. x-Werte definieren (sehr fein aufgelöst für glatte Kurven)
# Wir starten bei 1e-5 statt 0, um Division durch Null bei 1/x zu vermeiden
x = np.linspace(1e-5, 1, 5000)

# 2. n-Bereich festlegen (wie im Maple-Befehl n=5..20)
n_values = range(5, 21)

plt.figure(figsize=(10, 6))

# Maple benutzt oft eine kontrastreiche Farbskala ("Jet" oder ähnlich)
# Wir nutzen hier 'jet', um den Look des Bildes nachzuahmen
colors = plt.cm.jet(np.linspace(0, 0.9, len(n_values)))

for i, n in enumerate(n_values):
    # Die Funktion stückweise berechnen:
    # Wenn x < 1/n, dann y = n
    # Sonst y = 1/x
    y = np.where(x < 1/n, n, 1/x)
    
    plt.plot(x, y, color=colors[i], linewidth=1.5, label=f'n={n}')

# 3. Achsen und Limits wie im Original
plt.xlim(0, 1)
plt.ylim(0, 21)

plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Plot der Folge $f_n(x) = \min(n, 1/x)$ für $n=5..20$')

# Gitter optional (im Originalbild nicht direkt sichtbar, aber hilfreich)
# plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("Konvergenz_FF_01.png", dpi=300, bbox_inches='tight')

plt.show()
