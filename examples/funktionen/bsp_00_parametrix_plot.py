import numpy as np
import matplotlib.pyplot as plt

# Beispiel-Parametrisierung (Lissajous-Kurve, lässt sich anpassen)
t = np.linspace(-1/4, 1/4, 500)
x = np.sin(np.pi*t)
y = np.cos(2*np.pi*t)

plt.figure(figsize=(7,5))
plt.plot(x, y, label=r'$x=\sin(\pi t)$, $y=\cos(2\pi t)$, $t\in [-\frac{1}{4}, \frac{1}{4}]$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("function_parametric_plot.png", dpi=300, bbox_inches='tight')
plt.show()
