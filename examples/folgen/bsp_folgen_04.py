import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 2

def bisection_visualization(a, b, tol=1e-4, max_iter=8):
    intervals = [(a, b)]
    for _ in range(max_iter):
        c = (a + b) / 2
        if f(c) == 0 or (b - a) / 2 < tol:
            break
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        intervals.append((a, b))
    
    # Visualisierung
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (lower, upper) in enumerate(intervals):
        ax.plot([i, i], [lower, upper], 'bo-', linewidth=4)
        ax.plot(i, lower, 'ro')  # Untere Intervallgrenze (rot)
        ax.plot(i, upper, 'go')  # Obere Intervallgrenze (grün)

    ax.set_title('Intervallschachtelung beim Bisektionsverfahren zur Berechnung von sqrt(2)')
    ax.set_xlabel('Iterationsschritt')
    ax.set_ylabel('Intervallgrenzen')
    ax.grid(True)
    plt.show()

def bisection_1d_visualization(a, b, tol=1e-4, max_iter=6):
    intervals = [(a, b)]
    for _ in range(max_iter):
        c = (a + b) / 2
        if f(c) == 0 or (b - a) / 2 < tol:
            break
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        intervals.append((a, b))

    fig, ax = plt.subplots(figsize=(10, 2))
    y_pos = 1  # konstanter y-Wert, damit alles auf einer Linie liegt

    for i, (lower, upper) in enumerate(intervals):
        ax.plot([lower, upper], [y_pos, y_pos], 'b-', linewidth=7, solid_capstyle='round')
        ax.text(lower, y_pos + 0.03, f'{lower:.5f}', ha='center', fontsize=9, color='red')
        ax.text(upper, y_pos + 0.03, f'{upper:.5f}', ha='center', fontsize=9, color='green')
        ax.text((lower + upper)/2, y_pos - 0.05, f'Schritt {i}', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylim(0.8, 1.2)
    ax.set_yticks([])
    ax.set_xlabel('reelle Achse (Intervall)')
    ax.set_title('Eindimensionale Visualisierung der Intervallschachtel beim Bisektionsverfahren')

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x):
    return x**2 - 2

def bisection_steps(a, b, tol=1e-4, max_iter=8):
    steps = [(a, b)]
    for _ in range(max_iter):
        c = (a + b) / 2
        if f(c) == 0 or (b - a) / 2 < tol:
            break
        if f(c)*f(a) < 0:
            b = c
        else:
            a = c
        steps.append((a, b))
    return steps

def simple_steps(a, b, tol=1e-4, max_iter=8):
    steps = [(a, b)]
    for _ in range(max_iter):
        a = a + (b - a) / 10
        b = b - (b - a) / 10
        steps.append((a, b))
    return steps

def animate_bisection(steps):
    fig, ax = plt.subplots(figsize=(10, 2))
    y_pos = 1
    line, = ax.plot([], [], 'b-', linewidth=7, solid_capstyle='round')
    txt_lower = ax.text(0, 0, '', ha='center', color='red', fontsize=9)
    txt_upper = ax.text(0, 0, '', ha='center', color='green', fontsize=9)
    txt_step = ax.text(0.5, 0.85, '', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

    ax.set_ylim(0.7, 1.3)
    ax.set_xlim(steps[0][0] - 0.1, steps[0][1] + 0.1)
    ax.set_yticks([])
    ax.set_xlabel('reelle Achse (Intervall)')
    ax.set_title('Animation der Intervallschachtel beim Bisektionsverfahren')

    def update(frame):
        lower, upper = steps[frame]
        line.set_data([lower, upper], [y_pos, y_pos])
        txt_lower.set_position((lower, y_pos + 0.03))
        txt_lower.set_text(f'{lower:.5f}')
        txt_upper.set_position((upper, y_pos + 0.03))
        txt_upper.set_text(f'{upper:.5f}')
        txt_step.set_text(f'Schritt {frame}')
        return line, txt_lower, txt_upper, txt_step

    ani = FuncAnimation(fig, update, frames=len(steps), interval=1000, blit=True, repeat=False)
    plt.show()

def subplot_bisection(steps):
    n_steps = len(steps)
    cols = 3
    rows = (n_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    y_pos = 1  # konstanter y-Wert

    for i, (lower, upper) in enumerate(steps):
        ax = axes[i]
        ax.plot([lower, upper], [y_pos, y_pos], 'b-', linewidth=7, solid_capstyle='round')
        ax.plot(lower, y_pos, 'ro')
        ax.plot(upper, y_pos, 'go')
        ax.set_ylim(0.7, 1.3)
        ax.set_xlim(steps[0][0] - 0.1, steps[0][1] + 0.1)
        ax.set_yticks([])
        ax.set_title(f'Schritt {i}')
        ax.set_xlabel('reelle Achse')
        if i % cols == 0:
            ax.set_ylabel('Intervallgrenzen')
        ax.grid(True, linestyle='--', alpha=0.5)

    # Unbenutzte Subplots ausblenden
    for j in range(n_steps, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Beispielaufruf
steps = bisection_steps(1, 2)
animate_bisection(steps)
subplot_bisection(steps)

steps = simple_steps(1, 2)
animate_bisection(steps)
subplot_bisection(steps)

# Startintervall [1, 2]
# bisection_1d_visualization(1, 2)


# Startwerte: Intervall, in dem sqrt(2) liegt (z.B. 1 bis 2)
bisection_visualization(1, 2)
