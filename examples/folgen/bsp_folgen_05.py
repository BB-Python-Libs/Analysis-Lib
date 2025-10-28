import numpy as np
import matplotlib.pyplot as plt

def sequence(n):
    return np.sin(n)

def plot_accumulation_points(seq_func, n_max=10000):
    n_values = np.arange(1, n_max + 1)
    y_values = seq_func(n_values)

    plt.figure(figsize=(8, 4))

    # Folgepunkte
    plt.scatter(n_values, y_values, color='blue', label='Folgenwerte $a_n$')

    # Verdichtungspunkte (bekannt aus Analyse der Folge)
    accumulation_points = np.array([-1, 1])
    plt.plot([0]*len(accumulation_points), accumulation_points, color='red', linewidth=8.0, linestyle='-', label='Verdichtungspunkte')

    # Hervorhebung der x=0 als Ort der Verdichtungspunkte
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)

    plt.title('Visualisierung von Verdichtungspunkten (Häufungspunkten)')
    plt.xlabel('Index $n$')
    plt.ylabel('$a_n$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

plot_accumulation_points(sequence)
