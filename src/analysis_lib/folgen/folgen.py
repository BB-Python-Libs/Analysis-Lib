import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class FolgenBibliothek:
    @staticmethod
    def symbolische_tabelle(expr, n_symbol, n_start=1, n_end=10):
        print(f"
Symbolische Darstellung für a_n = {sp.pretty(expr)}")
        for n in range(n_start, n_end + 1):
            print(f"a_{n} =", expr.subs(n_symbol, n))

    @staticmethod
    def numerische_tabelle(expr, n_symbol, n_start=1, n_end=10):
        print("
Numerische Tabelle der Folge:")
        for n in range(n_start, n_end + 1):
            print(f"a_{n} = {expr.subs(n_symbol, n).evalf():.10f}")

    @staticmethod
    def plot_folge(expr, n_symbol, n_start=1, n_end=20):
        values = [float(expr.subs(n_symbol, n).evalf()) for n in range(n_start, n_end + 1)]
        plt.figure(figsize=(8, 4))
        plt.plot(range(n_start, n_end + 1), values, "o-")
        plt.title("Folge a_n gegen Index n")
        plt.xlabel("Index n")
        plt.ylabel("Wert a_n")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
