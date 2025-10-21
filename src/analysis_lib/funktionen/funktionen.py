import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class FunktionenBibliothek:
    @staticmethod
    def plot_function_and_inverse(f_expr, f_inv_expr, x_range=(-10, 10), y_range=(-10, 10), plot_range=(-10, 10)):
        """
        Plottet eine gegebene Funktion f(x) und ihre Umkehrfunktion f⁻¹(x)
        im selben Diagramm, ergänzt durch die Winkelhalbierende y=x.
        """
        x, y = sp.symbols('x y')
        f_lambda = sp.lambdify(x, f_expr, 'numpy')
        f_inv_lambda = sp.lambdify(y, f_inv_expr, 'numpy')

        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = np.linspace(y_range[0], y_range[1], 5000)
        diag_vals = np.linspace(plot_range[0], plot_range[1], 2)

        y_vals_f = f_lambda(x_vals)
        x_vals_f_inv = f_inv_lambda(y_vals)

        plt.figure(figsize=(8, 8))
        plt.plot(x_vals, y_vals_f, label=f'$f(x) = {sp.latex(f_expr)}$', color='blue')
        plt.plot(y_vals, x_vals_f_inv, label=f'$f^{{-1}}(y) = {sp.latex(f_inv_expr)}$', color='green') 
        plt.plot(diag_vals, diag_vals, label='y = x', color='red', linestyle='--')

        plt.title('Funktion und Umkehrfunktion')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.grid(True)
        plt.legend()
        plt.show()


    @staticmethod
    def plot_multiple_functions(functions, x_symbol, x_range=(-10, 10), y_range=None, title="Vergleich mehrerer Funktionen", xlabel="x", ylabel="f(x)"):
        """
        Zeichnet mehrere Funktionen in ein gemeinsames Diagramm.

        Args:
            functions: Liste von Tupeln (Ausdruck, Label)
            x_symbol: Das Sympy-Symbol für die Variable x
            x_range: Tupel mit Start- und Endwert für die x-Achse
            y_range: Optional. Tupel mit Start- und Endwert für die y-Achse. Wenn None, wird automatisch skaliert.
            title: Titel des Plots
            xlabel: Beschriftung der x-Achse
            ylabel: Beschriftung der y-Achse
        """
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        plt.figure(figsize=(10, 6))
        
        for expr, label in functions:
            f = sp.lambdify(x_symbol, expr, "numpy")
            y_vals = f(x_vals)
            plt.plot(x_vals, y_vals, label=label)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        
        # Nur y_range setzen, wenn es angegeben ist
        if y_range is not None:
            plt.ylim(y_range)
            
        plt.legend()
        plt.tight_layout()
        plt.show()

