import sympy as sp

x, y = sp.symbols('x y')
expr = sp.exp(y-1) + y - x**3

sp.plot_implicit(sp.Eq(expr, 0), (x, -2, 2), (y, -3, 4),
                 title='Implizite Kurve: $e^{y-1}+y-x^3=0$')

p = sp.plot(1/x, (x, -5, 5), title='Kehrwert-Funktion f(x)=$\\frac{1}{x}$ mit Lücken', ylabel='f(x)', xlabel='x', show=False)

