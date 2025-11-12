import sympy as sp

x, y = sp.symbols('x y')
expr = sp.exp(y-1) + y - x**3

sp.plot_implicit(sp.Eq(expr, 0), (x, -2, 2), (y, -3, 4),
                 title='Implizite Kurve: $e^{y-1}+y-x^3=0$')
