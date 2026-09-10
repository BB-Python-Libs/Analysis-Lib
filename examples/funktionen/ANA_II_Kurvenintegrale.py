import sympy as sp

t = sp.symbols('t')

# Vektorfeld k(x, y) = (x*y**2, x*y)
x, y = sp.symbols('x y')
# k = sp.Matrix([x*y**2, x*y])
# k = sp.Matrix([y**2/2, x*y])
k = sp.Matrix([x/sp.sqrt(x**2 + y**2), y/sp.sqrt(x**2 + y**2)])

def line_integral_along(f, t_symbol, t0, t1):
    """k · dr/dt integriert von t0 bis t1"""
    r = sp.Matrix(f(t_symbol))         # r(t)
    drdt = r.diff(t_symbol)            # r'(t)
    k_on_r = k.subs({x: r[0], y: r[1]})
    integrand = k_on_r.dot(drdt)
    return sp.integrate(integrand, (t_symbol, t0, t1))

# Weg C1: f1(t) = (t, 1 - t), t ∈ [0, 2]
f1 = lambda s: (s, 1 - s)
I1 = line_integral_along(f1, t, 0, 2)

# Weg C2: f2(t) = (t, 1 - 1/2 t**2), t ∈ [0, 2]
f2 = lambda s: (s, 1 - sp.Rational(1, 2)*s**2)
I2 = line_integral_along(f2, t, 0, 2)

# Weg C3: zwei Teilkurven:
#   C31: von (0,1) nach (2,1) horizontal
#   C32: von (2,1) nach (2,-1) vertikal
f31 = lambda s: (2*s, 1)        # t von 0 bis 1
f32 = lambda s: (2, 1 - 2*s)    # t von 0 bis 1

I31 = line_integral_along(f31, t, 0, 1)
I32 = line_integral_along(f32, t, 0, 1)
I3 = sp.simplify(I31 + I32)

print("I1 =", sp.simplify(I1))
print("I2 =", sp.simplify(I2))
print("I3 =", sp.simplify(I3))
