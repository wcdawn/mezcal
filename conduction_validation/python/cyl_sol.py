import numpy as np
import sympy as sp

r, R_HP, R_B, T_HP, T_B = sp.symbols("r R_HP R_B T_HP T_B")
q0, k = sp.symbols("q0 k")
c1, c2 = sp.symbols("c1 c2")

q = q0 * (R_B - r) / (R_B - R_HP)

int_with_constants = -1 / k * (sp.integrate((sp.integrate(q * r, r) + c1) / r, r) + c2)
print(sp.cxxcode(int_with_constants, standard="C++11"))

dumb = sp.linsolve(
    [
        sp.Eq(int_with_constants.subs(r, R_HP), T_HP),
        sp.Eq(int_with_constants.subs(r, R_B), T_B),
    ],
    (c1, c2),
)

print("c1")
print(sp.cxxcode(dumb.simplify().args[0], standard="C++11"))
