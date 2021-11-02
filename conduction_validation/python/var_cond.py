import os
import numpy as np
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)

LW = 2
FN = "Times New Roman"
FS = 16
plt.rc("lines", lw=LW)
plt.rc("mathtext", fontset="stix")
plt.rc("font", family=FN, size=FS)

q0 = 200
k0 = 10
alpha = 1e-2

T0 = 600.0
TL = 300.0
L = 5

c2 =  0.5 / alpha * (k0 ** 2 - (alpha * T0 - k0) ** 2) - q0 * (2 * L / np.pi) ** 2
c1 = 1.0 / L * (0.5 / alpha * (k0 ** 2 - (alpha * TL - k0) ** 2) - c2)
T = (
    lambda x: (
        k0
        - np.sqrt(
            k0 ** 2
            - 2
            * alpha
            * (q0 * (2 * L / np.pi) ** 2 * np.cos(0.5 * np.pi * x / L) + c1 * x + c2)
        )
    )
    / alpha
)

k = lambda x: k0 - alpha * T(x)

x = np.linspace(0, L, 4096)

Tbar = np.trapz(T(x), x) / L
kbar = k0 - alpha * Tbar

print(Tbar, kbar)
print('{:.6f}'.format(kbar))

T_hat = (
    lambda x: (q0 / kbar)
    * (2.0 * L / np.pi) ** 2
    * (np.cos(0.5 * np.pi * x / L) + x/L - 1.0)
    + (TL - T0) / L * x
    + T0
)

plt.figure()
plt.plot(x, T(x))
plt.plot(x, T_hat(x))
plt.legend(("Variable", "Constant"))
plt.xlabel("$x$")
plt.ylabel("$T(x)$")
plt.xlim((0, L))
plt.title("Temperature")
plt.tight_layout()
plt.savefig(os.path.join(dirname, '../figs/var_cond_temperature.pdf'))

plt.figure()
plt.plot(x, k(x))
plt.xlabel("$x$")
plt.ylabel("$k(x)$")
plt.xlim((0, L))
plt.title("Thermal Conductivity")
plt.tight_layout()
plt.savefig(os.path.join(dirname, '../figs/var_cond_conductivity.pdf'))

plt.figure()
plt.plot(x, T(x) - T_hat(x))
plt.xlabel("x")
plt.ylabel("$T(x) - \\hat{T}(x)$")
plt.xlim((0, L))
plt.title("Temperature Difference")
plt.tight_layout()
plt.savefig(os.path.join(dirname, '../figs/var_cond_diff.pdf'))

#plt.show()
