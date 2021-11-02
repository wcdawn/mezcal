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

T_HP = 400.0
T_B = 300.0
k = 0.1
L = 1.0
q0 = 100

q = lambda x: q0 * np.sin(np.pi * x / L)

c2 = T_HP
c1 = (T_B - T_HP) / L

T = lambda x: q0 / k * (L / np.pi) ** 2 * np.sin(np.pi * x / L) + c1 * x + c2

x = np.linspace(0.0, L)

# geometry
plt.figure(figsize=(6, 2))
ax = plt.gca()
ax.annotate("", xy=(L, 0.0), xytext=(0, 0), arrowprops=dict(arrowstyle="|-|"))

plt.text(0.5 * L, 0.1, "$q(r)$", horizontalalignment="center")
plt.text(0, -0.1, "$x=0$", horizontalalignment="center")
plt.text(L, -0.1, "$x=L$", horizontalalignment="center")
plt.text(0, 0.1, "$T_{HP}$", horizontalalignment="center")
plt.text(L, 0.1, "$T_B$", horizontalalignment="center")

plt.axis("equal")
plt.axis("off")
plt.xlim((-0.1 * L, 1.1 * L))
plt.ylim((-1, 1))
plt.tight_layout()
plt.savefig(os.path.join(dirname, "../figs/linear_geometry.pdf"))
plt.close()

# solution
plt.figure()
plt.plot(x, T(x))
plt.xlabel("x")
plt.ylabel("T(x)")
plt.title("One-Dimension Cartesian Heat Conduction Solution")
plt.xlim((0.0, L))
plt.savefig(os.path.join(dirname, "../figs/linear.pdf"))
plt.close()
