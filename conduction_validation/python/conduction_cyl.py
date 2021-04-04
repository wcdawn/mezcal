import os
import numpy as np
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)

LW = 2
FN = 'Times New Roman'
FS = 16
plt.rc('lines', lw=LW)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family=FN, size=FS)

T_HP = 400.
T_B = 300.
k = 0.1
R_HP = 0.5
R_B = 2.0
q0 = 100.

q = lambda r : q0 * (R_B - x) / (R_B - R_HP)

c1 = 1.0/np.log(R_HP/R_B) * (
    (T_HP-T_B)*(R_B-R_HP)*k/q0 - R_HP**3/9. + 0.25*R_B*R_HP**2 - 5./36.*R_B**3)
c2 = T_B * k/q0 * (R_B-R_HP) + 5./36.*R_B**3 - c1*np.log(R_B)

T = lambda r : q0/k * (1.0/(R_B-R_HP)) * (r**3/9. - 0.25*R_B*r**2 + c1*np.log(r) + c2)

r = np.linspace(R_HP, R_B, 1000)

plt.figure(figsize=(6,2))
ax = plt.gca()
ax.annotate("", xy=(R_B*1.1, 0.0), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->'))
plt.plot((R_HP, R_HP), (-0.025, 0.025), '-k', lw=2)
plt.plot((R_B, R_B), (-0.025, 0.025), '-k', lw=2)
plt.plot((0.0,0.0), (-0.025, 0.025), '-k', lw=2)

plt.text(0.0, -0.1, '$r=0$', horizontalalignment='center')
plt.text(R_HP, -0.1, '$r=R_{HP}$', horizontalalignment='center')
plt.text(R_B, -0.1, '$r=R_{B}$', horizontalalignment='center')

plt.text(R_HP, 0.1, '$T_{HP}$', horizontalalignment='center')
plt.text(R_B, 0.1, '$T_B$', horizontalalignment='center')

plt.text(R_HP*0.5, 0.25, 'Heat\nPipe', horizontalalignment='center')
plt.text((R_HP+R_B)*0.5, 0.1, '$q(r)$', horizontalalignment='center')

plt.axis('equal')
plt.axis('off')
plt.xlim((-0.1,1.25*R_B))
plt.tight_layout()
plt.savefig(os.path.join(dirname, '../figs/radial_geometry.pdf'))
plt.close()

plt.figure()
p = plt.plot(r, T(r))
plt.plot((0,R_HP), (T_HP,T_HP), c=p[0].get_color())
yl = plt.ylim()
plt.plot((R_HP,R_HP), yl, '-k', lw=1)
plt.plot((R_B,R_B), yl, '-k', lw=1)

plt.text(R_HP*0.5, np.mean(yl)+5, 'Heat Pipe', horizontalalignment='center', size=14)

plt.xlim((0.0, R_B*1.1))
plt.ylim(yl)

plt.xlabel('r')
plt.ylabel('T(r)')
plt.title('Radial Heat Conduction Solution')
plt.savefig(os.path.join(dirname, '../figs/radial_line.pdf'))
plt.close()

N_GRID = 500
delta = 2*R_B/N_GRID
sol = np.zeros((N_GRID,N_GRID))

for i in range(N_GRID):
    for j in range(N_GRID):
        x_center = (i+0.5)*delta - R_B
        y_center = (j+0.5)*delta - R_B
        r = np.sqrt(x_center**2 + y_center**2)
        if (r > R_B):
            sol[i,j] = np.NaN
        elif (r < R_HP):
            sol[i,j] = T_HP
        else:
            sol[i,j] = T(r)

radian = np.linspace(0.0, 2*np.pi)

fig = plt.figure()
pos = plt.imshow(sol, extent=(-R_B, R_B, -R_B, R_B))
cbar = fig.colorbar(pos)

plt.plot(R_HP*np.cos(radian), R_HP*np.sin(radian), '-k', lw=1)

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('T(r)', rotation=270)
plt.title('Radial Heat Conduction Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(os.path.join(dirname, '../figs/radial_circle.pdf'))
plt.close()
