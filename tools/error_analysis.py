import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("out.csv", delimiter=',')

# format
# x, y, z, fem, analytic, error

x        = data[:,0]
y        = data[:,1]
z        = data[:,2]
fem      = data[:,3]
analytic = data[:,4]
error    = data[:,5]

sort_mask = np.argsort(x)

x        = x[sort_mask]
y        = y[sort_mask]
z        = z[sort_mask]
fem      = fem[sort_mask]
analytic = analytic[sort_mask]
error    = error[sort_mask]

plt.figure()
plt.plot(x, fem)
plt.plot(x, analytic)

plt.legend(["FEM", "Analytic"])

ax1 = plt.gca()
ax2 = ax1.twinx()

ax2.plot(x, error)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(x, y, fem)

plt.show()
