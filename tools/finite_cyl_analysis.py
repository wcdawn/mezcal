import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("../src/out.csv", delimiter=',')

# format
# x, y, z, fem, analytic, error

x        = data[:,0]
y        = data[:,1]
z        = data[:,2]
fem      = data[:,3]
analytic = data[:,4]
error    = data[:,5]

r = np.sqrt(x**2 + y**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(r, z, analytic, cmap=plt.get_cmap('viridis'))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(r, z, fem, cmap=plt.get_cmap('viridis'))

plt.show()
