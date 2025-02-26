import numpy as np
import matplotlib.pyplot as plt

#fname = 'out_000000.csv'
fname = 'out_r5.csv'
data = np.loadtxt(fname, delimiter=',')

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

fem /= fem[0]

plt.figure()
plt.plot(x, fem)
plt.title("MEZCAL Solution")

plt.show()
