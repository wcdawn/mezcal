import numpy as np
import matplotlib.pyplot as plt

L = 1 # [cm]
sigma_t = 2 # [1/cm]
sigma_s = 1 # [1/cm]
source = 1.0/(4.0*np.pi)

scalar_flux = lambda x : (1.0 
    - (np.exp(np.sqrt(6.0)*(1.0-x)) + np.exp(np.sqrt(6.0)*x))
    / (1.0 - 1.0/np.sqrt(2.0) + (1.0 + 1.0/np.sqrt(2.0)) * np.exp(np.sqrt(6.0))))

fname = 'out_000000.csv'
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

plt.figure()
plt.plot(x, scalar_flux(x))
plt.plot(x, fem)
plt.legend(('Analytic', 'FEM'))

plt.figure()
#plt.plot(x, error)
plt.plot(x, scalar_flux(x)/fem)
plt.title('Error')

plt.show()
