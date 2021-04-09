import numpy as np
import matplotlib.pyplot as plt
import re

iteration = \
    re.compile('^\s*(\d+)\sEPS nconv=0 first unconverged value \(error\) ((\d|\.|\+|\-|i)*)\s\((.*)\)')

fname = 'out.790850'
file_data = open(fname, 'r').readlines()

it = []
eig = []
err = []

for line in file_data:
    entry = iteration.match(line)
    if (entry is not None):
        it.append(int(entry.group(1)))

        eig_tmp = entry.group(2)
        eig_tmp = eig_tmp.replace('i', 'j')
        eig_tmp = complex(eig_tmp)
        eig.append(abs(eig_tmp)) # only use magnitude for imaginary numbers

        err.append(float(entry.group(4)))

plt.figure()
plt.semilogy(it, err)
plt.title('Convergence Error')
plt.xlabel('Iteration')
plt.ylabel('Error')

plt.figure()
plt.plot(it, eig)
plt.title('Convergence Eigenvalue')
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue')

plt.show()
