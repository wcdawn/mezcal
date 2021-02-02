import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import re

def petsc_to_coord(fname):
    file_data = open(fname, 'r').readlines()

    row = re.compile('^\s*row\s+(\d+)')
    record = re.compile('\((\d+),\s*([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?)\)')

    row_ind = []
    col_ind = []
    data = []

    last_row = 0

    for line in file_data:
        entry = row.match(line)
        if (entry is not None) :
            this_row = int(entry.group(1))
            last_row = this_row
            records = record.finditer(line)
            for rec in records:
                this_col = int(rec.group(1))
                this_val = float(rec.group(2))
                row_ind.append(this_row)
                col_ind.append(this_col)
                data.append(this_val)

    shape = (last_row+1, max(col_ind)+1)

    return row_ind, col_ind, data, shape

row_ind, col_ind, data, shape = petsc_to_coord('./petsca_vector')
A = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)

print('A nnz {:d}'.format(A.getnnz()))
print('A sparsity {:.2f}%'.format(A.getnnz() / (A.get_shape()[0] * A.get_shape()[1])))

row_ind, col_ind, data, shape = petsc_to_coord('./petscm_vector')
square = (max(shape), max(shape))
M = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=square)

print('M nnz {:d}'.format(M.getnnz()))
print('M sparsity {:.2f}%'.format(M.getnnz() / (M.get_shape()[0] * M.get_shape()[1])))


file_data = np.loadtxt('hypre_lobpcg_a.00000', skiprows=1, delimiter=' ')
row_ind = file_data[:,0].astype(int)
col_ind = file_data[:,1].astype(int)
data    = file_data[:,2].astype(float)
A_hypre = scipy.sparse.csr_matrix((data, (row_ind, col_ind)))

print('A_hypre nnz {:d}'.format(A_hypre.getnnz()))
print('A_hypre sparsity {:.2f}%'.format(A_hypre.getnnz() / (A_hypre.get_shape()[0] * A_hypre.get_shape()[1])))

file_data = np.loadtxt('hypre_lobpcg_m.00000', skiprows=1, delimiter=' ')
row_ind = file_data[:,0].astype(int)
col_ind = file_data[:,1].astype(int)
data    = file_data[:,2].astype(float)
M_hypre = scipy.sparse.csr_matrix((data, (row_ind, col_ind)))

print('M_hypre nnz {:d}'.format(M_hypre.getnnz()))
print('M_hypre sparsity {:.2f}%'.format(M_hypre.getnnz() / (M_hypre.get_shape()[0] * M_hypre.get_shape()[1])))

print(A_hypre[0,0])
print(A[0,0])

if (square[1] < 1000):

    plt.figure()
    plt.imshow((A-A_hypre).toarray())
    plt.title("A difference")
    plt.colorbar()

    plt.figure()
    plt.imshow((M-M_hypre).toarray())
    plt.title("M difference")
    plt.colorbar()

    w, vr = scipy.linalg.eig(A.toarray(), M.toarray())
    keff_petsc = 1.0 / np.abs(w).min()
    print("keff_petsc = {:.16f}".format(keff_petsc))
    w, vr = scipy.linalg.eig(A_hypre.toarray(), M_hypre.toarray())
    keff_hypre = 1.0 / np.abs(w).min()
    print("keff_hypre = {:.16f}".format(keff_hypre))
    print("diff [pcm] = {:.6f}".format((keff_petsc - keff_hypre)*1e5))

    plt.show()
