import numpy as np
import scipy.sparse


# base_fname is the stub of the file name before the .00000 or INFO or whatever
# this is so that the reader will work with parallel output
def read_vector(base_fname):

    x = np.ndarray((0,0))

    idx = 0
    while (True):
        this_data_fname = base_fname + "." + str(idx)
        this_info_fname = base_fname + ".INFO." + str(idx)

        try:
            fid = open(this_data_fname, 'r')
            fid.close()
            idx += 1
        except:
            break

        info = np.loadtxt(this_info_fname).astype(int)
        data = np.loadtxt(this_data_fname, skiprows=1)

        if (idx == 1):
            x = np.resize(x, (info[0]))

        start = info[1]
        end = info[2]

        x[start:end] = data
    
    return x

# base_fname is the stub of the file name before the 00000 or whatever
# this is so that the reader will work with parallel output
def read_matrix(base_fname):

    i = np.ndarray(0)
    j = np.ndarray(0)
    x = np.ndarray(0)

    nrow = 0
    ncol = 0
    
    idx = 0
    while (True):
        this_fname = '{:s}.{:05d}'.format(base_fname, idx)

        try:
            fid = open(this_fname, 'r')
            idx += 1
        except:
            break

        data = np.loadtxt(this_fname, skiprows=1)

        info = fid.readline().split()
        fid.close()

        nrow = max((nrow, int(info[1])))
        ncol = max((nrow, int(info[3])))

        i = np.append(i, data[:,0].astype(int))
        j = np.append(j, data[:,1].astype(int))
        x = np.append(x, data[:,2])

    return scipy.sparse.csr_matrix((x, (i, j)), shape=(nrow+1,ncol+1))
