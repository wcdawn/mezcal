import numpy as np
import matplotlib.pyplot as plt

def get_flux(fname):

    data = np.loadtxt(fname, delimiter=",")

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

    fem /= fem[0] # normalize

    return x, fem

if __name__ == "__main__":

    eval_pts = np.array([0.25, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    dat = np.zeros((len(eval_pts),6))

    for r in range(0,6):

        x, y = get_flux("out_r{:d}.csv".format(r))

        # note: there are four entries per coordinate b/c this is a weird collapsed 2d mesh
        # the good news is that after checking the flux at each of the four entries is exaxtly the same to the printed precision

        npts = len(x)

        for i in range(len(eval_pts)):
            point = int(npts*(eval_pts[i]/3.0))
            if (point == npts):
                point = -1
            dat[i,r] = y[point]

    for i in range(len(eval_pts)):
        print(eval_pts[i]) 
        for r in range(0,6):
            print(dat[i,r])

        print()
