import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

LW = 2
FN = 'Times New Roman'
FS = 14
plt.rc('lines', lw=LW)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family=FN, size=FS)

cmap = mpl.colormaps['viridis']

def hex_coord(ring, around, ppitch):

    if ring == 0:
        return 0.0, 0.0

    p2p = 2 * ring * ppitch  # point-to-point
    p = 0.5 * p2p
    a = p * np.sqrt(3.0)  # side length

    hex_coord = np.array(
        [
            [p, 0.0],
            [0.5 * p, 0.5 * p * np.sqrt(3)],
            [-0.5 * p, 0.5 * p * np.sqrt(3)],
            [-p, 0.0],
            [-0.5 * p, -0.5 * p * np.sqrt(3)],
            [0.5 * p, -0.5 * p * np.sqrt(3)],
        ]
    )

    hex_dxdy = np.zeros_like(hex_coord)
    for i in range(6):
        hex_dxdy[i, :] = hex_coord[(i + 1) % 6, :] - hex_coord[i, :]

    base_idx = int(around / ring)
    frac_around = (around % ring) / ring

    x = hex_coord[base_idx, 0] + frac_around * hex_dxdy[base_idx, 0]
    y = hex_coord[base_idx, 1] + frac_around * hex_dxdy[base_idx, 1]

    return x, y

R_F = 1.486
pin_pitch = 3.5

hex_coord_pair = np.array([
    [1, 0],
    [1, 1],
    [2, 0],
    [2, 1],
    [2, 2],
    [2, 3],
    [3, 0],
    [3, 1],
    [3, 2],
    [3, 3],
    [3, 4]
    ])

hex_coord_val = np.zeros((hex_coord_pair.shape[0],2))

for i in range(hex_coord_pair.shape[0]):
    x, y = hex_coord(hex_coord_pair[i,0], hex_coord_pair[i,1], pin_pitch)
    hex_coord_val[i,:] = [x,y]
    print('pin {:.8f} {:.8f} {:.3f}'.format(x, y, R_F))

pin_power = np.ones(11); # TODO results
pin_power = np.array((26.6459, 53.2993, 26.7467, 53.4756, 53.581, 26.7169,
    27.8531, 55.1222, 55.3903, 56.3578, 55.0074))
factor = np.array([2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

for i in range(len(pin_power)):
    pin_power[i] *= factor[i]

pin_power /= pin_power.mean()

circle_list = []
for i in range(hex_coord_val.shape[0]):
    fraction = (pin_power[i]-pin_power.min()) / (pin_power.max() - pin_power.min())
    c = cmap(fraction, 0.8)
    print(fraction, c)
    circle_list.append(plt.Circle(hex_coord_val[i,:], R_F, edgecolor='black',
        linewidth=2, facecolor=c))

plt.figure()
ax = plt.gca()
for c in circle_list:
    ax.add_artist(c)
for i in range(len(pin_power)):
    plt.text(hex_coord_val[i,0], hex_coord_val[i,1],
            '{:.4f}'.format(pin_power[i]), horizontalalignment='center',
            verticalalignment='center')
plt.axis('square')
plt.ylim((-1.75, 10.5+1.75))
plt.xlim((-1.75, 10.5+1.75))
plt.axis('off')
plt.tight_layout()
plt.show()
