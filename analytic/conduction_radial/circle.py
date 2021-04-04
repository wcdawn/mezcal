import numpy as np
import matplotlib.pyplot as plt

R_HP = 0.5
R_B  = 2.0

heat_pipe_attribute = 1
heat_source_attribute = 2

VTK_CELL_TYPE = 22

out = []
out.append('# vtk DataFile Version 2.0\n')
out.append('curvilinear heat pipe heat conduction mesh\n')
out.append('ASCII\n')
out.append('DATASET UNSTRUCTURED_GRID\n')

node = []

# hexagon
node.append(np.array([0,0,0]))

# outer hexagon
radian = 0.5*np.pi
for i in range(6):
    node.append(np.array([R_HP*np.cos(radian), R_HP*np.sin(radian), 0]))
    radian += np.pi / 3

# outer (offset) hexagon
radian = 0.5*np.pi + np.pi/6
for i in range(6):
    node.append(np.array([R_HP*np.cos(radian), R_HP*np.sin(radian), 0]))
    radian += np.pi / 3

# inner
radian = 0.5*np.pi
for i in range(6):
    node.append(np.array([0.5*R_HP*np.cos(radian), 0.5*R_HP*np.sin(radian), 0]))
    radian += np.pi / 3

# outer R_B hexagon
radian = 0.5*np.pi + np.pi / 6
for i in range(6):
    node.append(np.array([R_B*np.cos(radian), R_B*np.sin(radian), 0]))
    radian += np.pi / 3

# outer R_B hexagon
radian = 0.5*np.pi + np.pi/3
for i in range(6):
    node.append(np.array([R_B*np.cos(radian), R_B*np.sin(radian), 0]))
    radian += np.pi / 3

# in between
for i in range(6):
    j = i+18+1
    node.append(0.5*(node[j] + node[j-18]))
    node.append(0.5*(node[j] + node[(j-18)%6+1]))

material = []
element = []
for i in range(6):
    j = i+1
    element.append([0, j, j%6+1, j+12, j+6, j%6+13])
    material.append(heat_pipe_attribute)

for i in range(6):
    j = i+1
    element.append([j, j+18, (j%6)+1, 2*j+29, 2*j+30, j+6]) # points up
    element.append([(j%6)+1, j+18, (j%6)+19, 2*j+30, j+24, 2*(j%6)+31]) # points down
    material.append(heat_source_attribute)
    material.append(heat_source_attribute)

out.append('\n')
out.append('POINTS {:d} double\n'.format(len(node)))
count = 0
for point in node:
    count += 1
    for val in point:
        out.append('{:13.6E} '.format(val))
    else:
        out.append('\n')

out.append('\n')
out.append('CELLS {:d} {:d}\n'.format(len(element), len(element)*(len(element[0])+1)))
for elem in element:
    out.append('{:d} '.format(len(elem)))
    for val in elem:
        out.append('{:d} '.format(val))
    else:
        out.append('\n')

out.append('\n')
out.append('CELL_TYPES {:d}\n'.format(len(element)))
for i in range(len(element)):
    out.append('{:d}\n'.format(VTK_CELL_TYPE))

out.append('\n')
out.append('CELL_DATA {:d}\n'.format(len(material)))
out.append('SCALARS material int 1\n')
out.append('LOOKUP_TABLE default\n')
for mat in material:
    out.append('{:d}\n'.format(mat))

fid = open('circle.vtk', 'w')
fid.writelines(out)
fid.close()
