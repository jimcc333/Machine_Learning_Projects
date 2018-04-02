""" Cylindrical reactor flux visualization. """

import numpy as np
import copy
from scipy.special import j0
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calculate the grid data
radius = 5
height = 80
grid_r, grid_t, grid_z = np.mgrid[0:radius:20j, 0:np.pi*1:40j,  -height/2:height/2:8j]
constant = 100

# Perform projection to cartesian coordinates
X, Y = grid_r * np.cos(grid_t), grid_r * np.sin(grid_t)
Z = grid_z

# This is the line that calculates flux
color = constant * j0(2.405 * grid_r / radius) * np.cos(3.14159 * grid_z / height)
size = [100 for i in color]

# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

scat = ax.scatter(X, Y, Z, c=color, s=size, edgecolors='none')

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

# The following trickery is needed to scale the axes equally
mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5
number = 8
ax.set_xlim(mid_x - max_range/number, mid_x + max_range/number)
ax.set_ylim(mid_y - max_range/number, mid_y + max_range/number)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Label and show
plt.xlabel('x')
plt.ylabel('y')
plt.show()


""""
# Uncomment this block for the alternative visual
radius = 7
height = 80
grid_x, grid_y, grid_z = np.mgrid[-radius:radius:10j, -radius:radius:10j,  -height/2:height/2:10j]
constant = 100
grid_r = (grid_x ** 2 + grid_y ** 2) ** 0.5

color = constant * j0(2.405 * grid_r / radius) * np.cos(3.14159 * grid_z / height)
size = copy.copy(color)

for x in range(len(grid_x)):
    for y in range(len(grid_x[0])):
        for z in range(len(grid_x[0, 0])):
            if (grid_x[x, y, z] ** 2 + grid_y[x, y, z] ** 2) ** 0.5 > radius:
                grid_x[x, y, z] = 0
                grid_y[x, y, z] = 0
                grid_z[x, y, z] = 0
                color[x, y, z] = 0
                size[x, y, z] = 0
            else:
                size[x, y, z] = 50 + color[x, y, z]

# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

X = grid_x
Y = grid_y
Z = grid_z

scat = ax.scatter(X, Y, Z, c=color, s=size, edgecolors='none')

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5
number = 8
ax.set_xlim(mid_x - max_range/number, mid_x + max_range/number)
ax.set_ylim(mid_y - max_range/number, mid_y + max_range/number)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""
