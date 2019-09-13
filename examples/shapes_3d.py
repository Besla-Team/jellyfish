from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'size':10, 'family':'serif'}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111, projection='3d')

# Make data ellipsoid
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 73 * np.outer(np.cos(u), np.sin(v)) * 10
y = 37.46 * np.outer(np.sin(u), np.sin(v)) * 10 
z = 37.86 * np.outer(np.ones(np.size(u)), np.cos(v)) * 10


# Principal axis plot
pa = np.array([[-0.71037951,  0.5109366 , -0.48405035],
               [ 0.00175195,  0.68903005,  0.72473065],
               [-0.70381665, -0.51398577,  0.49036797]])*100


ax.plot([0, pa[0,0]], [0, pa[0,1]], [0, pa[0,2]])
ax.plot([0, pa[1,0]], [0, pa[1,1]], [0, pa[1,2]])
ax.plot([0, pa[2,0]], [0, pa[2,1]], [0, pa[2,2]])


ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)
ax.set_zlim(-400, 400)
# Rotate ellipsoid

RM_euler = np.array([[ 0.49036797, -0.72473065,  0.48405035],  [ 0.51398577, -0.20804721, -0.83218687],  [ 0.70381665,  0.65687278,  0.27048156]])
xyz_rot = np.dot(np.array([x.flatten(), y.flatten(), z.flatten()]).T, pa/300.)
# Plot the surface
#ax.plot_surface(x, y, z, color='b', alpha=0.4)
ax.plot_surface(xyz_rot[:,0].reshape(100, 100), xyz_rot[:,1].reshape(100, 100),
                xyz_rot[:,2].reshape(100, 100), color='r', alpha=0.4)

# plot points

halo = np.loadtxt('rot_halo_1000.txt')
ax.scatter(halo[:,0], halo[:,1], halo[:,2], c='k', marker='.', s=0.2,  edgecolors="", facecolor='1')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()



