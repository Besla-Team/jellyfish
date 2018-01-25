"""
Compute the Angular Momentun, spin parameter,
anisotropy parameter, potential and kinetic 
energy of the halo.
"""

import numpy as np
from scipy import linalg

def angular_momentum(xyz, vxyz, M):

    """
    Computes the angular momentum of the DM halo.
    """

    r_c_p = ([np.cross(xyz[i], vxyz[i]) for i in range(len(xyz))])
    J_x_i = np.zeros(len(r_c_p))
    J_y_i = np.zeros(len(r_c_p))
    J_z_i = np.zeros(len(r_c_p))
    for i in range(len(r_c_p)):
        J_x_i[i] = r_c_p[i][0]
        J_y_i[i] = r_c_p[i][1]
        J_z_i[i] = r_c_p[i][2]
    J_x = np.sum(J_x_i)
    J_y = np.sum(J_y_i)
    J_z = np.sum(J_z_i)
    M_tot = np.sum(M)*1E10
    J = np.array([J_x, J_y, J_z])
    J_t = np.dot(M_tot, J)
    return J_t

def spin_param(J, M, xyz):

    """
    Spin parameter:
    """
    J_n = linalg.norm(J) # Norm of J
    M_t = np.sum(M)*1E10 # Enclosed mass within Rmax
    R = np.max(np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)) #Rmax
    V_c = np.sqrt(G*M_t/R) # V_c at Rmax and M_t
    l = J_n / (np.sqrt(2.0) * M_t * V_c * R)
    return l.value

def kinetic_energy(vxyz,M):

    """
    Kinetic energy
    """
    U = 0.5*M*(vxyz[:,0]**2.0+vxyz[:,1]**2.0+vxyz[:,2]**2.0)
    return U


def radial_velocity(xyz, vxyz):

    """
    Radial velocity dispersion
    """

    r = np.linspace(0, 500, 50)
    r_p = np.sqrt(xyz[:,0]**2.0 + xyz[:,1]**2.0 + xyz[:,2]**2.0)
    vr_disp = np.zeros(len(r))
    for j in range(1,len(r)):
        index = np.where((r_p<r[j]) & (r_p>r[j-1]))[0]
        pos = xyz[index]
        vel = vxyz[index]
        vr = np.array([np.dot(vel[i], pos[i]) / np.linalg.norm(pos[i]) for i in range(len(pos))])
        vr_disp[j] = np.mean((vr-np.mean(vr))**2.0)
    return np.sqrt(vr_disp)

def tangential_velocity(xyz, vxyz):

    """
    Tangential velocity dispersion.
    """

    r = np.linspace(0, 500, 50)
    r_p = np.sqrt(xyz[:,0]**2.0 + xyz[:,1]**2.0 + xyz[:,2]**2.0)
    vt_disp = np.zeros(len(r))
    for j in range(1,len(r)):
        index = np.where((r_p<r[j]) & (r_p>r[j-1]))[0]
        pos = xyz[index]
        vel = vxyz[index] 
        vt = [np.linalg.norm(np.cross(vel[i], pos[i]))/ np.linalg.norm(pos[i]) for i in range(len(pos))]
        vt_disp[j] = np.mean((vt-np.mean(vt))**2.0)
    return np.sqrt(vt_disp)

def beta_anisotropy(xyz, vxyz):

    """
    anisotropy parameter
    """

    sigma_vt = tangential_velocity(xyz, vxyz)
    sigma_vr = radial_velocity(xyz, vxyz)
    Beta = 1.0 - sigma_vt**2.0/(2.0*sigma_vr**2.0)
    return Beta
