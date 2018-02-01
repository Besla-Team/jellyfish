#!/sr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from pygadgetreader import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Function that computes the center of mass for the halo and disk and
# the corresponsing orbits for the host and satellite simultaneously

def re_center(vec, cm):
    """
    Subtract a vector from a each dimension of another vector, this is done to recenter a halo
    positions and velocities to its center of mass.

    Input:
    ------
    vec : numpy.array
        A numpy array to which
    cm : numpy array
        A numpy 1d array with

    Output:
    -------

    vec : numpy.array
        A new vector with a subtracted vector in each dimension.
    """
    assert np.len(vec)==len(cm), "Make sure the len of your N-vector is the same as your 1d vector"

    for i in range(len(cm)):
        vec[:,i] -= cm[i]

    return vec


def host_sat_particles(xyz, vxyz, pids, list_num_particles, gal_ind, **kwargs):
    """
    Return a satellite or the host galaxy properties see **kwargs.


    Input:
    ------
    xyz: coordinates with shape (n,3)
    vxys: velocities with shape (n,3)
    pids: array with all the DM particles ids
    list_num_particles: A list with the number of particles of all the galaxies
                        in the ids.
    gal_ind : Index of the galaxy that you need.


    Output:
    --------
    xyz_mw, vxyz_mw, xyzlmc, vxyz_lmc: coordinates and velocities of
    the host and the sat.


    TODO:
    1. add Kwargs

    """

    assert len(xyz)==len(vxyz)==len(pids), "your input parameters have different length"
    assert type(galaxy) == int, "your galaxy type should be an integer"
    assert gal_ind >= 0, "Galaxy type can't be negative"

    sort_indexes = np.sort(pids)

    if gal_ind==0:
        N_cut_min = sort_indexes[0]
        N_cut_max = sort_indexes[sum(list_num_particles[:gal_ind+1])-1]

    elif gal_ind == len(list_num_particles)-1:
        N_cut_min = sort_indexes[sum(list_num_particles[:gal_ind])]
        N_cut_max = sort_indexes[-1]

    else:
        N_cut_min = sort_indexes[sum(list_num_particles[:gal_ind])]
        N_cut_max = sort_indexes[sum(list_num_particles[:gal_ind+1])]

    sat_ids = np.where((pids>=N_cut_min) & (pids<=N_cut_max))[0] # selecting id

    return xyz[sat_ids], vxyz[sat_ids]


def com_disk_potential(xyz, vxyz, Pdisk):
    V_radius = 2
    vx = vxyz[:,0]
    vy = vxyz[:,1]
    vz = vxyz[:,2]
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    min_pot = np.where(Pdisk==min(Pdisk))[0]
    x_min = x[min_pot]
    y_min = y[min_pot]
    z_min = z[min_pot]
    # This >2.0 corresponds to the radius in kpc of the particles that
    # I am taking into account to compute the CM
    avg_particles = np.where(np.sqrt((x-x_min)**2.0 + (y-y_min)**2.0 + (z-z_min)**2.0)<V_radius)[0]
    x_cm = sum(x[avg_particles])/len(avg_particles)
    y_cm = sum(y[avg_particles])/len(avg_particles)
    z_cm = sum(z[avg_particles])/len(avg_particles)
    vx_cm = sum(vx[avg_particles])/len(avg_particles)
    vy_cm = sum(vy[avg_particles])/len(avg_particles)
    vz_cm = sum(vz[avg_particles])/len(avg_particles)
    return np.array([x_cm, y_cm, z_cm]), np.array([vx_cm, vy_cm, vz_cm])

def velocities_com(cm_pos, pos, vel, rvir):
    """
    Function to compute the COM velocity in a sphere of 10% of Rvir (kpc)
    """
    # Compute the distance with respect to the COM
    R_cm = ((pos[:,0]-cm_pos[0])**2 + (pos[:,1]-cm_pos[1])**2 + (pos[:,2]-cm_pos[2])**2)**0.5
    # Select the particles inside 15 kpc
    index = np.where(R_cm < 0.1*rvir)[0]
    # Compute the velocities of the COM:
    velx_cm = sum(vel[index,0])/len(vel[index,0])
    vely_cm = sum(vel[index,1])/len(vel[index,1])
    velz_cm = sum(vel[index,2])/len(vel[index,2])

    return velx_cm, vely_cm, velz_cm


def CM(xyz, vxyz, delta=0.025):
    """
    Compute the center of mass coordinates and velocities of a halo
    using the Shrinking Sphere Method Power et al 2003.
    It iterates in radii until reach a convergence given by delta
    or 1% of the total number of particles.

    Parameters:
    -----------
    xyz: cartesian coordinates with shape (n,3)
    vxys: cartesian velocities with shape (n,3)
    delta(optional): Precision of the CM, D=0.025

    Returns:
    --------
    rcm, vcm: 2 arrays containing the coordinates and velocities of
    the center of mass with reference to a (0,0,0) point.

    """
    N_i = len(xyz)
    N = N_i

    xCM = 0.0
    yCM = 0.0
    zCM = 0.0

    xCM_new = sum(xyz[:,0])/N_i
    yCM_new = sum(xyz[:,1])/N_i
    zCM_new = sum(xyz[:,2])/N_i


    while (((np.sqrt((xCM_new-xCM)**2 + (yCM_new-yCM)**2 + (zCM_new-zCM)**2) > delta) & (N>N_i*0.01)) | (N>1000)):
        xCM = xCM_new
        yCM = yCM_new
        zCM = zCM_new
        # Re-centering sphere
        xyz[:,0] = xyz[:,0]
        xyz[:,1] = xyz[:,1]
        xyz[:,2] = xyz[:,2]
        R = np.sqrt((xyz[:,0]-xCM_new)**2 + (xyz[:,1]-yCM_new)**2 + (xyz[:,2]-zCM_new)**2)
        Rmax = np.max(R)
        # Reducing Sphere by its 2.5%
        index = np.where(R<Rmax*0.975)[0]
        xyz = xyz[index]
        N = len(xyz)
        #Computing new CM coordinates and velocities
        xCM_new = np.sum(xyz[:,0])/N
        yCM_new = np.sum(xyz[:,1])/N
        zCM_new = np.sum(xyz[:,2])/N

    vxCM_new, vyCM_new, vzCM_new = velocities_com([xCM_new, yCM_new, zCM_new], xyz, vxyz)
    return np.array([xCM_new, yCM_new, zCM_new]), np.array([vxCM_new, vyCM_new, vzCM_new])
