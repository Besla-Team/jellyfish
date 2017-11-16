#!/sr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from pygadgetreader import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from com import host_sat_particles, CM

#Function that computes the center of mass orbits for the host and satellite simultaneously 

def orbit(path, snap_name, initial_snap, final_snap, Nhost_particles, delta, lmc=False, disk=False):
    """
    Computes the orbit of the host and the sat. It computes the CM of the
    host and the sat using the shrinking sphere method at each snapshot.

    Parameters:
    -----------
    path: Path to the simulation snapshots
    snap_name: Base name of the snaphot without the number and
    file type, e.g: sathost
    initial_snap: Number of the initial snapshot
    final_snap: Number of the final snapshot
    Nhost_particles: Number of host particles in the simulation.
    delta: convergence distance
    lmc: track the lmc orbit. (default = False)
    Returns:
    --------
    Xhostcm, vhostcm, xsatcm, vsatcm: 4 arrays containing the coordinates
    and velocities of the center of mass with reference to a (0,0,0) point
    at a given time.

    """

    N_snaps = final_snap - initial_snap + 1
    host_rcm = np.zeros((N_snaps,3))
    host_vcm = np.zeros((N_snaps,3))
    sat_rcm = np.zeros((N_snaps,3))
    sat_vcm = np.zeros((N_snaps,3))

    for i in range(initial_snap, final_snap+1):
        # Loading the data!
        xyz = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'pos', 'dm')
        vxyz = readsnap(path + snap_name +'_{:03d}.hdf5'.format(i),'vel', 'dm')
        pids = readsnap(path + snap_name +'_{:03d}.hdf5'.format(i),'pid', 'dm')

        if (disk==True):
            host_xyz_disk = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'pos', 'disk')
            host_vxyz_disk = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'vel', 'disk')
            host_pot_disk = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'pot', 'disk')

        ## computing COM
        if lmc==True:
            host_xyz, host_vxyz, sat_xyz, sat_vxyz = host_sat_particles(xyz, vxyz, pids, Nhost_particles)
            if disk==True:
                host_rcm[i-initial_snap], host_vcm[i-initial_snap] = CM_disk_potential(host_xyz_disk, host_vxyz_disk, host_pot_disk)
            else:
                host_rcm[i-initial_snap], host_vcm[i-initial_snap] = CM(host_xyz, host_vxyz, delta)
            sat_rcm[i-initial_snap], sat_vcm[i-initial_snap] = CM(sat_xyz, sat_vxyz, delta)
            sat_vx, sat_vy, sat_vz, R_shell = ss_velocities(sat_rcm[i-initial_snap], sat_xyz, sat_vxyz, 0.5)
            #plot_velocities(sat_vx, sat_vy, sat_vz, R_shell, i-initial_snap)

        else:
            if disk==True:
                host_rcm[i-initial_snap], host_vcm[i-initial_snap] = CM_disk_potential(host_xyz_disk, host_vxyz_disk, host_pot_disk)
            else:
                host_rcm[i-initial_snap], host_vcm[i-initial_snap] = CM(xyz, vxyz, delta)

    return host_rcm, host_vcm, sat_rcm, sat_vcm


def write_orbit(filename, hostpos, hostvel, satpos, satvel):
    f = open(filename, 'w')

    f.write('# host x(kpc), host y(kpc), host z(kpc), host vx(km/s), host vy(km/s), host vz(km/s) sat x(kpc), sat y(kpc), sat z(kpc), sat vx(km/s), sat vy(km/s), sat vz(km/s) \n')

    for i in range(len(hostpos[:,0])):
        f.write(("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} \n").format(hostpos[i,0],\
                 hostpos[i,1], hostpos[i,2], hostvel[i,0], hostvel[i,1], \
                 hostvel[i,2], satpos[i,0], satpos[i,1], \
                 satpos[i,2], satvel[i,0], satvel[i,1], satvel[i,2]))

    f.close()

