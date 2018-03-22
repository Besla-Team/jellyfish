"""
Compute the Angular Momentun, spin parameter,
anisotropy parameter, potential and kinetic
energy of the halo.
"""

import numpy as np
from scipy import linalg
import astropy.units as u
from astropy.constants import G




class PhysProps:
    
    def __init__(self, pos, vel, mass, pot):
        """
        it uses G from Astropy, note that this onw is different from the Gadget definition
        see : https://github.com/jngaravitoc/MW_anisotropy/blob/master/code/equilibrium/G_units_gadget.ipynb
        """
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.M = np.sum(mass)
        self.pot = pot
        #self.R =
        #self.V =
        self.G = G.to(u.kpc*u.km**2/u.s**2/u.Msun)



    def angular_momentum(self):

        """
        Computes the angular momentum of the DM halo.
        Inputs: position vector, velocity vector, particle masses
        Output: total orbital angular momentum vector

        """


        r_c_p = np.array([np.cross(self.pos[i], self.vel[i]) for i in range(len(self.pos))])
        J_x = np.sum(r_c_p[:,0])*u.kpc*u.km/u.s
        J_y = np.sum(r_c_p[:,1])*u.kpc*u.km/u.s
        J_z = np.sum(r_c_p[:,2])*u.kpc*u.km/u.s
        M_tot = np.sum(self.M)*u.Msun
        return J_x*M_tot, J_y*M_tot, J_z*M_tot

    def spin_param(self, J):

        """
        Spin parameter:
        Inputs: total angular momentum vector, particle masses, position vector
        Output: Bullock 2001 halo spin parameter (should be between 0-1)
        """


        J_n = linalg.norm(J) # Norm of J
         # Enclosed mass within Rmax

        #print 'R:', R
        V_c = np.sqrt(G*self.M/self.R).to('km/s') # V_c at Rmax and M_t
        #print 'V', V_c
        l = J_n / (np.sqrt(2.0) * self.M * V_c * self.R)
        return l.value

    def kinetic_energy(vxyz, M):

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
