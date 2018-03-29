"""
Class that computes the angular momentun, spin parameter,
anisotropy parameter, potential and kinetic energies.
of a halo.

To-Do:
------

0. Speed up angular momentum computation!

1. Rvir and Mvir are not defined jet to compute the circular velocity.
   Should we comoute them here or made them as an input parameter.

2. Peebles spin parameter





History:
--------

03/29/18: 
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
        self.R = np.max(np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2))
        self.V = np.max(np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2))
        self.G = G.to(u.kpc*u.km**2/u.s**2/u.Msun)



    def angular_momentum(self):

        r"""
        Computes the angular momentum of a DM halo.
        It assumes that all the particles have the same mass.
        
        $J_i = M  \sum_i (\vec{r_i} \times \vec{v_i}$)_i
         
        
        """

        # this is slow!
        r_c_p = np.array([np.cross(self.pos[i], self.vel[i]) for i in range(len(self.pos))])
        J_x = np.sum(r_c_p[:,0])
        J_y = np.sum(r_c_p[:,1])
        J_z = np.sum(r_c_p[:,2])
        M_tot = np.sum(self.M)
        return [J_x*M_tot, J_y*M_tot, J_z*M_tot]

    def spin_param(self, J):

        """
        Bullock Spin parameter:
        
        $\lambda = \dfrac{J}{\sqrt{2}MVR}$
    
        Reference: Equation 5 in http://adsabs.harvard.edu/abs/2001ApJ...555..240B
        
        Output:
        -------
        
        lambda : float
        """


        J_n = linalg.norm(J) # Norm of J
         # Enclosed mass within Rmax

        print('Assumes that the velocities are in km/s, Masses in Msun and distances in kpc.')
        
        V_c = np.sqrt(G.value*self.M/(self.R)) # V_c at Rmax and M_t
        #print 'V', V_c
        Lambda = J_n / (np.sqrt(2.0) * self.M * V_c * self.R)
        
        assert Lambda <=1 ,'Error lambda is larger than 1.'
        assert Lambda >0 ,'Error lambda is negative'
        
        return Lambda.value

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
