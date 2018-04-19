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
        self.R = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
        #self.V = np.max(np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2))# 
        self.G = G.to(u.kpc*u.km**2/u.s**2/u.Msun)


    def kinetic_energy(self, R):

        """
        Kinetic energy:
        
        
        """
        index = np.where(self.R<R)[0]
        mass_trunc = np.sum(self.mass[index])
        K = 0.5*mass_trunc*(self.vel[index,0]**2.0 + self.vel[index,1]**2.0\
                            + self.vel[index,2]**2.0)
        
        return K

    def angular_momentum(self, R):

        r"""
        Computes the angular momentum of a DM halo in units of l^2/time * M.
        It assumes that all the particles have the same mass.
    
        
        $J_i = M  \sum_i (\vec{r_i} \times \vec{v_i}$)_i
         
        ## Check this!    
        """

        # this is slow!
        index = np.where(self.R<R)[0]
        pos_c = self.pos[index]
        vel_c = self.vel[index]
        mass_trunc = np.sum(self.mass[index])
        mp = self.mass[index]
        
        r_c_p = np.array([np.cross(pos_c[i], mp[i]*vel_c[i]) for i in range(len(pos_c))])
        J_x = np.sum(r_c_p[:,0])
        J_y = np.sum(r_c_p[:,1])
        J_z = np.sum(r_c_p[:,2])
        return np.array([J_x, J_y, J_z])

    def bullock_spin_param(self, J, R):

        r"""
        Bullock Spin parameter:
        
        $\lambda = \dfrac{J}{\sqrt{2}MVR}$
    
        Reference: Equation 5 in http://adsabs.harvard.edu/abs/2001ApJ...555..240B
        
        Output:
        -------
        
        lambda : float
        """


        #J_n = np.linalg.norm(J) # Norm of J
        J_n = np.sqrt(J[0]**2 + J[1]**2 + J[2]**2)
         # Enclosed mass within Rmax

        print('Assumes that the velocities are in km/s, Masses in Msun and distances in kpc.')
        index = np.where(self.R<R)[0]
        mass_trunc = np.sum(self.mass[index])
        
        V_c = np.sqrt(self.G.value*mass_trunc/(R)) # V_c at Rmax and M_t
        Lambda = J_n / (np.sqrt(2.0) * mass_trunc * V_c * R)
        
        assert Lambda <=1 ,'Error lambda is larger than 1.'
        assert Lambda >0 ,'Error lambda is negative'
        
        return Lambda

    def peebles_spin_parameter(self, J, R):
        r"""


        """
        #J_n = np.linalg.norm(J) # Norm of J
        J_n = np.sqrt(J[0]**2 + J[1]**2 + J[2]**2)
         # Enclosed mass within Rmax

        print('Assumes that the velocities are in km/s, Masses in Msun and distances in kpc.')
        index = np.where(self.R<R)[0]
        mass_trunc = np.sum(self.mass[index])
        
        K = self.kinetic_energy(R)

        E = K + self.pot[index]
        print(np.sum(E), np.sum(K), np.sum(self.pot[index]))

        Lambda_p = np.sqrt(np.sum(E))*J_n/(G*mass_trunc**(5/2.))

        return Lambda_p 


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
