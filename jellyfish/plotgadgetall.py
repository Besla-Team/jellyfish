# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

from pygadgetreader import *
import numpy as np
import yt
from yt.units import parsec, Msun, kpc,g
from octopus import CM, CM_disk_potential
import matplotlib.pyplot as plt

class PlotGadgetAll:

    def __init__(self, path, box_size=1000):
        """ Inputs: path to Gadget3 snapshot, box_size (kpc)
        """
        print(path)
        self.path = path
        self.box_size = box_size

        self.dmmass = readsnap(self.path, 'mass', 'dm')
        self.diskmass = readsnap(self.path, 'mass', 'disk')
        self.bulgemass = readsnap(self.path, 'mass', 'bulge')


        #print len(dm), np.min(dm), np.max(dm)
        #print len(disk), np.min(disk), np.max(disk)
        #print len(bulge), np.min(bulge), np.max(bulge)

        self.dmpos = readsnap(self.path, 'pos','dm')
        self.dmvel = readsnap(self.path, 'vel', 'dm')
        self.diskpos = readsnap(self.path, 'pos','disk')
        self.diskvel = readsnap(self.path, 'vel', 'disk')
        self.diskpot = readsnap(self.path, 'pot', 'disk')
        self.bulgepos = readsnap(self.path, 'pos','bulge')

        self.halo_com, self.halo_vcom = CM(self.dmpos, self.dmvel)
        print(self.halo_com)
        self.disk_com, self.disk_vcom = CM_disk_potential(self.diskpos, self.diskvel, self.diskpot)
        print(self.disk_com)


    def header_time(self):
        return readheader(self.path, 'time')

    def plot_enclosed_mass(self, savename=False, newfig=False):
        ''' This function assumes that there is a DM, disk, and bulge component. 
        '''

        rs = np.arange(1., self.box_size, 1.)

        masses = []
        for rmin in rs:
            thismass = 0.
            for pos,mass in zip([self.dmpos, self.diskpos, self.bulgepos], [self.dmmass, self.diskmass, self.bulgemass]):
                x,y,z = pos[:,0], pos[:,1], pos[:,2]
                x = x - self.halo_com[0]
                y = y - self.halo_com[1]
                z = z - self.halo_com[2]

                r = (x**2 + y**2 + z**2)**0.5
                cut = np.where(r < rmin)[0]
                thismass += np.sum(mass[cut])
            if rmin ==5:
                print(thismass)
            masses.append(thismass*1e10)

        if newfig:
            plt.figure()
        plt.plot(rs, masses, lw=1.5, label='%s Gyr'%round(self.header_time(),2))
        plt.xlabel('distance from COM')
        plt.ylabel('total mass enclosed')
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(loc='best', ncol=2, fontsize=12)
        if savename:
            plt.savefig('%s'%savename)
        return


    def plot_total_rot_curve(self, bin_width, savename=False, newfig=False, xlim=20):
        ''' Currently all components use the halo COM, but really the disk should use the disk COM, etc.
        '''

        rs = np.arange(1., self.box_size, bin_width)
        vcs = []
        G = 4.498768e-6

        for rmin in rs:
            thismass = 0.
            for pos,mass in zip([self.dmpos, self.diskpos, self.bulgepos], [self.dmmass, self.diskmass, self.bulgemass]):
                x,y,z = pos[:,0], pos[:,1], pos[:,2]
                x = x - self.halo_com[0]
                y = y - self.halo_com[1]
                z = z - self.halo_com[2]
                r = (x**2 + y**2 + z**2)**0.5
                cut = np.where(r < rmin)[0]
                thismass += np.sum(mass[cut])
            vc = np.sqrt(G*thismass*1e10/rmin)
            vcs.append(vc)

        if newfig:
            plt.figure()

        plt.plot(rs, vcs, lw=1.5, label=r'%s Gyr, $\rm v_{max}=%s$'%(round(self.header_time(),2), round(np.max(vcs),2)))
        plt.xlim(0.,xlim)
        plt.xlabel(r'$\rm galactocentric \; radius \; [kpc]$', fontsize=16)
        plt.ylabel(r'$\rm v_{circ}\; [km \; s^{-1}]$', fontsize=16)
        plt.legend(loc='best', ncol=2, fontsize=12)

        if savename:
            print('saving figure')
            plt.savefig('%s'%savename)

        return

    def rho_enclosed(self, rmin=0, rmax=300, nbins=30):
        """
        Compute the density profile of a halo with its componenets.

        Input:
        ------
        rmin : float
            Minimum radius to compute the density profile (default=0).

        rmax : float
            Maximum radius to compute the density profile (default=300)

        nbins : int
            Number of radial bins to compute the density profile.

        Output:
        ------
        r : numpy 1D array.
            Array with the radial bins.
        rho : numpy array.
            Array with the density in each radial bin.

        """

        assert type(nbins) == int, "nbins should be an int variable."

        r_posh = np.sqrt(self.dmpos[:,0]**2 + self.dmpos[:,1]**2 + self.dmpos[:,2]**2)
        r_posd = np.sqrt(self.diskpos[:,0]**2 + self.diskpos[:,1]**2 + self.diskpos[:,2]**2)
        r_posb = np.sqrt(self.bulgepos[:,0]**2 + self.bulgepos[:,1]**2 + self.diskpos[:,2]**2)

        r = np.linspace(rmin, rmax, nbins-1)

        rho = np.zeros(nbins-1)

        # Loop over the radial bins.
        for i in range(1, len(r)):
            indexh = np.where((r_posh<r[i]) & (r_posh>r[i-1]))[0]
            indexd = np.where((r_posd<r[i]) & (r_posd>r[i-1]))[0]
            indexb = np.where((r_posb<r[i]) & (r_posb>r[i-1]))[0]

            rho[i-1] = (3*(len(indexh)*self.dmmass[0]\
                           + len(indexd)*self.diskmass[0]\
                           + len(indexb)*self.bulgemass[0])) / (4*np.pi*r[i]**3)

        return r, rho

#if __name__ == "__main__":
#     path = './m31a_25oct/gadget3_m31a_25oct_000'
#     this = PlotGadgetAll(path, 350)
    
#     this.plot_total_rot_curve(savename='test_PlotGadgetAll_rotcurve.pdf', newfig=True)
#     this.plot_enclosed_mass(savename='test_PlotGadgetAll.pdf', newfig=True)
    
