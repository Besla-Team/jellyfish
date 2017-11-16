# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

from pygadgetreader import *
import numpy as np
import yt
from yt.units import parsec, Msun, kpc,g
from octopus import CM, CM_disk_potential
import matplotlib.pyplot as plt

class PlotGadget:

    def __init__(self, path, type, box_size=1000):
        """ Inputs: path to Gadget3 snapshot, particle type, box_size (kpc)
        """
        print path
        self.path = path
        self.box_size = box_size
        pos = readsnap(self.path, 'pos',type)
        vel = readsnap(self.path, 'vel', type)
        mass = readsnap(self.path, 'mass', type)
        pot = readsnap(self.path, 'pot', type)
        print len(pos), len(vel), len(mass), len(pot)

        if type == 'dm' or 'bulge':
            self.com, self.vcom = CM(pos, vel)

        if type == 'disk':
            self.com, self.vcom = CM_disk_potential(pos, vel, pot)

        x,y,z = pos[:,0], pos[:,1], pos[:,2]  
        x = x - self.com[0]
        y = y - self.com[1]
        z = z - self.com[2]

        r = (x**2 + y**2 + z**2)**0.5

        cut = np.where(r < box_size)[0]

        self.pos  = pos[cut]
        self.x, self.y, self.z = self.pos[:,0] - self.com[0], self.pos[:,1]-self.com[1], self.pos[:,2]-self.com[2] 
        self.r = (self.x**2 + self.y**2 + self.z**2)**0.5
        self.vel = vel[cut]
        self.vx, self.vy, self.vz = self.vel[:,0], self.vel[:,1], self.vel[:,2]
        
    
        self.mass = mass[cut]
        self.pot = pot[cut]

        self.data = {'particle_position_x': self.x,\
                'particle_position_y': self.y,\
                'particle_position_z': self.z,\
                'particle_velocity_x': self.vel[:,0],\
                'particle_velocity_y': self.vel[:,1],\
                'particle_velocity_z': self.vel[:,2],\
                'particle_mass': self.mass,\
                'particle_pot' : self.pot}


        bbox_lim = 2e5 #kpc

        self.bbox = [[-bbox_lim,bbox_lim],[-bbox_lim,bbox_lim],[-bbox_lim,bbox_lim]]

    def totalmass_bytype(self):
        return np.sum(self.data['particle_mass'])*1e10

    def header(self):
        return readheader(self.path, 'header')

    def header_time(self):
        return readheader(self.path, 'time')

    def peak_density_loc(self):
        ds = yt.load(self.path,bounding_box=self.bbox)
        ad= ds.all_data()
        density = ad[("deposit","all_density")]
        wdens = np.where(density == np.max(density))
        coordinates = ad[("all","Coordinates")]
        center = coordinates[wdens][0]
        return center 
        
    def plot_fullbox_projected_density(self, projection_plane, savename):
        ds = yt.load(self.path,bounding_box=self.bbox)
        ad= ds.all_data()
        px = yt.ProjectionPlot(ds, projection_plane, ('deposit', 'all_density'))
        px.annotate_text((0.7, 0.85), '%s Gyr'%round(self.header_time(),2), coord_system='figure', text_args={'color':'white'})
        px.save('%s'%savename)
        return
        
    def plot_projected_density(self, projection_plane, savename, zoom=False):
        center = self.com
        ds = yt.load(self.path,bounding_box=self.bbox)
        ad = ds.all_data()
        px = yt.ProjectionPlot(ds, projection_plane, ('deposit', 'all_density'), center=center, width=self.box_size)
        if zoom:
            px.zoom(zoom)
        
        px.annotate_text((0.7, 0.85), '%s Gyr'%round(self.header_time(),2), coord_system='figure', text_args={'color':'white'})
        px.save('%s'%savename)
        
        return

    def plot_all_density(self, projection_plane, savename):
        '''CAUTION: this plots the DISK density
        '''
        center = self.com
        ds = yt.load(self.path,bounding_box=self.bbox)
        ad = ds.all_data()

        slc = yt.SlicePlot(ds, projection_plane, ('deposit', 'Disk_density'), center=center)
        slc.set_width((self.box_size, 'kpc'))
        slc.annotate_text((0.7, 0.85), '%s Gyr'%round(self.header_time(),2), coord_system='figure', text_args={'color':'white'})
        slc.save('%s'%savename)
        return

    def plot_particle_density(self, projection_plane, savename):
        bbox = 1.1*np.array([[np.min(self.x), np.max(self.x)], [np.min(self.y), np.max(self.y)], [np.min(self.z), np.max(self.z)]])
        ds = yt.load_particles(self.data, length_unit=kpc,mass_unit=1e10,  bbox = bbox, n_ref=4)
        center = self.com
        p = yt.ParticleProjectionPlot(ds, projection_plane, ['particle_mass'])#, center=center)#, width=self.box_size, depth=self.box_size)
        p.set_colorbar_label('particle_mass', 'Msun')
        #p.zoom(2)
        p.annotate_text((0.7, 0.85), '%s Gyr'%round(self.header_time(),2), coord_system='figure', text_args={'color':'white'})
        p.save('%s'%savename)
        return

    def plot_rot_curve(self, bin_width, savename=False, newfig=False, xlim=20):
        rs = np.arange(1., self.box_size, bin_width)
        vcs = []
        G = 4.498768e-6
        for r in rs:
            cut = np.where(self.r < r)[0]
            mass = np.sum(self.mass[cut])
            vc = np.sqrt(G*mass*1e10/r)
            vcs.append(vc)

        if newfig:
            plt.figure()

        plt.plot(rs, vcs, lw=1.5, label='%s Gyr'%round(self.header_time(),2))
        plt.xlim(0.,xlim)
        plt.xlabel(r'$\rm galactocentric \; radius \; [kpc]$', fontsize=16)
        plt.ylabel(r'$\rm v_{circ}\; [km \; s^{-1}]$', fontsize=16)
        plt.legend(loc='best', ncol=2,fontsize=12)
        if savename:
            plt.savefig('%s'%savename)
        return



    def pot_enclosed(self, path, type, rmin, rmax, nbins):
        """
        Compute the potential profile of a given galaxy component

        Input:
        ------
        type : str
            Particle type to use. ('dm', 'disk', 'bulge')
        path : str
            Path to N-body simulation
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

        self.path = path
        pos = readsnap(self.path, 'pos', type)
        pot = readsnap(self.path, 'pot', type)

        r_pos = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2) 
        r = np.linspace(rmin, rmax, nbins-1)
        pot_bin = np.zeros(nbins-1)

        for i in range(1, len(r)):
            indexh = np.where((r_pos<r[i]) & (r_pos>r[i-1]))[0]
            if len(indexh)==0:
                pot_bin[i-1] = 0
            else:
                pot_bin[i-1] = np.mean(pot[indexh]) 

        return r, pot_bin


    def rho_enclosed(self, path, type, rmin=0, rmax=300, nbins=30):
        """
        Compute the density profile of a given galaxy component

        Input:
        ------
        type : str
            Particle type to use. ('dm', 'disk', 'bulge')
        path : str
            Path to N-body simulation
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
        self.path = path
        pos = readsnap(self.path, 'pos', type)
        mass = readsnap(self.path, 'mass', type)

        r_pos = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)

        r = np.linspace(rmin, rmax, nbins-1)

        rho = np.zeros(nbins-1)

        # Loop over the radial bins.
        for i in range(1, len(r)):
            indexh = np.where((r_pos<r[i]) & (r_pos>r[i-1]))[0]

            rho[i-1] = (3*(len(indexh)*mass[0])) / (4*np.pi*r[i]**3)

        return r, rho

if __name__ == "__main__":
#    path = './m31a_25oct_gadget3_m31a_25oct_000'
#    this = PlotGadget(path, 350)
#    print this.plot_enclosed_mass('test3b_m31_enclosed_mass.pdf')

#     print this.header_time()
#     print this.totalmass()
#     print this.find_bbox_lim()
#     print this.peak_density_loc()
#     this.plot_fullbox_projected_density(0,'test3b_m31_fullbox')
#     this.plot_projected_density(0, 'test3b_m31')
#     this.plot_all_density(0, 'test3b_m31')
#     this.plot_particle_density(0, 'test3b_m31')
