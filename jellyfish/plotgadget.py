# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pygadgetreader import *
import numpy as np
import yt
from yt.units import parsec, Msun, kpc,g
from octopus import CM, CM_disk_potential
import matplotlib.pyplot as plt

class PlotGadget:

    def __init__(self, path, type, box_size=1000):
        """ 

        Plot various density plots, particle plots, and profiles for Gadget output
        
        Input:
        ------
        path : str
            path to Gadget3 snapshot

        type : str
            'disk', 'dm', or 'bulge' component

        box_size : float
            size of the box in which all particles will be included (kpc)

        """

        print(path)
        self.path = path
        self.type = type
        self.box_size = box_size

        pos = readsnap(self.path, 'pos',type)
        vel = readsnap(self.path, 'vel', type)
        mass = readsnap(self.path, 'mass', type)
        pot = readsnap(self.path, 'pot', type)

        # choose COM function based on component 
        if type == 'dm' or 'bulge':
            self.com, self.vcom = CM(pos, vel)

        if type == 'disk':
            self.com, self.vcom = CM_disk_potential(pos, vel, pot)

        x,y,z = pos[:,0], pos[:,1], pos[:,2]  
        x = x - self.com[0]
        y = y - self.com[1]
        z = z - self.com[2]

        r = (x**2 + y**2 + z**2)**0.5

        # cut the particles based on given box size
        cut = np.where(r < box_size)[0]

        self.pos  = pos[cut]
        self.x, self.y, self.z = self.pos[:,0] - self.com[0], self.pos[:,1]-self.com[1], self.pos[:,2]-self.com[2] 
        self.r = (self.x**2 + self.y**2 + self.z**2)**0.5
        self.vel = vel[cut]
        self.vx, self.vy, self.vz = self.vel[:,0], self.vel[:,1], self.vel[:,2]

        self.mass = mass[cut]
        self.pot = pot[cut]

        # create data structure for particle plots
        self.data = {'particle_position_x': self.x,\
                'particle_position_y': self.y,\
                'particle_position_z': self.z,\
                'particle_velocity_x': self.vel[:,0],\
                'particle_velocity_y': self.vel[:,1],\
                'particle_velocity_z': self.vel[:,2],\
                'particle_mass': self.mass,\
                'particle_pot' : self.pot}


        # establish bounding box
        bbox_lim = 2e5 #kpc
        self.bbox = [[-bbox_lim,bbox_lim],[-bbox_lim,bbox_lim],[-bbox_lim,bbox_lim]]

    def totalmass_bytype(self):
        """
        Output:
        ------
        Sum of all particles of this type in units of Msun
        """

        return np.sum(self.data['particle_mass'])*1e10

    def header(self):
        """
        Output:
        ------
        Print all header information
        """

        return readheader(self.path, 'header')

    def header_time(self):
        """
        Output:
        ------
        Print the time in Gyr from the header
        """

        return readheader(self.path, 'time')

    def peak_density_loc(self):
        """
        Output:
        ------
        Returns the x,y,z position for the peak density location.
        """

        ds = yt.load(self.path,bounding_box=self.bbox)
        ad= ds.all_data()
        density = ad[("deposit","all_density")]
        wdens = np.where(density == np.max(density))
        coordinates = ad[("all","Coordinates")]
        center = coordinates[wdens][0]
        return center 

    def plot_fullbox_projected_density(self, projection_plane, savename):
        """
        Inputs:
        ------
        projection_plane: int or str
            0,1,2 or 'x', 'y', 'z' to set which cross section of the 3D data is plotts
            
        savename: str
            String to save file name. Leave off the file extension

        Output:
        ------
        A 2D plot showing the density projection of the entire box integrated along the line of sight.
        """

        ds = yt.load(self.path,bounding_box=self.bbox)
        ad= ds.all_data()
        px = yt.ProjectionPlot(ds, projection_plane, ('deposit', 'all_density'))
        px.annotate_text((0.7, 0.85), '%s Gyr'%round(self.header_time(),2), coord_system='figure', text_args={'color':'white'})
        px.save('%s'%savename)
        return

    def plot_projected_density(self, projection_plane, savename, zoom=False):
        """
        Inputs:
        ------
        projection_plane: int or str
            0,1,2 or 'x', 'y', 'z' to set which cross section of the 3D data is plotts
            
        savename: str
            String to save file name. Leave off the file extension

        zoom : int
            The number of times to zoom on the center (i.e. 2x, 4x)

        Output:
        ------
        A 2D plot showing the density projection of the cut box integrated along the line of sight.
        """
        
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
        """
        Inputs:
        ------
        projection_plane: int or str
            0,1,2 or 'x', 'y', 'z' to set which cross section of the 3D data is plotts
            
        savename: str
            String to save file name. Leave off the file extension

        Output:
        ------
        A 2D plot showing the volumetric density of the cut box 
        """

        center = self.com
        ds = yt.load(self.path,bounding_box=self.bbox)
        ad = ds.all_data()

        slc = yt.SlicePlot(ds, projection_plane, ('deposit', 'Disk_density'), center=center)
        slc.set_width((self.box_size, 'kpc'))
        slc.annotate_text((0.7, 0.85), '%s Gyr'%round(self.header_time(),2), coord_system='figure', text_args={'color':'white'})
        slc.save('%s'%savename)
        return

    def plot_particle_density(self, projection_plane, savename):
        """
        Inputs:
        ------
        projection_plane: int or str
            0,1,2 or 'x', 'y', 'z' to set which cross section of the 3D data is plotts
            
        savename: str
            String to save file name. Leave off the file extension

        Output:
        ------
        A 2D particle plot showing the mass density of the cut box 
        """

        bbox = 1.1*np.array([[np.min(self.x), np.max(self.x)], [np.min(self.y), np.max(self.y)], [np.min(self.z), np.max(self.z)]])
        ds = yt.load_particles(self.data, length_unit=kpc,mass_unit=1e10,  bbox = bbox, n_ref=4)
        center = self.com
        p = yt.ParticleProjectionPlot(ds, projection_plane, ['particle_mass'])#, center=center)#, width=self.box_size, depth=self.box_size)
        p.set_colorbar_label('particle_mass', 'Msun')
        #p.zoom(2)
        p.annotate_text((0.7, 0.85), '%s Gyr'%round(self.header_time(),2), coord_system='figure', text_args={'color':'black'})
        p.save('%s'%savename)
        return

    def plot_rot_curve(self, bin_width, savename=False, newfig=False, xlim=20):
        """
        Inputs:
        ------
        bin_width: float
            The bin width (in kpc) of how often to sample the rotation curve.
            
        savename: str
            String to save file name. Leave off the file extension

        newfig: bool
            If True, it will initiate a new figure on which to plot.

        xlim: float
            The limit in kpc for how far out to plot the rotation curve.

        Output:
        ------
        A plot showing r vs. v_circ

        rs : list of floats
            The range over which the rotation curve is computed. Determined by bin_width and box_size.

        vcs : the circular velocity at a given value of r, computed from a spherical approximation of v_circ = sqrt(G*M(r)/r)

        """

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

    def pot_enclosed(self, rmin=0, rmax=300, nbins=30):
        """
        Compute the potential profile of a given galaxy component

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


        pos = readsnap(self.path, 'pos', self.type)
        pot = readsnap(self.path, 'pot', self.type)

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


    def rho_enclosed(self, rmin=0, rmax=300, nbins=30):
        """
        Compute the density profile of a given galaxy component

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

        #assert type(nbins) == int, "nbins should be an int variable."

        pos = readsnap(self.path, 'pos', self.type)
        mass = readsnap(self.path, 'mass', self.type)

        r_pos = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)

        r = np.linspace(rmin, rmax, nbins-1)

        rho = np.zeros(nbins-1)

        # Loop over the radial bins.
        for i in range(1, len(r)):
            indexh = np.where((r_pos<r[i]) & (r_pos>r[i-1]))[0]

            rho[i-1] = (3*(len(indexh)*mass[0])) / (4*np.pi*r[i]**3)

        return r, rho

