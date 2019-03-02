import numpy as np
from pygadgetreader import *

class Hello_sim:
    """
    Class to read simulations in Gadget format.
    """

    def __init__(self, path, snap_name, host_npart, sat_npart, component, com, prop):
        """
        parameters:
        ----------

        path : str
            path to simulation
        snap_name : str
            name of the snapshot
        host_npart : int
            Number of *Dark Matter* particles of the host galaxy.
        sat_npart : int
            Number of *Dark Matter* particles of the satellite galaxy.
        component : str
            Component to analyze (host_dm, disk, bulge, sat_dm)
        com : str
            What coordinates center to use. 
            'com_host': com of the host
            'com_sat' : com of the satellite
            'com_000' : com at the 0,0,0 point.
            'com_host_disk' : com at the host disk using its potential
        prop : str
            What properties of the particles to return ('pos', 'vel', 'pot','mass', 'ids')

        """
        self.path = path
        self.host_npart = host_napart
        self.sat_npart = sat_npart
        self.snap_name = snap_name
        self.component = component
        self.com = com
        self.prop = prop

        
    def host_particles(self, pids, N_host_particles):
       """
       Function that return the host and the sat particles
       positions and velocities.

       Parameters:
       -----------
       pids: particles ids
       
       Returns:
       --------
       host_indices : numpy.array 
           index of the host galaxies

       """
       sort_indexes = np.sort(pids)
       N_cut = sort_indexes[N_host_particles]
       host_indices = np.where(pids<N_cut)[0]
       return host_indices

       #eturn xyz[host_ids], vxyz[host_ids], pids[host_ids], pot[host_ids], mass[host_ids]


    def sat_particles(self, pids, Nhost_particles):
        """
        Function that return the host and the sat particles
        positions and velocities.
        Parameters:
        -----------
        pids: particles ids
        Nhost_particles: Number of host particles in the snapshot
        Returns:
        --------
        sat_indices : numpy.array
            Array with the indices of the satellite galaxy.
        """
        sort_indexes = np.sort(pids)
        N_cut = sort_indexes[Nhost_particles]
        sat_indices = np.where(pids>=N_cut)[0]
        return sat_induces


    def com_disk_potential(self, xyz, vxyz, Pdisk, v_rad=2):
        """
        Function to compute the COM of the disk using the most bound particles
        within a sphere of 2 kpc.
        
        Parameters:
        ----------
        v_rad : float
            Radius of the sphere in kpc (default : 2 kpc)

        """
        min_pot = np.where(self.Pdisk==min(self.Pdisk))[0]
        x_min = self.pos_disk[min_pot,0]
        y_min = self.pos_disk[min_pot,1]
        z_min = self.pos_disk[min_pot,2]

        # Most bound particles.
        avg_particles = np.where(np.sqrt((self.pos_disk[:,0]-x_min)**2.0 +
                                         (self.pos_disk[:,1]-y_min)**2.0 +
                                         (self.pos_disk[:,2]-z_min)**2.0) < v_rad)[0]

        x_cm = sum(self.pos_disk[avg_particles,0])/len(avg_particles)
        y_cm = sum(self.pos_disk[avg_particles,1])/len(avg_particles)
        z_cm = sum(self.pos_disk[avg_particles,2])/len(avg_particles)
        vx_cm = sum(self.vel_disk[avg_particles,0])/len(avg_particles)
        vy_cm = sum(self.vel_disk[avg_particles,1])/len(avg_particles)
        vz_cm = sum(self.vel_disk[avg_particles,2])/len(avg_particles)

        return np.array([x_cm, y_cm, z_cm]), np.array([vx_cm, vy_cm, vz_cm])

    def re_center(self, vec, com):

        """
        Re center vector to a given com.
        """

        vec_new = np.copy(vec)
        vec_new[:,0] = vec[:,0] - com[0]
        vec_new[:,1] = vec[:,1] - com[1]
        vec_new[:,2] = vec[:,2] - com[2]
        return vec_new


    def read_MW_snap_com_coordinates(self):#path, snap, LMC, N_halo_part, pot, **kwargs):
        """
        Returns the MW properties.
        
        Parameters:
        path : str
            Path to the simulations
        snap : name of the snapshot
        LMC : boolean
            True or False if LMC is present on the snapshot.
        N_halo_part : int
            Number of particles in the MW halo.
        pot : boolean
            True or False if you want the potential back.
            
        Returns:
        --------
        MWpos : 
        MWvel : 
        MWpot : 
        

        """

        if (self.component == 'host_dm') |  (self.component == 'sat_dm')):
            pos = readsnap(self.path+self.snap, 'pos', 'dm')
            vel = readsnap(self.path+self.snap, 'vel', 'dm')
            ids = readsnap(self.path+self.snap, 'pid', 'dm')
            x = readsnap(self.path + self.snap, self.prop, 'dm')    

            if self.component == 'host_dm':
                ids_host = host_particles(ids, self.host_npart)
                y = x[ids_host]
            elif self.component == 'sat_dm':
                ids_sat = host.partiles(ids, self.host_npart)
                y = x[ids_sat]

        else :
            x = readsnap(self.path + self.snap, self.prop, self.component)

        if self.com == 'com_host_disk':
            self.pos_disk = readsnap(self.path+self.snap, 'pos', 'disk')
            self.vel_disk = readsnap(self.path+self.snap, 'vel', 'disk')
            self.pot_disk = readsnap(self.path+self.snap, 'pot', 'disk')
            pos_cm, vel_cm = bcom_disk_potential(pos_disk, vel_disk, pot_disk)

        
        MW_pos_cm = re_center(MW_pos, pos_cm)
        MW_vel_cm = re_center(MW_vel, vel_cm)
        
        #if 'LSR' in kwargs:
        #    pos_LSR = np.array([-8.34, 0, 0])
        #    vel_LSR = np.array([11.1,  232.24,  7.25])
            # Values from http://docs.astropy.org/en/stable/api/astropy.coordinates.Galactocentric.html
            MW_pos_cm = re_center(MW_pos_cm, pos_LSR)
            MW_vel_cm = re_center(MW_vel_cm, vel_LSR)
            
        assert len(MW_pos) == N_halo_part, 'something is wrong with the number of selected particles'

        
        
