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
            'com_xyz' : com at the 0,0,0 point.
            'com_host_disk' : com at the host disk using its potential
        prop : str
            What properties of the particles to return ('pos', 'vel', 'pot','mass', 'ids')

        """
        self.path = path
        self.snap = snap_name
        self.host_npart = host_npart
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
        return sat_indices
        
    def COM(self, xyz, vxyz, m):
        """
        Returns the COM positions and velocities. 

        \vec{R} = \sum_i^N m_i \vec{r_i} / N
        
        """


        # Number of particles 
        N = sum(m)


        xCOM = np.sum(xyz[:,0]*m)/N
        yCOM = np.sum(xyz[:,1]*m)/N
        zCOM = np.sum(xyz[:,2]*m)/N

        vxCOM = np.sum(vxyz[:,0]*m)/N
        vyCOM = np.sum(vxyz[:,1]*m)/N
        vzCOM = np.sum(vxyz[:,2]*m)/N

        return [xCOM, yCOM, zCOM], [vxCOM, vyCOM, vzCOM]

    def com_shrinking_sphere(self, m, delta=0.025):
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
        
        xCM = 0.0
        yCM = 0.0
        zCM = 0.0

        xyz = self.pos
        vxyz = self.vel

        N_i = len(xyz)
        N = N_i
        
        rCOM, vCOM = self.COM(xyz, vxyz, m)
        xCM_new, yCM_new, zCM_new = rCOM
        vxCM_new, vyCM_new, vzCM_new = vCOM
      


        while (((np.sqrt((xCM_new-xCM)**2 + (yCM_new-yCM)**2 + (zCM_new-zCM)**2) > delta) & (N>N_i*0.01)) | (N>1000)):
            xCM = xCM_new
            yCM = yCM_new
            zCM = zCM_new
            # Re-centering sphere
            R = np.sqrt((xyz[:,0]-xCM_new)**2 + (xyz[:,1]-yCM_new)**2 + (xyz[:,2]-zCM_new)**2)
            Rmax = np.max(R)
            # Reducing Sphere by its 2.5%
            index = np.where(R<Rmax*0.75)[0]
            xyz = xyz[index]
            vxyz = vxyz[index]
            m = m[index]
            N = len(xyz)
            #Computing new CM coordinates and velocities
            rCOM, vCOM = self.COM(xyz, vxyz, m)
            xCM_new, yCM_new, zCM_new = rCOM
            vxCM_new, vyCM_new, vzCM_new = vCOM

        if self.prop == 'pos':
            i_com, j_com, k_com = xCM_new, yCM_new, zCM_new
        elif self.prop == 'vel':
            print('this is not implemented yet')
            #i_com, j_com, k_com = velocities_com([xCM_new, yCM_new, zCM_new], xyz, vxyz)
            
        return np.array([i_com, j_com, k_com])

    def com_disk_potential(self, v_rad=2):
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

        vx_cm = sum(self.vel_disk[avg_particles,0])/len(avg_particles)
        vy_cm = sum(self.vel_disk[avg_particles,1])/len(avg_particles)
        vz_cm = sum(self.vel_disk[avg_particles,2])/len(avg_particles)
        if self.prop == 'pos':
            i_cm = sum(self.pos_disk[avg_particles,0])/len(avg_particles)
            j_cm = sum(self.pos_disk[avg_particles,1])/len(avg_particles)
            k_cm = sum(self.pos_disk[avg_particles,2])/len(avg_particles)
        elif self.prop == 'vel':
            i_cm = sum(self.vel_disk[avg_particles,0])/len(avg_particles)
            j_cm = sum(self.vel_disk[avg_particles,1])/len(avg_particles)
            k_cm = sum(self.vel_disk[avg_particles,2])/len(avg_particles)

        return np.array([i_cm, j_cm, k_cm])

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

        if ((self.component == 'host_dm') |  (self.component == 'sat_dm')):
            pos = readsnap(self.path+self.snap, 'pos', 'dm')
            vel = readsnap(self.path+self.snap, 'vel', 'dm')
            ids = readsnap(self.path+self.snap, 'pid', 'dm')
            y = readsnap(self.path + self.snap, self.prop, 'dm')    

            if self.component == 'host_dm':
                ids_host = self.host_particles(ids, self.host_npart)
                x = y[ids_host]
            elif self.component == 'sat_dm':
                ids_sat = self.sat_particles(ids, self.host_npart)
                x = y[ids_sat]

        else :
            x = readsnap(self.path + self.snap, self.prop, self.component)

        if self.com == 'com_host_disk':
            # Add assertion to check if the disk potential is available.
            self.pos_disk = readsnap(self.path+self.snap, 'pos', 'disk')
            self.vel_disk = readsnap(self.path+self.snap, 'vel', 'disk')
            self.pot_disk = readsnap(self.path+self.snap, 'pot', 'disk')
            com = self.com_disk_potential()
            x = self.re_center(x, com)

        elif self.com == 'com_sat':
	    # Generalize this to any particle type.
            self.pos = readsnap(self.path+self.snap, 'pos', 'dm')
            self.vel = readsnap(self.path+self.snap, 'vel', 'dm')
            com = self.com_shrinking_sphere(m=np.ones(len(self.pos)))
            print('Satellite COM computed with the Shrinking Sphere Algorithm at', com)
            x = self.re_center(x, com)
            
        elif type(self.com) != str:
            x = self.re_center(x, self.com)
             
        
        #if 'LSR' in kwargs:
        #    pos_LSR = np.array([-8.34, 0, 0])
        #    vel_LSR = np.array([11.1,  232.24,  7.25])
            # Values from http://docs.astropy.org/en/stable/api/astropy.coordinates.Galactocentric.html
            
        #assert len(MW_pos) == N_halo_part, 'something is wrong with the number of selected particles'

        return x
        
