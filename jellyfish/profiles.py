# various analytic mass profiles: Hernquist, NFW, Plummer, Isothermal, Miyamoto-Nagai (for disks)
import numpy as np
from cosmo_tools import *
import astropy.units as u
from astropy import constants

G = constants.G.to(u.kpc * u.km**2. / u.Msun/ u.s**2.)

class NFW:
    def __init__(self, Mvir, r, cvir):
        """
        Inputs: Mvir (solar mass)
        r: radius (kpc)
        c: r_vir/r_scale (dimensionless)
        """
        self.Mvir = Mvir
        self.r = r
        self.cvir = cvir
        self.rs = r_vir(0.3, 0.7, Mvir)/cvir
        self.x = r/self.rs

    def density(self):
        rhos = self.Mvir/ (4*np.pi*f(self.cvir)*self.rs**3)
        return rhos/(self.x*(1+self.x)**2)

    def mass(self):
        return self.Mvir*f(self.x)/f(self.cvir)

    def potential(self):
        phi = -G*self.Mvir/f(self.cvir) *np.log(1+self.x)/self.r
        return phi

    def v_esc(self):
        phi = self.potential()
        return np.sqrt(-2*phi)

    def v_rot(self):
        m = self.mass()
        return np.sqrt(G*m/self.r)

    def acc(self):
        ## check this formula
        return -G*self.Mvir*f(self.x)/(f(self.cvir)*self.r**2.)
        

class Isothermal:
    def __init__(self, r, vc):
        """
        Inputs: r: radius (kpc)
        vc: circular velocity at a given position (i.e. solar circle) [km/s]
        """
        self.r = r
        self.vc = vc

    def potential(self):
        return - self.vc**2. * np.log(self.r)

    def density(self):
        return (self.vc)**2./ (4.*np.pi*G*self.r**2.)
    
    def mass(self):
        return (self.vc)**2.*self.r/G

    def v_esc(self):
        phi = self.potential()
        return np.sqrt(-2.*phi)

    def acc(self, pos_vec):
        # check this formula
        r = self.r
        vc = self.vc
        x,y,z = pos_vec
        rr = np.sqrt(x**2 + y**2 + z**2)
        ax = x * vc**2 /rr**2
        ay = y * vc**2 /rr**2
        az = z * vc**2 /rr**2
        acc = [ax, ay, az]
        return acc


class MN:
    def __init__(self, Mdisk, a, b, r, z):
        """
        Inputs: Mass of disk (solar mass)
        a: disk scale length (kpc)
        b: disk scale height (kpc)
        r: radius (kpc)
        z: galactocentric height (kpc)
        
        """
        self.Mdisk = Mdisk
        self.a = a
        self.b = b
        self.z = z
        self.B = np.sqrt(self.z**2 + self.b**2)
        self.r = r

    def potential(self):
        Mdisk = self.Mdisk
        r = self.r
        a = self.a
        B = self.B
        return -G*Mdisk / np.sqrt(r**2 + (a**2 + B**2)**2)

    def mass(self):
        b = self.b
        a = self.a
        r = self.r
        z = self.z
        Mdisk = self.Mdisk
        K = a + np.sqrt(z**2 + b**2)
        num = r**2.
        den = (r**2 + K**2)**(1.5)
        t1 = num/den

        num = z * K
        den = np.sqrt(z**2 + b**2) * (K**2 + r**2)**(1.5)
        t2 = num/den

        return Mdisk * ((r *t1) + (t2 * z))

    def v_rot(self):
        # taken from Bullock 05 paper
        b = self.b
        a = self.a
        r = self.r
        z = self.z
        Mdisk = self.Mdisk
        K = a + np.sqrt(z**2 + b**2)
        num = G * Mdisk * r**2
        den = (r**2 + K**2)**(1.5)
        t1 = num/den
        num = G * Mdisk * z**2 * K
        den = np.sqrt(z**2 + b**2) * (K**2 + r**2)**(1.5)
        t2 = num/den 
        return np.sqrt(t1+t2) 

    def density(self):
        Mdisk = self.Mdisk
        r = self.r
        a = self.a
        B = self.B
        b = self.b
        k = b**2 * Mdisk/ (4*np.pi)
        num = a*r**2 + ((a+3*B)*(a+B)**2)
        den = (r**2 + (a+B)**2)**(2.5) * B**3
        return k * num / den
        
    def v_esc(self):
        phi = self.potential()
        return np.sqrt(-2*phi)

    def acc(self, pos_vec):
        ## check this formula
        G = self.G
        b = self.b
        a = self.a
        r = self.r
        z = self.z
        Mdisk = self.Mdisk
        x,y,z = pos_vec
        rr = np.sqrt(x**2 + y**2 + z**2)
        c = G * Mdisk
        K = a + np.sqrt(z**2 + b**2)

        accx = c * x / (rr**2 + K**2)**(1.5)
        accy = c * y / (rr**2 + K**2)**(1.5)
        accz = c * z * component**2 / (np.sqrt(z**2 + b**2) * (rr**2 + K**2)**(1.5)) * rr        
        return [accx, accy, accz]
        
class Hernquist:
    def __init__(self, Mvir, r, a):
        """
        Inputs: Mvir: total mass (solar mass)
                a: Hernquist length scale (kpc)
                r: radius (kpc)
        """
        self.Mvir = Mvir
        self.a = a
        self.r = r 

    def density(self):
        M = self.Mvir
        r = self.r
        a = self.a
        return M*a / (2.*np.pi*r*(r+a)**3.)

    def potential(self):
        M = self.Mvir
        a = self.a
        r = self.r
        return -G*M /(r+a)

    def mass(self):
        M = self.Mvir
        r = self.r
        a = self.a
        return M*r**2. / (r+a)**2.

    def v_esc(self):
        phi = self.potential()
        return np.sqrt(-2.*phi)

    def v_rot(self):
        M = self.mass()
        r = self.r
        return np.sqrt(G*M/r)

    def a_grav(self, position):
        #check this formula
        M = self.Mvir
        a = self.a
        r = self.r
        x,y,z = position
        rr = np.sqrt(x**2 + y**2 + z**2.)
        ax = G*M* x**2 /(rr**2 * (rr+a)**2)
        ay = G*M* y**2 /(rr**2 * (rr+a)**2)
        az = G*M* z**2 /(rr**2 * (rr+a)**2)
        acc = [ax, ay, az]
        return acc

class Plummer:
    def __init__(self, Mtot, r, a):
        """
        Inputs: Mtot: total mass (solar mass)
                a: Plummer length scale (kpc)
                r: radius (kpc)
        """
        self.Mtot = Mtot
        self.a = a
        self.r = r 

    def density(self):
        M = self.Mtot
        a = self.a 
        r = self.r 
        return 3*M/(4*np.pi*a**3) * (1+(r/a)**2)**(-2.5)

    def potential(self):
        M = self.Mtot
        a = self.a
        r = self.r
        return - G*M/ np.sqrt(r**2 + a**2) 

    def v_esc(self):
        r = self.r
        phi = self.potential()
        return np.sqrt(-2*phi)
        
    def mass(self):
        M = self.Mtot
        a = self.a
        r = self.r
        mass_enc = M*r**3/ (r**2 + a**2)**(1.5)
        return mass_enc

    def v_rot(self):
        r = self.r
        M = self.mass()
        return np.sqrt(G*M/r)

    def a_grav(self, position):
        ## check this formula
        ''' Output: 3D acceleration at a given 3D position, useful for orbit integrators'''

        M = self.Mtot
        a = self.a
        x,y,z = position
        rr = np.sqrt(x**2 + y**2 + z**2)
        ax = (G * M * x**2)/(rr * (rr**2 + a**2)**(1.5) )
        ay = (G * M * y**2)/(rr * (rr**2 + a**2)**(1.5) )
        az = (G * M * z**2)/(rr * (rr**2 + a**2)**(1.5) )
        acc = [ax, ay, az]
        return acc
