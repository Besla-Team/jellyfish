#!/usr/bin/env python
"""Functions to calculate cosmological definitions in the Illustris cosmology."""

__author__ = "Ekta Patel and contributing authors"
__copyright__ = "Copyright 2015, The Authors"
__credits__ = ["Ekta Patel and contributing authors"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ekta Patel"
__email__ = "ektapatel@email.arizona.edu"
__status__ = "Beta -- forever."

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.integrate import simps
import astropy.units as u
from astropy import constants
import scipy.integrate as integrate
from scipy.integrate import quad

G = constants.G.to(u.kpc * u.km**2. / u.Msun/ u.s**2.)

def time(z, h, OM,OL):
    '''
    Calculate lookback time for a flat cosmology
    '''
    if z == 0.:
        time = 0. 
    else: 
        H0 = h*100
        H0 = H0 * 3.241e-20 / 3.171e-17 # final units of[Gyr ^-1]

        def f(z):
            return 1/ (H0*(1+z)*np.sqrt(OM*(1+z)**3 + OL)) 

        zs = np.arange(0., z, 1e-4)
        y = f(zs)
        time = simps(y,zs)
    return time #in Gyrs

# delta_c taken from Bryan and Norman (1998)
def delta_vir(omegam):
    x = omegam - 1.
    deltac = 18*np.pi**2 + 82*x -39*x**2
    return deltac/omegam

# taken from van der Marel 2012, equation A1 ("THE M31 VELOCITY VECTOR. II. RADIAL ORBIT TOWARD THE MILKY WAY AND IMPLIED LOCAL GROUP MASS")
def r_vir(omegam, h, Mvir):
    a = 206./h
    b = delta_vir(omegam) * omegam / 97.2
    c = Mvir * h / 1e12
    return a * b**(-1./3.) * c**(1./3.)

def v_vir(omegam, h, Mvir):
    rvir = r_vir(omegam, h, Mvir)
    return np.sqrt(G*Mvir/rvir)

# vdM 2012, equation A5
def q(omegam):
    return 200. / (delta_vir(omegam) * omegam)

# taken from Klypin 2011 on Bolshoi simulations, equation 10
def c_vir(h, Mvir):
    a = Mvir *h/ 1e12
    return 9.60*(a)**(-0.075)

# vdM 2012 definition for cvir = rvir/rs
def r_s(omegam, h, Mvir, c=False):
    rv = r_vir(omegam, h, Mvir)
    cv = c_vir(h, Mvir)
    if c: 
        cv = c 
    return rv/cv

# vdM 2012, equation A3
def f(x):
    a = np.log(1+x) 
    b = x / (1+x)
    return a - b

# http://glowingpython.blogspot.com/2012/01/fixed-point-iteration.html
# fp iteration reference in link above; vdM 2012, equation A6
def c_200(cvir,omegam, h, Mvir):
    c0 = cvir
    qi = q(omegam)
    fvir = f(cvir)
    tol = 1e-7
    maxiter=200
    e = 1
    itr = 0
    cp = []
    while(e > tol and itr < maxiter):
        c = cvir * (f(c0)/(qi*fvir))**(1./3.)
        e = norm(c0 - c)
        c0 = c
        cp.append(c0)
        itr = itr + 1.
    return c

# vdM 2012 definition for c200=r200/rs
def r_200(omegam, h, Mvir):
    cvir = c_vir(h, Mvir)
    c200 = c_200(cvir, omegam, h, Mvir)
    rs = r_s(omegam, h, Mvir)
    return c200*rs

# vdM 2012, equation A7
def M200(omegam, h, Mvir):
    cvir = c_vir(h, Mvir)
    c200 = c_200(cvir, omegam, h, Mvir)
    f200 = f(c200)
    fvir = f(cvir)
    return f200 * Mvir / fvir

# vdM 2012, equation A7
def mass_ratio(omegam, h, Mvir):
    ''' Returns ratio of M200 to Mvir'''
    m200 = M200(omegam, h, Mvir)
    return m200/Mvir

# vdM 2012, equation A11
def ascale(omegam, h, Mvir, a200=False):
    ''' calculates virial hernquist scale radius from the NFW profile; 
    if 200=True calculated the 200 hernquist scale radius'''

    rs = r_s(omegam, h, Mvir)
    cvir = c_vir(h, Mvir)
    a = 1./np.sqrt(2*f(cvir))
    b = 1./cvir
    scale = rs/(a-b)

    if a200:
        c200 = c_200(cvir, omegam, h, Mvir)
        a = 1./np.sqrt(2*f(c200))
        b = 1./c200
        scale = rs/(a-b)
    return scale

# vdM 2012, equation A12
def Mh2Mvir(omegam, h, Mvir, a200=False):
    ''' Returns ratio of hernquist mass to virial mass '''

    avir = ascale(omegam, h, Mvir)
    rs = r_s(omegam, h, Mvir)
    cvir = c_vir(h, Mvir)
    ratio = (avir/rs)**2 / (2*f(cvir))
    
    if a200:
        a_200 = ascale(omegam, h, Mvir, a200=True)
        ratio = (a_200/rs)**2 / (2*f(cvir))
    return ratio

class CosmologicalTools:
    '''
    Author: Katie Chamberlain - 2018
	Compute various cosmological quantities for
    a given cosmology. 
    Function is called as CosmologicalTools(OmegaM, OmegaR, OmegaL, h)
    '''
        
    def __init__(self, OmegaM, OmegaR, OmegaL, h):
        '''
        initialize class - for any cosmology:
        Inputs:    Omega M matter density parameter
                   Omega R radiation density parameter
                   Omega L  dark energy density parameter
                   h  normalization for the hubble parameter
        '''
        
        # initialize the cosmology
        self.OmegaM = OmegaM # Matter Density Parameter
        self.OmegaL = OmegaL # Dark Energy Density Parameter
        self.OmegaR = OmegaR # Radiation density Parameter
        self.OmegaK = 1.0 - (OmegaM + OmegaL + OmegaR)  # Curvature Parameter
        self.h = h # Normalization of Hubble Parameter   
        self.Ho = h*100 # Hubble Constant at z=0  100 h km/s/Mpc

        # physical constants
        self.c = 299792.458 # km/s
            
        # if open universe, compute the radius of curvature for distance measure
        if self.OmegaK > 0:
            k = -1
            self.Rc = np.sqrt((-k*self.c**2)/((self.Ho**2)*self.OmegaK)) 
    
    def HubbleParameterZ(self, z):
        '''
        Hubble parameter as a function of redshift
        Redshift can be number or array - Returns in units of km/s/Mpc
        '''       
        Omz = self.OmegaM*(1+z)**3
        Orz = self.OmegaR*(1+z)**4
        OKz = self.OmegaK*(1+z)**2
        OLz = self.OmegaL
        Hz=np.sqrt(self.Ho**2*(Omz+Orz+OKz+OLz))
        return Hz

    def OmegaM_Z(self,z):
        '''
        Matter density parameter as a function of redshift
        Redshift can be number or array
        '''
        omz = self.OmegaM*(1+z)**3*(self.Ho/self.HubbleParameterZ(z))**2
        return omz
    
    def OmegaR_Z(self,z):
        '''
        Radiation density parameter as a function of redshift
        Redshift can be number or array
        '''
        orz = self.OmegaR*(1+z)**4*(self.Ho/self.HubbleParameterZ(z))**2
        return orz
    
    def OmegaL_Z(self,z):
        '''
        Dark energy density parameter as a function of redshift
        Redshift can be number or array
        '''
        olz = self.OmegaL*(self.Ho/self.HubbleParameterZ(z))**2
        return olz

    def comovingDistance(self,z):
        '''
        calculates comoving distance as a function of redshift
        Redshift must be number (not array!) - gives distance between 0 and z
        '''
        def integrand(x):
            return self.c/self.HubbleParameterZ(x)
        return integrate.quad(integrand, 0, z)[0]
    
    def distanceMeasure(self,z):
        '''
        calculates distance measure if the universe is open
        Redshift must be number
        '''
        if self.OmegaK > 0:
            return self.Rc*np.sinh(self.comovingDistance(z)/self.Rc)
        else: 
            return self.comovingDistance(z)
    
    def angularDiameter(self,z):
        '''
        angular diameter distance as a function of redshift
        Redshift must be number - integrates between 0 and z
        '''
        return self.distanceMeasure(z)/(1+z) # Mpc/rad
    
    def luminosityDistance(self,z):
        '''
        luminosity distance as a function of redshift
        Redshift must be number - integrates between 0 and z
        '''
        return self.distanceMeasure(z)*(1+z)
    
    def distanceModulus(self,z):
        '''
        distance modulus at z
        Redshift must be number
        '''
        return (5*np.log10(self.luminosityDistance(z)*1e5))
        
    def lookbackTime(self,z):
        '''
        lookback time at z
        Redshift must be number - from z=0 to z and returns in gigayears
        '''
        def integrand(x):
                return (self.HubbleParameterZ(x)*(1+x))**-1
        return integrate.quad(integrand, 0, z)[0]*9.77799e2

    def ageUniverse(self,z):
        '''
        age of the universe at z
        Redshift must be number - from z to infinity and returns in gigayears
        '''
        def integrand(x):
                return (self.HubbleParameterZ(x)*(1+x))**-1
        intTotal = integrate.quad(integrand, 0, np.inf)[0]*9.77799e2
        return intTotal-self.lookbackTime(z)
    
    def comovingVolume(self,z):
        '''
        comoving volume per deg squared per z at a given z
        Redshift must be number - between z=0 and z in Mpc^3/deg^2/z
        '''
        st = 41253/(4*np.pi) # 1 steradian is this many square degrees
        return (self.distanceMeasure(z)**2)*self.c/self.HubbleParameterZ(z)/st
    
    def H0Limits(self,t):
        '''
        enter time in Gyr
        returns the upper bound on the hubble constant 
        for a universe with age t or older in km/s/Mpc
        '''
        def integrand(x):
                return ((self.HubbleParameterZ(x)/self.Ho)*(1+x))**-1
        unitConv = ((1)/(3.154e16))*((1)/(3.241e-20)) # 1/gyr to km/s/Mpc
        return (integrate.quad(integrand, 0, np.inf)[0]/t)*unitConv

    
