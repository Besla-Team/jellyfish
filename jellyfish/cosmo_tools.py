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

