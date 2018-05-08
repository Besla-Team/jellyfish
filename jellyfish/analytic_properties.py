from scipy.optimize import bisect
import numpy as np

def r_t(R, M, m):
    ''' Tidal radius calculation when both the host and satellite are considered point masses.
    R (kpc): The distance between the two galaxies (i.e. pericenter).
    M (Msun): Total mass of the host galaxy.
    m (Msun): Total mass of the satellite galaxy.
    '''
    return R*(m/(2*M))**(1/3.)


def mass_nfw(Mvir, R, cvir, omegam=0.3):
    ''' NFW halo mass profile
    Mvir (Msun): virial mass
    R (kpc): distance from halo COM
    cvir: virial concentration
    omegam: cosmological baryon fraction
    '''

    def f(x):
        return np.log(1+x) - (x / (1+x))

    def delta_vir(omegam=0.3):
        ''' Bryan & Norman 1998 definition assuming spherical top-hat perturbation
        '''
        x = omegam - 1.
        deltac = 18*np.pi**2 + 82*x -39*x**2
        return deltac/omegam

    h = 0.7
    term2 = delta_vir(omegam) * omegam / 97.2
    term3 = Mvir * h / 1e12
    rvir = (206./h) * term2**(-1./3.) * term3**(1./3.)
    rs = rvir/cvir
    x = R/rs
    return Mvir*f(x)/f(cvir)

def mass_plummer(M, R, a):
    '''Plummer sphere mass profile
    M (Msun): total mass
    R (kpc): distance from COM
    a (kpc): Plummer scale length
    '''
    return M*R**3/ (R**2 + a**2)**(1.5)

def mass_hernquist(M, R, a):
    ''' Hernquist mass profile
    M (Msun): total mass
    R (kpc): distance from COM
    a (kpc): Hernquist scale length
    '''
    return M*R**2. / (R+a)**2.


def r_t0(R, host, M, A, m):
    ''' Tidal radius calculation when the host is an extended body and the satellite is a point mass.
    R (kpc): 
    '''
    if host == 1:
        MR = mass_nfw(M, R, A)

    if host == 2:
        MR = mass_plummer(M, R, A)

    if host == 3:
        MR = mass_hernquist(M, R, A)

    print(R, M, m)
    return R*(m/(2*MR))**(1/3.)


def r_t1(R, host, sat, Mhost, Ahost, msat, asat):
    ''' Eq. 3 from van den Bosch 2018a: Tidal radius calculation where both host and satellite are extended bodies.

        R (kpc): the separation between two galaxies' COMs
        host (int): 1,2,3 which correspond to nfw, plummer, hernquist profiles
        sat (int): 1,2,3 which correspond to nfw, plummer, hernquist profiles
        Mhost (Msun): the mass of the host
        Ahost (kpc): the scale radius or concentration for the chosen host mass profile
        msat (Msun): the mass of the satellite
        asat (kpc): the scale radius or concentration for the chosen satellite mass profile
       
    '''

    def r_t1_b(r_t, R, host, sat, Mhost, Ahost, msat, asat):
        ''' Function to use scipy bisection method to numerically solve for r_t
        '''

        R_R = np.arange(1., R+0.1*2., 0.1)

        if host == 1:
            MR = mass_nfw(Mhost, R_R, Ahost)

        if host == 2:
            MR = mass_plummer(Mhost, R_R, Ahost)

        if host == 3:
            MR = mass_hernquist(Mhost, R_R, Ahost)

        if sat == 1:
            mr = mass_nfw(msat, r_t, asat)

        if sat == 2:
            mr = mass_plummer(msat, r_t, asat)

        if sat == 3:
            mr = mass_hernquist(msat, r_t, asat)

        dlnM_dlnR = np.gradient(list(np.log(MR)), R_R[1]-R_R[0])
        r_cut = np.argmin(np.abs(R_R-R))
        der = dlnM_dlnR[r_cut]
        f = r_t - R*(mr/MR[r_cut] * (1/(2-der)))**(1/3.0)
        return f

    rt_find = bisect(r_t1_b, 1., R, args=(R, host, sat, Mhost, Ahost, msat, asat))

    return rt_find


if __name__ == "__main__":
    """
    M = 1.5e12
    A = 9.56
    a = 20.
    m = 2.5e11
    R = 150.

    #r_t1 should be larger than r_t
    print 'this', r_t1(R,1, 2, M, A, m, a)#, r_t(R, M , m)


    M = 2e12
    m = 2.5e11
    A = 9.36
    a = 20.


    #r_t1 should be larger than r_t
    print 'this', r_t1(R,1, 2, M, A, m, a), r_t(R, M , m), r_t0(R, 1, M, A, m)

    """
