import numpy as np
import matplotlib.pyplot as plt

###################################################
#  Katie Chamberlain  -  February 2019            #
#  Abundance matching code relating stellar mass  #
#  to halo mass for a galaxy/DMH                  #
#  ...                                            #
#  Follows Moster, Naab, and White (2012)         #
#  https://arxiv.org/pdf/1205.5807.pdf            #
###################################################


#  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  #
#  Note: to make plots, run plottingRoutine(z, "kind")  #
#     where "kind" can be "random" or "same"            #
#  ...                                                  #
#  Comment out the savefig line at the end of the       #
#     plotting routine if show is preferred             #
#  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  #

f = np.random.normal

def logM1(z, kind = "median"):
    """eq. 11"""
    M10      = 11.59
    M10range = 0.236
    M11      = 1.195
    M11range = 0.353
    
    if kind == "median":                 # picking out only the median value
        return M10 + M11*(z/(1+z))
    else:                                # picking gaussian distributed variables
        M10Gauss, M11Gauss = f(M10, M10range,1), f(M11, M11range, 1)
        return M10Gauss + M11Gauss*(z/(1+z))
    
def N(z, kind="median"):
    """eq. 12"""
    N10      = 0.0351
    N10range = 0.0058
    N11      = -0.0247
    N11range = 0.0069
    
    if kind == "median":                 # picking out only the median value
        return N10 + N11*(z/(1+z))
    else:                                # picking gaussian distributed variables
        N10Gauss, N11Gauss = f(N10, N10range,1),f(N11, N11range, 1)
        return N10Gauss + N11Gauss*(z/(1+z))
    
def beta(z, kind="median"):
    """eq. 13"""
    beta10      = 1.376
    beta10range = 0.153
    beta11      = -0.826
    beta11range = 0.225
    
    if kind == "median":                 # picking out only the median value
        return beta10 + beta11*(z/(1+z))
    else:                                # picking gaussian distributed variables
        beta10Gauss, beta11Gauss = f(beta10, beta10range,1),f(beta11, beta11range, 1)
        return beta10Gauss + beta11Gauss*(z/(1+z))   

def gamma(z, kind="median"):
    """eq. 14"""
    gamma10      = 0.608
    gamma10range = 0.059
    gamma11      = 0.329
    gamma11range = 0.173
    
    if kind == "median":                 # picking out only the median value
        return gamma10 + gamma11*(z/(1+z))
    else:                                # picking gaussian distributed variables
        gamma10Gauss, gamma11Gauss = f(gamma10, gamma10range,1),f( gamma11, gamma11range, 1)
        return gamma10Gauss + gamma11Gauss*(z/(1+z))

def SHMratio(M, z, kind="median"):
    """
    Inputs: masses M in solar masses (NOT in logspce)
    Outputs: Stellar mass to halo mass ratio
    """
    M1 = 10**logM1(z, kind)
    A = (M/M1)**(-beta(z,kind))
    B = (M/M1)**(gamma(z,kind))
    SHMratio = 2*N(z,kind)*(A+B)**-1
    return SHMratio

def stellarMass(M, z, kind="median"):
    """ returns the stellar mass in Msun """
    return M*SHMratio(M, z, kind)

def bounds(Ms, z, x):
    """ 
    finding x-sigma bounds on the stellar mass
    via error propagation 
    credit: Mark Vogelsberger and Gurtina Besla
    Inputs: mass in solar masses
    Outputs: upper and lower stellar mass at x-sigma from mean (not log)
    """

    N1 = N(z)
    M1 = logM1(z)
    Beta1 = beta(z)
    Gamma1 = gamma(z)
    a = 1/(1+z)

    M = np.log10(Ms)
    
    M10range, M11range = 0.236, 0.353
    N10range, N11range = 0.0058, 0.0069
    B10range, B11range = 0.153, 0.225
    G10range, G11range = 0.059, 0.173

    Mstar = M + np.log10(2*N1) - np.log10( (10**(M-M1))**(-Beta1) + (10**(M-M1))**(Gamma1) ) 
    dmdN10 = np.log10(np.e)/N1
    dmdN11 = dmdN10*(1.0-a)

    eta = np.exp(np.log(10.0)*(M-M1))
    alpha = eta**(-Beta1) + eta**(Gamma1)
    dmdM10 = (Gamma1*eta**(Gamma1) - Beta1*eta**(-Beta1))/alpha
    dmdM11 = (Gamma1*eta**(Gamma1) - Beta1*eta**(-Beta1))/alpha*(1.0-a)
    dmdB10 = np.log10(np.e)/alpha*np.log(eta)*eta**(-Beta1)
    dmdB11 = np.log10(np.e)/alpha*np.log(eta)*eta**(-Beta1)*(1.0-a)
    dmdG10 = -np.log10(np.e)/alpha*np.log(eta)*eta**(Gamma1)
    dmdG11 = -np.log10(np.e)/alpha*np.log(eta)*eta**(Gamma1)*(1.0-a)

    sigma = np.sqrt(dmdM10**2.0*M10range**2.0 + dmdM11**2.0*M11range**2.0 + dmdN10**2.0*N10range**2.0 + dmdN11**2.0*N11range**2.0 + dmdB10**2.0*B10range**2.0 + dmdB11**2.0*B11range**2.0 + dmdG10**2.0*G10range**2.0 + dmdG11**2.0*G11range**2.0)

    Mstaru = Mstar + (sigma*x)
    Mstarl = Mstar - (sigma*x)

    return 10**Mstaru, 10**Mstarl
 
# ---------------------------------------#

def plottingRoutine(z=0, kind = "random"):
    
    if kind == "random": # to plot a random distribution of masses and their stellarMass
        Ms          = np.logspace(10,15,1000)                       # logarithmically spaced halo masses
        mstarMedian = stellarMass(Ms, z)                            # median mstar masses
        
        numPoints   = 100000                                        # number of random draws to make
        MsGauss     = 10**(5*np.random.rand(numPoints)+10)          # random halo mass vals
        mstarGauss  = [stellarMass(i, z, "gauss") for i in MsGauss]  # draws from gaussian dists
        
        numBins      = 150
        binsx, binsy = np.logspace(10,15,numBins),np.logspace(8,12, numBins) # bin definitions
    
    else:    # to plot a run of the stellarMass function using the same draws at each mass
        numPoints   = 10000
        Ms          = np.logspace(10,15,numPoints)                  # logarithmically spaced halo masses
        mstarMedian = stellarMass(Ms, z)                            # median mstar masses
        
        trials     = 10000
        MsGauss    = np.concatenate([Ms]*trials)
        mstarGauss = np.concatenate([stellarMass(Ms,0,"gauss") for i in range(trials)])
        
        numBins      = 150
        binsx, binsy = np.logspace(10,15,numBins),np.logspace(8,12,numBins)

    upper1, lower1 = bounds(Ms, z, 1)                           # one sigma bounds on mstar
    upper2, lower2 = bounds(Ms, z, 2)                           # two sigma bounds on mstar

    fig,ax = plt.subplots(figsize=(10,8))
    plt.hist2d(MsGauss,mstarGauss,bins=[binsx, binsy],cmap = "RdPu")
    plt.plot(Ms,mstarMedian,label = 'Median',color='black')
    plt.plot(Ms,upper1,'--',label = '1$\sigma$',lw = 1,color='black')
    plt.plot(Ms,lower1,'--',color='black',lw = 1)
    plt.plot(Ms,upper2,'--',label = '2$\sigma$',lw = .5,ms = 2,color='black')
    plt.plot(Ms,lower2,'--',color='black',lw=0.5)
    plt.title('z = '+str(z))
    plt.xlabel('M$_h$')
    plt.ylabel('m$_\star$')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
#    plt.savefig('abundanceMatchingPlots/z'+str(z)+'.png', bbox_inches='tight',dpi=1200)
    plt.show()


#plottingRoutine()            # randomly drawn masses at z = 0
#plottingRoutine(0, "random") # gives same as above
#plottingRoutine(1, "same")   # concatenates many trials (as specified) so you get a new draw for each trial, but the same gaussian draws are made for all M_halo in each different trial; z = 1



