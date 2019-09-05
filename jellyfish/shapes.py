import numpy as np
from scipy import linalg

"""
To-Do:

1. Rotation
2. Weight function
"""

def shells(pos, width, r, q, s):
    r_shell = np.sqrt(pos[:,0]**2.0 +pos[:,1]**2.0/q**2.0 +  pos[:,2]**2.0/s**2.0)
    index_shell = np.where((r_shell<(r+width/2.)) & (r_shell>(r-width/2.)))[0]
    pos_shell = pos[index_shell]
    return pos_shell

def volumes(pos, r, q, s):
    r_vol = np.sqrt(pos[:,0]**2.0 +pos[:,1]**2.0/q**2.0 +  pos[:,2]**2.0/s**2.0)
    index_vol = np.where(r_vol<r)[0]
    pos_vol = pos[index_vol]
    return pos_vol

#Computing the shape tensor
def shape_tensor(pos):
    """
    Compute the shape tensor as defined in Chua+18
    https://ui.adsabs.harvard.edu/abs/2019MNRAS.484..476C/abstract
    S_{ij} = 1/sum_{k}m_k  \sum_{k}1/w_k m_k r_{k,i} r_{k,j}
    For equal mass particles:


    S_{ij} = \sum_{k} k r_{k,i} r_{k,j}

    """
    assert(np.shape(pos)[1]==3), "Wrong dimensions for pos"
    shape_T = np.zeros([3, 3])
    npart = len(pos)
    shape_T = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            s = np.zeros(npart)
            for n in range(npart):
                s[n] = pos[n, i] * pos[n,j]
            shape_T[i][j] = sum(s)
    return shape_T

#Computing the axis ratios from the
#eigenvalues of the Shape Tensor
def axis_ratios(pos):
    """
    Computes the axis ratio of the ellipsoid defined by the eigenvalues of
    the Shape tensor.

    a = major axis
    b = intermediate axis
    c = minor axis
    The axis ratios are defined as:

    q = b/a
    s = c/a
    
    Parameter:
    ------
    pos : numpy.ndarray
        positions of the DM particles.

    Returns:
    -------
    s : double 
    q : double 
    """

    ST = shape_tensor(pos)
    eival, evec = linalg.eig(ST)
    oeival = np.sort(eival)
    c, b, a = oeival[2], oeival[1], oeival[0]
    s = np.sqrt(c/a)
    q = np.sqrt(b/a)

    return evec, [a, b, c], [s, q]

def iterate_shell(x, y, z, r, dr, tol):
    """
    Computes the halo axis rates (q,s)
    Where q=c/a and s=b/a
    Where a>b>c are the principal length of the axis.

    Parameters:
    -----------
    x, y, z: arrays with the positions of the particles
    r: distance at which you want the shape
    dr: Width of the shell
    tol: convergence factor

    Returns:
    s: c/a
    q: b/a

    """
    s_i = 1.0 #first guess of shape
    q_i = 1.0 #first guess of shape
    x_s, y_s, z_s = shells(x, y, z, dr, r, q_i, s_i)
    s_tensor = shape_tensor(x_s, y_s, z_s)
    rot_i, s, q = axis_ratios(s_tensor)
    while ((abs(s-s_i)>tol) & (abs(q-q_i)>tol)):
        s_i, q_i = s, q
        x_s, y_s, z_s = shells(x, y, z, dr, r, q_i, s_i)
        s_tensor = shape_tensor(x_s, y_s, z_s)
        rot, s, q = axis_ratios(s_tensor)
    return s, q

def iterate_volume(x, y, z, r, tol):
    """
    Computes the halo axis rates (q,s)
    Where q=c/a and s=b/a
    Where a>b>c are the principal length of the axis.

    Parameters:
    -----------
    x, y, z: arrays with the positions of the particles
    r: distance at which you want the shape
    tol: convergence factor

    Returns:
    s: c/a
    q: b/a

    """
    s_i = 1.0 #first guess of shape
    q_i = 1.0
    x_s, y_s, z_s = volumes(x, y, z, r, q_i, s_i)
    s_tensor = shape_tensor(x_s, y_s, z_s)
    rot_i, s, q = axis_ratios(s_tensor)
    counter = 0
    while ((abs(s-s_i)>tol) & (abs(q-q_i)>tol)):
        s_i, q_i = s, q
        x_s, y_s, z_s = volumes(x, y, z, r, q_i, s_i)
        s_tensor = shape_tensor(x_s, y_s, z_s)
        rot, s, q = axis_ratios(s_tensor)
        counter +=1
        if counter >=2000:
           s, q = [0.0, 0.0]
           break
    return s.real, q.real


