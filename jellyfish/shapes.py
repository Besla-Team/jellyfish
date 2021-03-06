"""
Routines to compute halo shapes.
And plotting ellipsoids in 3d and 2d.

"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy import linalg
from scipy.spatial.transform import Rotation as R

def shells(pos, width, r, q, s):
    r_shell = np.sqrt(pos[:,0]**2.0 + pos[:,1]**2.0/q**2.0 + pos[:,2]**2.0/s**2.0)
    index_shell = np.where((r_shell<(r+width/2.)) & (r_shell>(r-width/2.)))
    pos_shell = pos[index_shell]
  
    return pos_shell

def volumes(pos, r, q, s):
    r_vol = np.sqrt(pos[:,0]**2.0 + pos[:,1]**2.0/q**2.0 + pos[:,2]**2.0/s**2.0)
    index_vol = np.where(r_vol<r)
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
    assert(np.shape(pos)[1]==3), "Wrong dimensions for pos, try pos.T"

    XX = np.sum(pos[:,0]*pos[:,0])
    XY = np.sum(pos[:,0]*pos[:,1])
    XZ = np.sum(pos[:,0]*pos[:,2])
    YX = np.sum(pos[:,1]*pos[:,0])
    YY = np.sum(pos[:,1]*pos[:,1])
    YZ = np.sum(pos[:,1]*pos[:,2])
    ZX = np.sum(pos[:,2]*pos[:,0])
    ZY = np.sum(pos[:,2]*pos[:,1])
    ZZ = np.sum(pos[:,2]*pos[:,2])

    shape_T = np.array([[XX, XY, XZ],
                        [YX, YY, YZ],
                        [ZX, ZY, ZZ]])
    return shape_T



def sort_eig(eigval, eigvec):
    """
    Sorts eigenvalues and eigenvectors in the following order:
    a: Major eigval
    b: Intermediate eigval
    c: Minor eigval

    The eigenvectors are sorted in the same way.
    See Zemp+11 (https://arxiv.org/abs/1107.5582) for the definiton of the eigen values.
    """

    oeival = np.argsort(eigval)
    a, b, c = eigval[oeival[2]], eigval[oeival[1]], eigval[oeival[0]]
    s = np.sqrt(c)/np.sqrt(a)
    q = np.sqrt(b)/np.sqrt(a)
    eigvec_sort = np.array([eigvec[oeival[2]], eigvec[oeival[1]], eigvec[oeival[0]]])
    return eigvec, np.array([a, b, c]), s, q


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

    assert eival[0] != 'nan', 'nan values'
    assert eival[1] != 'nan', 'nan values'
    assert eival[2] != 'nan', 'nan values'
    assert eival[0] != 0, 'zeroth value in eigval'
    assert eival[1] != 0, 'zeroth value in eigval'
    assert eival[2] != 0, 'zeroth value in eigval'

    eivec_s, eival_s, s, q = sort_eig(eival, evec)


    return eivec_s, eival_s, s, q


def iterate_shell(pos, r, dr, tol, return_pos=False):
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
    weight: weight factor in shape tensor. unity (0), r_ell**2 (1)
    Returns:
    -------
    eigvec: eigen vectors

    eigval: eigen values 
    
    s: c/a
    
    q: b/a 
    
    Npart : Number of particles in the shell

    """

    s_i = 1.0 #first guess of shape
    q_i = 1.0 #first guess of shape
    pos_s = shells(pos, dr, r, q_i, s_i)
    rot, axis, s, q = axis_ratios(pos_s)
    counter = 0

    while ((abs(s-s_i)>tol) & (abs(q-q_i)>tol)):
        s_i, q_i = s, q
	# TODO: do I need to rolate to the principal axis frame? 
        pos_s = shells(np.dot(rot, pos.T).T, dr, r, q_i, s_i)
        assert len(pos_s) > 0, 'Error: No particles shell'
        rot, axis, s, q = axis_ratios(pos_s)
        counter+=1
        if counter == 10000:
            s = 0
            q = 0
            print('to many iterations to find halo shape')
            break

    N_part = len(pos_s)
    if return_pos==False:
        return rot, np.sqrt(3*axis/len(pos_s)), s.real, q.real

    elif return_pos==True:
        return rot, np.sqrt(3*axis/len(pos_s)), s.real, q.real, pos_s

def iterate_volume(pos, r, tol):
    """
    Computes the halo axis rates (q,s)
    Where q=c/a and s=b/a
    Where a>b>c are the principal length of the axis.

    Parameters:
    -----------
    pos: numpy ndarray with the positions of the particles
    r: distance at which you want the shape
    tol: convergence factor

    Returns:
    s: c/a
    q: b/a

    """
    s_i = 1.0 #first guess of shape
    q_i = 1.0
    pos_s = volumes(pos, r, q_i, s_i)
    rot, axis, s, q = axis_ratios(pos_s)
    counter = 0
    while ((abs(s-s_i)>tol) & (abs(q-q_i)>tol)):
        s_i, q_i = s, q
        pos_s = volumes(np.dot(rot, pos.T).T, r, q_i, s_i)
        assert len(pos_s) > 0, 'Error: No particles in the volume' 
        rot, axis, s, q = axis_ratios(pos_s)
        counter +=1
        if counter >= 2000:
            s, q = [0.0, 0.0]
            print('to many iterations to find halo shape')
            break
    N_part = len(pos_s)
    return rot, (5*axis/N_part)**0.5, s.real, q.real

def ellipse_3dcartesian(axis, rotmatrix, center=[0,0,0]):
    """
    Return the 3d cartessian coordinates of an ellipsoid.

    Parameters:
    
    axis : 3d numpy.array
        length of the axis. Note this has to be in the same order as the
        rotation matrix.
    rotmatrix : numpy.ndarray
        Rotation matrix. i.e: eigen vectors of the shape tensor.
    
    center : 3d numpy.array
        coordinates of the center of the ellipsoid default ([0,0,0])
    
    Returns:
    --------

    xyz : numpy.ndarray
        coordinates of the ellipsoid in cartessian coordinates.
    
    """
    
    # Function taken from
    # https://github.com/aleksandrbazhin/ellipsoid_fit_python/ellipsoid_fit.py
                
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
                    
    # cartesian coordinates that correspond to the spherical
    # angles:
    x = axis[0] * np.outer(np.cos(u), np.sin(v))
    y = axis[1] * np.outer(np.sin(u), np.sin(v))
    z = axis[2] * np.outer(np.ones_like(u), np.cos(v))
    # This is the magic!! 
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotmatrix) + center

    return np.array([x, y, z]).T

def twod_surface(x, y):
    """
    2d surface from the border point of a set of points
    """
    assert(len(x)==len(y))
    pos = np.array([x, y]).T
    hull = ConvexHull(pos)
    x_s = list(pos[hull.vertices, 0])
    y_s = list(pos[hull.vertices, 1])
    x_s.append(x_s[0])
    y_s.append(y_s[0])
                                            
    return x_s, y_s
    
    
    
def rotate_zxy(vec, angles):
    """
    Rotate using euler angles in the following order: zxy
    input:
    ------
    vec: 3d vec
    angles : list of rotatation angles in the following zxy in degrees
        e.g: [30, 0, 0]  would rotate 30 degrees around the z axis
    output:
    -------
    rotated_vec : rotated vector 
    """
    
    rot = R.from_euler('zxy', angles, degrees=True)
    vec_rot = rot.apply(vec)
    
    return vec_rot
