"""coordinate transformations and quaternions"""

import numpy as np

__version__ = "$Id: geometry.py 143 2014-01-17 19:37:40Z lindy.blackburn@LIGO.ORG $"
__author__ = "Lindy Blackburn"
__email__ = "lindy.blackburn@ligo.org"

# np.array((x, y, z)) = pt2xyz((phi, theta))
def pt2xyz(p, t=None):
    if t is None:
        (p, t) = p
    return np.array((np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)))

# np.array((phi, theta)) = xyz2pt((x, y, z))
def xyz2pt(xyz):
    (x, y, z) = xyz
    t = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))
    p = np.fmod(np.arctan2(y, x)+2*np.pi, 2*np.pi)
    return np.array((p, t))

# keep phi, theta within standard [0, 2pi], [0, pi] bounds (used for direct indexing)
def normalizept(p, t=None):
    if t is None:
        (p, t) = p
    p = p % (2*np.pi)
    t = t % (2*np.pi)
    t = -(np.abs(t - np.pi) - np.pi)
    return np.array((p, t))

# cartesian normalization of unit vectors in N dimensions along first index
def normalize(x):
    norm = np.sqrt(np.sum(x**2, axis=0)) # sum along first index
    return x / norm # broadcasting will work since trailing dimensions are compared first

# spherical interpolation between two unit vectors 0 <= u <= 1
def slerp(x1, x2, u):
    costheta = np.sum(x1 * x2, axis=0)
    theta    = np.arccos(costheta)
    sintheta = np.sin(theta)
    A = np.sin((1. - u)*theta) / sintheta
    B = np.sin(u * theta) / sintheta
    return A * x1 + B * x2

# quarterion routines
# http://www.genesis3d.com/~kdtop/Quaternions-UsingToRepresentRotation.htm
# http://www.j3d.org/matrix_faq/matrfaq_latest.html
# http://www.itk.org/CourseWare/Training/QuaternionsI.pdf
# routines work on a single quaternion: q = (w, x, y, z) with w, x, y, z floats
# or N quaternions q = np.array((w, x, y, z)) where w, x, y, z are each an array of size N
# this is computationally less efficient, but cleaner to code than using (q1, q2, q3..)

# multiply quaternion q1 by q2 (not commutative): q = (w, x, y, z)
# w, x, y, z can be arrays of coefficients (i.e. q[0] is array of w values)
def quatmultiply(q1, q2):
    (w1, x1, y1, z1) = q1
    (w2, x2, y2, z2) = q2
    return np.array((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2+y1*w2+z1*x2-x1*z2, w1*z2+z1*w2+x1*y2-y1*x2))

# conjugate of quaternion: q -> q' = (w, -x, -y, -z)
# w, x, y, z can be arrays of coefficients (i.e. q[0] is an array of w values)
# for a rotation (unit) quaternion, this amounts to theta -> -theta about the same axis
def quatconjugate(q):
    return np.array((q[0], -q[1], -q[2], -q[3]))

# q and -q represent the same rotation.
# for a timeseries of quaternion rotations, maintain positive dot product between
# elements to avoid a discontinuous flip which is bad for interpolation.
def quatfix(q):
    if len(q[0]) == 1: # no sense running on just 1 quaternion
        return q
    q = np.array(q)
    dot = np.sum(q[:,:-1] * q[:,1:], axis=0)
    flip = np.ones(len(q[0]))
    flip[np.nonzero(dot < 0)[0] + 1] = -1
    return q * np.cumprod(flip)[np.newaxis,:]

# apply quaternion rotation q to X
# use convenion X' = q X q*
def quatapply(q, x):
    if len(x) == 3:
        x0 = 0
        (x1, x2, x3) = x
    else:
        (x0, x1, x2, x3) = x
    y = quatmultiply(quatmultiply(q, (x0, x1, x2, x3)), quatconjugate(q))
    if len(x) == 3:
        return np.array((y[1], y[2], y[3]))
    else:
        return np.array(y)

# xyz rotation matrix from normalized quaternion, R rotates a column vector: [x'; y'; z'] = R * [x; y; z]
# w, x, y, z can be arrays of coefficients (i.e. q[0] is an array of w values)
# to convert output to an array of matrices rather than a matrix of arrays of coefficients,
#   use np.rollaxis(quat2rotationmatrix(q), -1, 0) or quat2rotationmatrix(q).transpose((2, 0, 1))
# not that np.dot cannot be used to apply a list of matrices to a list of vectors, it must be done manually with np.sum(a*b)
def quat2rotationmatrix(q):
    (w, x, y, z) = q
    xx = x**2; yy = y**2; zz = z**2
    wx = w*x;  wy = w*y;  wz = w*z
    xy = x*y;  xz = x*z;  yz = y*z
    return np.array(((1-2*(yy+zz), 2*(xy-wz), 2*(wy+xz)),
                     (2*(wz+xy), 1-2*(xx+zz), 2*(yz-wx)),
                     (2*(xz-wy), 2*(wx+yz), 1-2*(xx+yy))))

# re-normalized linear interpolation between quaternion rotations q1 and q2 where 0 <= u <= 1
# quatnlerp(q1, q2, 0) = q1, quatnlerp(q1, q2, 1) = q1
# u, w, x, y, z can be arrays of coefficients (i.e. q[0] is an array of w values)
# for interpolation over a sampling (t, q(t)) at tnew it may be convenient to use scipy instead:
#   normalize(scipy.interpolate.interp1d(t, np.array((w(t), x(t), y(t), z(t))))(tnew))
def quatnlerp(q1, q2, u):
    return normalize((1. - u) * q1 + u * q2)

# convert hh:mm:ss RA string to radians [0, 2pi], assuming all positive values
def hms2rad(hms):
    (h, m, s) = hms.split(':')
    return (((float(s)/60. + float(m))/60.) + float(h))/24. * 2*np.pi

# convert degrees:arcmin:arcsec DEC string to degrees [-90, 90]
def dms2deg(dms):
    (d, m, s) = map(float, dms.split(':'))
    if d >= 0:
        return d + m/60. + s/3600.
    else:
        return d - m/60. - s/3600.

# convert hh:mm:ss RA string to degrees [0, 360], assuming all positive values
def hms2deg(hms):
    (h, m, s) = hms.split(':')
    return (((float(s)/60. + float(m))/60.) + float(h))/24. * 360.

# convert degrees to h:m:s
def deg2hms(deg):
    hours = deg * 24./360.
    h = np.floor(hours)
    m = np.floor((hours - h) * 60.)
    s = (hours - h - m/60.)*3600.
    return (h, m, s)

# convert degrees to deg:amin:asec
def deg2dms(deg):
    sig = np.sign(deg)
    deg = sig * deg
    d = np.floor(deg)
    m = np.floor((deg - d) * 60.)
    s = (deg - d - m/60.)*3600.
    return (sig*d, sig*m, sig*s)

def pt2radec(p, t=None):
    if t is None:
        (p, t) = p
    ra = p * 180./np.pi
    dec = np.pi/2 - t
    return (deg2hms(ra), deg2dms(dec))

def j20002pt(j2000): # J191910.2+440936, J170516.0-821045
    import re
    j2000 = j2000.replace('.', '')
    m = re.match('J(\d+)([\+-]\d+)', j2000)
    (r, d) = m.groups()
    if len(r) > 6:
        r = r[:6] + '.' + r[6:]
    if len(d) > 7:
        d = d[:7] + '.' + d[7:]
    rrad = hms2rad(r[0:2] + ':' + r[2:4] + ':' + r[4:])
    drad = (90 - dms2deg(d[0:3] + ':' + d[3:5] + ':' + d[5:])) * np.pi/180.
    return np.array([rrad, drad])

def j2000sep(j2000a, j2000b): # angular separation in degrees between two j2000 cooords
    p1 = pt2xyz(j20002pt(j2000a))
    p2 = pt2xyz(j20002pt(j2000b))
    return 180 * np.arccos(np.dot(p1, p2)) / np.pi


