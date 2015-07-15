"""LIGO event and parameter estimation handling"""

import os
import numpy as np
import event
from geometry import *

__version__ = "$Id: ligo.py 222 2015-01-09 18:58:54Z lindy.blackburn@LIGO.ORG $"
__author__ = "Lindy Blackburn"
__email__ = "lindy.blackburn@ligo.org"

# histogram posterior samples into healpix grid, using skyres in degress as smoothed resolution
# returns healpix map
def skyhist(posteriorsamples='posterior_samples.dat', skyres=2.):
    import healpy
    if type(posteriorsamples) is str:
        posteriorsamples = loadposteriorsamples(posteriorsamples)
    baseres = (skyres / 4.) * np.pi/180. # require this resolution from healpix grid
    nside = 2**int(min(10, np.ceil(np.log2(healpy.nside2resol(1) / baseres)))) # don't go past 2**10 (12.6 million pixels)
    npix = healpy.nside2npix(nside)
    p = healpy.ang2pix(nside, posteriorsamples[:,2], posteriorsamples[:,1]) # convert (theta, phi) to healpix numbers
    # n = np.bincount(p, minlength=npix) # bin the samples
    n = np.zeros(npix)
    n[:max(p)+1] = np.bincount(p) # support old version of numpy that does not have minlength argument
    m = healpy.smoothing(n, sigma=skyres*np.pi/180.) # smoothed map
    return m

# place 2D Gaussian sky patch distribution at (phi, theta) with sigma=radius in radians
# optional skyres in degrees determines map resolution (at 4x skyres)
def skypatch(phi, theta, radius, skyres=2.):
    import healpy
    baseres = (skyres / 4.) * np.pi/180. # require this resolution from healpix grid
    nside = 2**int(min(10, np.ceil(np.log2(healpy.nside2resol(1) / baseres)))) # don't go past 2**10 (12.6 million pixels)
    npix = healpy.nside2npix(nside)
    p = healpy.ang2pix(nside, theta, phi)
    n = np.zeros(npix)
    n[p] = 1. # put all probability at (p, t)
    m = healpy.smoothing(n, sigma=radius) # smooth out Gaussian
    return m

# confidence area (sqdeg) for healpix map
def confidencearea(map, confidence=.9):
	import healpy
	isort = np.argsort(map)[::-1] # indices to sort from most to least probable
	normed = np.cumsum(map[isort])
	normed = normed / normed[-1]
	fourpi = 4 * (180)**2 / np.pi # number of sq degrees in sphere
	area = ((1. + np.nonzero(normed >= confidence)[0][0]) / len(normed)) * fourpi
	return area

# amount of area to search before getting to location (phi, theta) in healpix map
def searchedarea(map, phi, theta):
	import healpy
	nside = healpy.npix2nside(len(map))
	iloc = healpy.ang2pix(nside, theta, phi)
	fourpi = 4 * (180)**2 / np.pi # number of sq degrees in sphere
	area = (float(np.sum(map >= map[iloc])) / len(map)) * fourpi
	return area

# amount of probability to search before getting to location (phi, theta) in healpix map
def searchedprob(map, phi, theta):
	import healpy
	nside = healpy.npix2nside(len(map))
	iloc = healpy.ang2pix(nside, theta, phi)
	prob = (np.sum(map[map >= map[iloc]]) / np.sum(map))
	return prob

# load inspnest posterior samples into array: dist (Mpc), phi, theta, iota
# attempts to find correct table format by available columns (see code)
def loadposteriorsamples(file = 'posterior_samples.dat'):
    header = open(file).readline().lstrip('#').strip().split()
    col = dict((lab, i) for (i, lab) in enumerate(header))
    if 'dist' in col:
        btable = np.loadtxt(file, usecols=(col['dist'], col['ra'], col['dec'], col['cosiota']), skiprows=1)
        btable[:,3] = np.arccos(btable[:,3])
    elif 'inclination' in col:
        btable = np.loadtxt(file, usecols=(col['distance'], col['rightascension'], col['declination'], col['inclination']), skiprows=1)
    elif 'iota' in col:
        btable = np.loadtxt(file, usecols=(col['distance'], col['rightascension'], col['declination'], col['iota']), skiprows=1)
    else:
        print "unknown format"
        return None
    btable[:,2] = np.pi/2 - btable[:,2] # convert dec to spherical coords
    btable[:,3] = np.pi/2 - np.abs((np.pi/2 - btable[:,3])) # symmetrize iota
    return btable

# load probability skymap: phi, theta, prob, cumul
def loadrankedskypixels(file = 'ranked_sky_pixels.dat'):
    try:
        ctable = np.loadtxt(file, comments="#") # dec(deg.) ra(h.)  prob.   cumul.  OR  # dec(rad) ra(rad) cumul
    except:
        ctable = np.loadtxt(file, comments="d") # no # in comment line
    if ctable.shape[1] == 4:
        (ctable[:,0], ctable[:,1]) = (ctable[:,1] * np.pi/12., np.pi/2 - ctable[:,0] * np.pi/180.)
    elif ctable.shape[1] == 3:
        new = np.zeros((ctable.shape[0], 4))
        (new[:,0], new[:,1]) = (ctable[:,1], np.pi/2 - ctable[:,0])
        new[:,3] = ctable[:,2]
        new[0,2] = ctable[0,2]
        new[1:,2] = np.diff(ctable[:,2])
        ctable = new
    else:
        raise Exception("bad format")
    return ctable

# reduced chisq definition (lsctables)
# return self.get_column('chisq') / (2*self.get_column('chisq_dof') - 2)
def rchisq(chisq, dof):
    return(chisq / (2*dof - 2))

# definition of NEWSNR (default pars)
# note that we need the chisq_dof, which is usually(?) 16 for ihope
# note that we do NOT send the reduced chisq to this function
# nhigh = 2.
# newsnr = snr/ (0.5*(1+rchisq**(index/nhigh)))**(1./index)
# numpy.putmask(newsnr, rchisq < 1, snr)
def newsnr(snr, chisq, chisq_dof=16):
    return snr / np.maximum(1.0,(1 + (rchisq(chisq, chisq_dof))**3)/2.)**(1./6.)

# definition of EFFECTIVE SNR (lsctables)
# return snr/ (1 + snr**2/fac)**(0.25) / rchisq**(0.25)
def effsnr(snr, chisq, chisq_dof=16):
    return snr / (1. + snr**2/250.)**(0.25) / (rchisq(chisq, chisq_dof))**(0.25)

# note in ligolw_thinca_to_coinc we have:
#  if options.statistic == "new_snr":
#      coinc_inspiral.snr = math.sqrt(sum(event.get_new_snr(index = chisq_index)**2 for event in events))
#  elif options.statistic == "effective_snr":
#      coinc_inspiral.snr = math.sqrt(sum(event.get_effective_snr(fac = effective_snr_factor)**2 for event in events))
