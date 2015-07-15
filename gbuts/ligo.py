"""LIGO event handling and galaxy catalog"""

import os
import numpy as np
import event
from geometry import *

__version__ = "$Id: ligo.py 222 2015-01-09 18:58:54Z lindy.blackburn@LIGO.ORG $"
__author__ = "Lindy Blackburn"
__email__ = "lindy.blackburn@ligo.org"

# load GWGC into array: line number [0:], phi [rad], theta [rad], MWEG luminosities, dist [Mpc], dist_err [Mpc]
def loadgwgc(gwgcfile='nasaem/catalog/GWGCCatalog.txt'):
    """
    GWGC catalog header
     0 - PGC
     1 - Name
     2 - RA
     3 - Dec
     4 - Type
     5 - App_Mag
     6 - Maj_Diam (a)
     7 - err_Maj_Diam
     8 - Min_Diam (b)
     9 - err_Min_Diam
    10 - b/a
    11 - err_b/a
    12 - PA
    13 - Abs_Mag
    14 - Dist
    15 - err_Dist
    16 - err_App_Mag
    17 - err_Abs_Mag

    modified GWGC in LOOCUP SVN (more bands and out to 200 Mpc)
     0 - PGC
     1 - Name
     2 - RA
     3 - Dec
     4 - Type
     5 - App_Mag_B
     6 - err_App_Mag_B
     7 - Abs_Mag_B
     8 - err_Abs_Mag_B
     9 - App_Mag_I
    10 - err_App_Mag_I
    11 - Abs_Mag_I
    12 - err_Abs_Mag_I
    13 - App_Mag_K
    14 - err_App_Mag_K
    15 - Maj_Diam (a)
    16 - err_Maj_Diam
    17 - Min_Diam (b)
    18 - err_Min_Diam
    19 - b/a
    20 - err_b/a
    21 - PA
    22 - Dist
    23 - err_Dist
    """
    table = [line.strip().split('|') for line in open(gwgcfile)][1:]
    if len(table[0]) == 19: # original GWGC (extra | at end of file makes one extra col)
        #  index   RA  DEC   MAG   DIST   DIST_ERR
        atable = np.array([[i, float(a[2]), float(a[3]), float(a[13]), float(a[14]), float(a[15])]  for (i, a) in enumerate(table) \
                 if a[13] != "~" and a[4] != "GC" and i != 42740]) # exclude no distance measured and globular clusters
    elif len(table[0]) == 25: # additional I, K columns and out to 200 Mpc
        atable = np.array([[i, float(a[2]), float(a[3]), float(a[7]), float(a[22]), float(a[23])]  for (i, a) in enumerate(table) \
                 if a[7] != "~" and a[4] != "GC"]) # exclude no distance measured and globular clusters
    atable[:,1] = np.pi * atable[:,1] / 12. # convert hours to radians
    atable[:,2] = np.pi/2 - atable[:,2] * np.pi/180. # convert declination (deg) to theta (rad)
    atable[:,3] = 10**((-20.5-atable[:,3])/2.5) # convert to luminosity (MWEG=-20.5 mag)
    maxfractionalerror = 0.22008889 # FIXME there is bug in current GWGC 200 Mpc catalog for PGC1424345, hard-code this in for now
    atable[:,5] = np.minimum(atable[:,4]*maxfractionalerror, atable[:,5])
    return atable

# pretty print info about galaxies based on list of ids
def gwgcdump(ids):
    if type(ids) is int or type(ids) is float:
        ids = [ids]
    gwgcfile='nasaem/catalog/GWGCCatalog.txt'
    lines = open(gwgcfile).readlines()
    cols = [a.strip() for a in lines[0].strip().split('|')]
    table = [line.strip().split('|') for line in lines[1:]]
    showcols = [1, 2, 3, 4, 13, 14]
    collen = [len(cols[i]) for i in showcols]
    for n in ids:
        for (i,j) in enumerate(showcols):
            collen[i] = max(collen[i], len(table[n][j]))
    fmt = "  ".join(["%%%ds" % clen for clen in collen])
    hstr = fmt % tuple(cols[i] for i in showcols)
    print hstr
    print "-" * len(hstr)
    for n in ids:
        print fmt % tuple(table[n][i] for i in showcols)

# filter gwgc catalog against inspnest distance and sky posteriors with given bisquare width (Mpc or degrees)
# distance window taken from GWGC distance error. skyres parameter should match effective statistical resolution of sky posterior.
# if gwgc, posteriorsamples, or skyhist are strings, they are interpreted as files and loaded (every time)
# returns: (doverlap, soverlap, luminosity) = how many overlap / expected random overlap in distance and sky location
def gwgcfilter(gwgc='nasaem/catalog/GWGCCatalog.txt', posteriorsamples='posterior_samples.dat', skyhist='ranked_sky_pixels.dat', skyres=3., format='lalinference'):
    if type(gwgc) is str:
        gwgc = loadgwgc(gwgc)
    if type(posteriorsamples) is str:
        posteriorsamples = loadposteriorsamples(posteriorsamples, format=format)
    if type(skyhist) is str:
        skyhist = loadrankedskypixels(skyhist)
    luminosity = gwgc[:,3]
    skyhist = skyhist[skyhist[:,2] > 0] # use only positive probability bins
    # calculate bisq overlap in distance, hardcoded 0-100 Mpc prior for inspnest
    (counts, bins) = np.histogram(posteriorsamples[:,0], range=(0., 100.), bins=400., weights=1/posteriorsamples[:,0]**2, normed=True) # impose 1/D^2 prior
    center = (bins[1:]+bins[:-1]) / 2.
    ddsq = ((gwgc[:,4,np.newaxis] - center[np.newaxis,:]) / np.sqrt(7) / gwgc[:,5,np.newaxis])**2 # distance sq (#gwgc x #dbins), bisq a=1 has variance 1/7
    bsq = (ddsq <= 1.) * (1 - ddsq)**2 # bisq calculation, goes to zero at np.sqrt(7) = 2.6457 sigma
    norm = np.sum(bsq, axis=1) # distance window is explicitly normalized here (sky window is not), bisq a=1 continuous integrates to 16/15
    norm += 1. * (norm == 0) # get rid of divide by zero (will be multiplied by zero anyway)
    doverlap = np.sum(bsq * counts[np.newaxis,:], axis=1) / norm * (len(bins)/np.sum(counts)) # last factor = 100 by construction
    # calculate bisq overlap in sky position
    ressq = (skyres*np.pi/180.)**2  # sq radians
    gxyz = pt2xyz(gwgc[:,1:3].T)    # xyz coords for galaxies: gxyz[0] = x's, gxyz[1] = y's, gxyz[2] = z's
    sxyz = pt2xyz(skyhist[:,0:2].T) # xyz coords for skyhist
    ddsq = np.sum((gxyz[:,:,np.newaxis] - sxyz[:,np.newaxis,:])**2 / ressq, axis=0) # (#xyz=3 x #gwgc x #skygrid), axis=0 gives x^2+y^2+z^2
    bsq = (ddsq <= 1.) * (1 - ddsq)**2
    soverlap = np.sum(bsq * skyhist[np.newaxis,:,2], axis=1) * (12./ressq) # actual 2d bisq area is pi/3, skyhist already normalized
    return (doverlap, soverlap, luminosity)

# filter GWGC using posteriorsamples and healpix binning and interpolation
# optional distance prior will set boundaries on distance KDE (otherwise bounds by posterior samples +/- 2*disance_error)
# this is different from gwgcfilter() because it builds its own skyhist using healpy
def gwgcfilter2(gwgc='nasaem/catalog/GWGCCatalog.txt', posteriorsamples='posterior_samples.dat', skyres=2., distprior=[0, 250]):
    import healpy
    if type(gwgc) is str:
        gwgc = loadgwgc(gwgc)
    if type(posteriorsamples) is str:
        posteriorsamples = loadposteriorsamples(posteriorsamples)
    weights = 1 / posteriorsamples[:,0]**2 # import 1/D^2 prior because galaxies are already D^2
    effectivesamples = np.sum(weights)
    luminosity = gwgc[:,3]
    # run the distance overlap
    fractionalerror = gwgc[:,5]/gwgc[:,4]
    minfractionalerror = np.min(fractionalerror)
    maxfractionalerror = np.max(fractionalerror)
    minlogspacing = np.log(1 + minfractionalerror) # the d(logdist) that produces the minimal fractional error
    maxlogspacing = np.log(1 + maxfractionalerror) # the d(logdist) that produces the minimal fractional error
    logspacing = minlogspacing / 4. # provide bin resolution according to minimum fractional error
    mindist = max(np.min(posteriorsamples[:,0]) * (1 - 2*maxfractionalerror), np.min(gwgc[:,4] - 2 * gwgc[:,5]), distprior[0], 0.05) # clip bins at 2-sigma or predefined boundaries
    maxdist = min(np.max(posteriorsamples[:,0]) * (1 + 2*maxfractionalerror), np.max(gwgc[:,4] + 2 * gwgc[:,5]), distprior[1])
    bins = np.exp(np.arange(np.log(mindist), np.log(maxdist) + logspacing, logspacing)) # construct bin edges
    (counts, bins) = np.histogram(posteriorsamples[:,0], bins=bins, weights=weights) # impose 1/D^2 prior
    center = (bins[1:] + bins[:-1]) / 2.
    width = bins[1:] - bins[:-1]
	# note there is no ampltidue normalization here, that will occur later with explicit sum over bsq
    ddsq = ((gwgc[:,4,np.newaxis] - center[np.newaxis,:]) / np.sqrt(7) / gwgc[:,5,np.newaxis])**2 # distance sq (#gwgc x #dbins), bisq a=1 has variance 1/7, so we make distance smaller
    bsq = (ddsq <= 1.) * (1 - ddsq)**2 # bisq calculation, goes to zero at np.sqrt(7) = 2.6457 sigma, this is supposed to represent a Gaussian-like window around galaxy distance
    norm = np.sum(bsq * width * effectivesamples / (distprior[1] - distprior[0]), axis=1) # what we expect overlap to be if samples distributed randomly over distance prior
    norm += 1. * (norm == 0) # get rid of divide by zero (will be multiplied by zero anyway)
    doverlap = np.sum(bsq * counts[np.newaxis,:], axis=1) / norm
    # run the sky overlap with healpy
    m = skyhist(posteriorsamples, skyres=skyres)
    soverlap = (float(len(m))/posteriorsamples.shape[0]) * healpy.get_interp_val(m, gwgc[:,2], gwgc[:,1]) # get smoothed map at GWGC locations and normalize
    return (doverlap, soverlap, luminosity)

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

# brute force closest point galaxy lookup
def nearestgalaxy(phi, theta, gwgc='nasaem/catalog/GWGCCatalog.txt'):
    if type(gwgc) is str:
        gwgc = loadgwgc(gwgc)
    testxyz  = pt2xyz((phi, theta))
    tablexyz = pt2xyz(gwgc[:,1:3].T)
    costh = np.dot(testxyz.T, tablexyz)
    maxidx = np.argmax(costh)
    return gwgc[maxidx]

# pick random galaxy index from catalog according to weights (default=use blue luminosity)
def randomgalaxy(gwgc='nasaem/catalog/GWGCCatalog.txt', weights=None):
    if type(gwgc) is str:
        gwgc = loadgwgc(gwgc)
    if weights is None:
        weights = gwgc[:,3]
    cumweights = weights.cumsum()
    ran    = cumweights[-1] * np.random.random()
    ranidx = np.nonzero(cumweights > ran)[0][0]
    return ranidx

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


def loadlalinftable(table):
    f = open(table)
    cols = zip(a.strip().split() for a in f)

def loadjobtable(table="jobtable.txt"):
# 2856031 969257067.328613 coinc_event:coinc_event_id:2856031    9.2347     94.060    (mchirp) L1,V1
# 37362 969039589.491699 coinc_event:coinc_event_id:3152310    7.8530 311871.742    (mchrp) H1,V1  0.5041 sim_inspiral:simulation_id:37362 36.8418
    f = open(table).readlines()
    cols = "id time ceid snr cfar mchirp ifos".split()
    injcols = cols + "dt simid dist".split()
    result = event.AttrDict()
    for line in f:
        tok = line.strip().split()
        tok[1] = float(tok[1])
        tok[3] = float(tok[3])
        tok[4] = float(tok[4])
        tok[5] = float(tok[5])
        if len(tok) > len(cols):
            tok[7] = float(tok[7])
            tok[9] = float(tok[9])
            if abs(tok[7]) < 0.02: # spurious matches
                a = event.AttrDict(zip(injcols[1:], tok[1:]))
                result[tok[0]] = a
        else:
            a = event.AttrDict(zip(cols[1:], tok[1:]))
            result[tok[0]] = a
    return result

# sim_inspiral:simulation_id:44032   1.973909 coinc_event:coinc_event_id:3325590  30.480258      0.000000   0.003489
# DEPRECATED
def loadsimtable(table='simtable.txt'):
    cols = "simid dist ceid newsnr cfar dt ifos".split()
    f = open(table)
    result = event.AttrDict()
    for line in f:
        tok = line.strip().split()
        tok[1] = float(tok[1])
        tok[3] = float(tok[3])
        tok[4] = float(tok[4])
        tok[5] = float(tok[5])
        if abs(tok[5]) < 0.02: # get rid of spurious matches
            a = event.AttrDict(zip(cols[1:], tok[1:]))
            result[tok[0]] = a
    return result

# coinc_event:coinc_event_id:2563636   9.317820     54.205913
# DEPRECATED
def loadceidtable(table='ceidtable.txt'):
    cols = "ceid newsnr cfar ifos".split()
    f = open(table)
    result = event.AttrDict()
    for line in f:
        tok = line.strip().split()
        tok[1] = float(tok[1])
        tok[2] = float(tok[2])
        a = event.AttrDict(zip(cols[1:], tok[1:]))
        result[tok[0]] = a
    return result

