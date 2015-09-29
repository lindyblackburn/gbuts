"""handling and analysis of Fermi data"""

import os
import numpy as np
from geometry import *
from clock import *

__version__ = "$Id: gbm.py 230 2015-02-08 21:48:21Z lindy.blackburn@LIGO.ORG $"
__author__ = "Lindy Blackburn"
__email__ = "lindy.blackburn@ligo.org"

# names of GBM detectors
nlist = "n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 na nb".split()
blist = "b0 b1".split()
dlist = nlist + blist

# common locations of data files
fermidir = "./data/"
datalocations = """
./data/
/data/gbm/daily/
""".strip().split()
for testdir in datalocations:
    if os.path.isdir(testdir):
        fermidir = testdir
        break

# common locations of response files
# common locations of data files
respdir = "./response/"
resplocations = """
./response/
/data/gbm/response/
""".strip().split()
for testdir in resplocations:
    if os.path.isdir(testdir):
        respdir = testdir
        break

# return indeces for mc response in Valerie's tables, (phi, theta) in radians
def mcrespidx(phi, theta=None):
    if theta is None:
        (phi, theta) = phi
    (phi, theta) = normalizept((phi, theta))
    tcol = (theta * 180/np.pi + 0.5).astype(int) # round to nearest integer for theta column
    adist = np.sin(np.arange(1, 180) * np.pi/180.)   # angular distance around axis in 2*pi radians
    nphi  = np.hstack((1, (adist * 360 + 1e-8).astype(int), 1)) # some python roundoff error at 30 degrees
    tidx  = np.cumsum(np.hstack((0, nphi[:-1]))) # starting index of theta section
    phiidx = np.fmod((0.5 + nphi[tcol] * phi / (2*np.pi)).astype(int), nphi[tcol]) # phi index in theta section
    return tidx[tcol] + phiidx

# return indices in resp table within a certain dtheta (degrees) error radius
def mcrespidxwithin(pt, dtheta):
    testxyz  = pt2xyz(pt)
    tablexyz = pt2xyz(respgrid())
    costh = np.dot(testxyz, tablexyz)
    return(np.nonzero(np.sum(testxyz[:,np.newaxis] * tablexyz, axis=0) > np.cos(dtheta * np.pi/180.))[0])

# GBM detector array geometry in spherical coords
naitable = """
 0  45.9  20.6
 1  45.1  45.3
 2  58.4  90.2
 3 314.9  45.2
 4 303.2  90.3
 5   3.4  89.8
 6 224.9  20.4
 7 224.6  46.2
 8 236.6  90.0
 9 135.2  45.6
10 123.7  90.4
11 183.7  90.3
"""
naitablearray = np.array([map(float, line.strip().split()) for line in naitable.strip().split("\n")])
naipt = naitablearray[:,1:3].T * np.pi/180. # angle clockwise from spacecraft x axis, angle from z axis
naixyz = pt2xyz(naipt)

bgoxyz = np.array([[1, -1], [0, 0], [0, 0]])
bgopt = xyz2pt(bgoxyz)

# list of strong sources from occultation study
occtable = """
V0332+53,53.8,53.2
Crab,83.6,22.0
A0535+262,84.7,26.3
MXB 0656-072,104.6,-7.3
Vela X-1,135.5,-40.6
Cen X-3,170.3,-60.7
GX301-2,186.7,-62.8
Cen A,201.4,-43.0
2S 1417-624,215.3,-62.7
Sco X-1,245.0,-15.6
Her X-1,254.5,35.3
MAXI J1659-152,254.8,-15.3
GX 339-4,255.7,-48.7
4U 1700-377,256.0,-37.8
XTE J1752-223,268.1,-22.3
GRS 1915+105,288.8,11.0
XTE J1946+274,296.4,27.4
Cyg X-1,299.6,35.2
Cyg X-3,308.1,41.0
GX 304-1,324.9,57.0
"""
occdict = dict()
for line in occtable.strip().split('\n'):
    tok = line.split(',')
    occdict[tok[0]] = (float(tok[1]) * np.pi/180., (90.-float(tok[2])) * np.pi/180.)

strongsources = "Crab,Sco X-1,Cyg X-1".split(',')

# load swift table name:(time, ra, dec, t90)
def loadswifttable(filename = 'catalog/grb_table_1334177322.txt'):
    f = open(filename, 'r')
    # get rid of some known formatting problems and errors with swift table
    s = f.read().replace('\r\n', '').replace('~', '').replace('>','').replace('<','').replace('. PL', ', PL').replace('TBD', 'n/a').replace(' (PL)', ', PL').replace('081024A\t05:53:0\t','081024A\t05:53:08\t')
    f.close()
    table = s.strip().split('\n')
    cols   = table[0].strip().split('\t')
    table = [dict(zip(cols, line.split('\t'))) for line in table[1:]]
    # select BAT detections with BAT or XRT locations
    table = [line for line in table if line['BAT T90 [sec]'] != "n/a" and \
      (line['BAT RA (J2000)'] != "n/a" or line['XRT RA (J2000)'] != "n/a")]
    good = dict()
    for line in table:
        (name, time) = (line['GRB'], line['Time [UT]'])
        if line['XRT RA (J2000)'] != "n/a":
            (ra, dec) = (hms2rad(line['XRT RA (J2000)']), dms2deg(line['XRT Dec (J2000)'])*np.pi/180.)
        else:
            (ra, dec) = (float(line['BAT RA (J2000)']) * np.pi/180., float(line['BAT Dec (J2000)'])*np.pi/180.)
        t90 = float(line['BAT T90 [sec]'])
        good[name] = [time, ra, dec, t90]
    return good

def loadfermitable(filename = "catalog/fermitable.txt"):
    table = [line.strip().split('|')[1:-1] for line in open(filename) if line[0] == '|']
    cols = [key.strip() for key in table[0]]
    table = [dict(zip(cols, row)) for row in table[1:]]
    return table

def loadflaretable(filename = "catalog/fermi_gbm_flare_list.txt"):
    lines = open(filename, 'r').readlines()
    lines = [a[1:57] + ' ' + a[57:64] + ' ' + a[64:87] for a in lines if a[7] == "_"] # grep data table and fix catalog error
    lines = [a.strip().split() for a in lines]
    table = []
    for line in lines:
        name  = line[0]
        start = parsetime(line[1] + ' ' + line[2])
        peak  = parsetime(line[1] + ' ' + line[3])
        stop  = parsetime(line[1] + ' ' + line[4])
        dur   = float(line[5])
        peak  = float(line[6])
        total = float(line[7])
        d1    = line[8]
        d2    = line[9]
        table.append([name, start, peak, stop, dur, peak, total, d1, d2])
    return table

# create a new data dictionary which covers GBM data for all detectors with met0 <= TIME < met1
# uses less memory than loading in entire day of data (if all data is touched)
# which is necessary for running on clster nodes with 700 MB vmem limit
# also traverses day boundary, and avoids overlap
# returns dictionary of requested detector data in fits-like rec-array structure
# fork=True will fork a python process to read data (saves some memory)
# fork=False will try to send the data over STDOUT (not meant to use interactively)
# fork=None will return data without forking the process
# dlist: list of detectors to load (e.g. ['n6', 'n7']
# poshist=True: load poshist data into key data['poshist']
def precachedata(met0, met1, dlist=dlist, poshist=True, fermidir=fermidir, fork=None):
    if fork is True: # fork process to build cache
        import subprocess, cPickle
        proc = subprocess.Popen(['python', '-c', \
           "import gbuts.gbm; gbuts.gbm.precachedata(%f, %f, dlist=%s, poshist=%s, fermidir=%s, fork=False)" \
           % (met0, met1, dlist, poshist, repr(fermidir))], stdout=subprocess.PIPE)
        cache = cPickle.load(proc.stdout)
    else: # load fits files to build cache
        if met1 < met0:
            met1 = met0 + met1
        # dictionaries of start and end times for data collection by name
        (spect0, spect1) = (dict.fromkeys(dlist, met0), dict.fromkeys(dlist, met1))
        (posht0, posht1) = (met0-1., met1+1.)
        gtit0 = dict.fromkeys(dlist, 0)
        import pyfits, glob
        cache = dict()
        day = fermi2utc(met0).date()
        speccols = "COUNTS EXPOSURE QUALITY TIME ENDTIME".split()
        spectype = "8float64 float64 int16 float64 float64".split()
        gticols  = "START STOP".split()
        gtitype  = "float64 float64".split()
        poshcols = "SCLK_UTC QSJ_1 QSJ_2 QSJ_3 QSJ_4 WSJ_1 WSJ_2 WSJ_3 POS_X POS_Y POS_Z".split()
        poshtype = "float64 float64 float64 float64 float64 float64 float64 float64 float64 float64 float64".split()
        for d in dlist:
            cache[d] = {'SPECTRUM':[], 'GTI':[]}
        if poshist:
            cache['poshist'] = {'GLAST POS HIST':[]}
        while day <= fermi2utc(met1).date():
            datadir = day.strftime(fermidir + "%Y/%m/%d/current")
            basename = day.strftime("%y%m%d")
            for d in dlist:
                files = glob.glob(datadir + '/glg_ctime_' + d + '_' + basename + '_v*.pha')
                if not files:
                    import sys
                    sys.stderr.write('missing file: ' + datadir +  'glg_ctime_' + d + '_' + basename + '_v*.pha\n')
                    continue
                filename = sorted(files)[-1]
                data = pyfits.open(filename)
                spectable = data['SPECTRUM']
                i = np.flatnonzero((spectable.data['TIME'] >= spect0[d]) & (spectable.data['TIME'] < spect1[d]))
                if len(i) == 0:
                    continue
                idx = (np.min(i), np.max(i))
                specdata = spectable.data[idx[0]:idx[1]+1]
                spect0[d] = specdata['TIME'][-1] + 1e-6 # enforce no overlap
                speccopy = np.zeros(1+idx[1]-idx[0], dtype={'names':speccols, 'formats':spectype})
                for col in speccols:
                    speccopy[col] = specdata.field(col)
                cache[d]['SPECTRUM'].append(speccopy)
                gtitable = data['GTI']
                gticopy = np.zeros(len(gtitable.data), dtype={'names':gticols, 'formats':gtitype})
                for col in gticols:
                    gticopy[col] = np.maximum(gtit0[d], gtitable.data.field(col))
                gtit0[d] = max(gtit0[d], gticopy[gticols[-1]][-1])
                cache[d]['GTI'].append(gticopy)
                data.close()
                (data, spectable, specdata) = (None, None, None)
            if poshist:
                files = glob.glob(datadir + '/glg_poshist_all_' + basename + '_v*.fit')
                if not files:
                    import sys
                    sys.stderr.write('missing file: ' + datadir + '/glg_poshist_all_' + basename + '_v*.fit\n')
                else:
                    filename = sorted(files)[-1]
                    data = pyfits.open(filename)
                    poshtable = data['GLAST POS HIST']
                    i = np.flatnonzero((poshtable.data['SCLK_UTC'] >= posht0) & (poshtable.data['SCLK_UTC'] < posht1))
                    if len(i) == 0:
                        continue
                    idx = (np.min(i), np.max(i))
                    poshdata = poshtable.data[idx[0]:idx[1]+1]
                    posht0 = poshdata['SCLK_UTC'][-1] + 1e-6 # enforce no overlap
                    poshcopy = np.zeros(1+idx[1]-idx[0], dtype={'names':poshcols, 'formats':poshtype})
                    for col in poshcols:
                        poshcopy[col] = poshdata.field(col)
                    cache['poshist']['GLAST POS HIST'].append(poshcopy)
                    data.close()
                    (data, poshtable, poshdata) = (None, None, None)
            day += datetime.timedelta(1) # add 1 day
    if fork is False: # send cache over to stdout
        import sys, cPickle
        cPickle.dump(cache, sys.stdout, protocol=2)
        return None
    else: # fork is True or fork is None
        import sys
        import collections
        Datastruct = collections.namedtuple('Datastruct', 'data') # provide ".data" attribute to mimic pyfits object
        try:
            for d in dlist:
                cache[d]['SPECTRUM'] = Datastruct(data=np.hstack(cache[d]['SPECTRUM']).view(np.recarray))
                cache[d]['GTI'] = Datastruct(data=np.hstack(cache[d]['GTI']).view(np.recarray))
            if poshist:
                cache['poshist']['GLAST POS HIST'] = \
                  Datastruct(data=np.hstack(cache['poshist']['GLAST POS HIST']).view(np.recarray))
            return cache
        except:
            print sys.exc_info() # probably out of GTI or no data
            return None

# load GBM daily data for UTC datetime (download if necessary)
def loaddailydata(timestruc, dlist=dlist, poshist=True):
    import pyfits, glob
    if type(timestruc) is not datetime.datetime:
        timestruc = parsetime(timestruc)
    data = dict()
    datadir = timestruc.strftime(fermidir + "%Y/%m/%d/current")
    basename = timestruc.strftime("%y%m%d")
    if not glob.glob(datadir + '/glg_ctime_nb_' + basename + '_v*.pha'):
        return None # skip downloading missing data
        # cmd = "wget -nv -nH --cut-dirs=3 -P /data -r -l0 -c -N -np -R 'index*,*tte*.fit' -erobots=off --retr-symlinks ftp://heasarc.gsfc.nasa.gov/FTP/fermi" + datadir
        cmd = "wget -nv -nH --cut-dirs=5 -P /data/gbm/daily/ -r -l0 -c -N -np -A '*ctime*,*poshist*' -erobots=off --retr-symlinks ftp://heasarc.gsfc.nasa.gov/FTP/fermi" + datadir
        os.system(cmd)
    for d in dlist:
        filename = sorted(glob.glob(datadir + '/glg_ctime_' + d + '_' + basename + '_v*.pha'))[-1]
        data[d] = pyfits.open(filename)
    filename = sorted(glob.glob(datadir + '/glg_poshist_all_' + basename + '_v*.fit'))[-1]
    if poshist:
        data['poshist'] = pyfits.open(filename)
    return data

# (phi, theta) grid designed to match up with the ones in GBM response tables (in radians)
# while table values are rounded to the nearest degree, the actual response was calculated at arcmin precision
# except for the channel-by-channel direct response which is at the rounded resolution
def respgrid():
    theta = np.arange(1, 180)
    adist = np.sin(theta * np.pi/180.)   # angular distance around axis in 2*pi radians
    nphi  = np.floor(adist * 360 + 1e-8) # some python roundoff error at 30 degrees
    rows  = [[0, 0]]                     # initial point at north pole (0, 0)
    for (t, n) in zip(theta, nphi):      # go down the rows
        phi = np.linspace(0, 360, n, endpoint=False) # evenly spaced snapped to 1deg grid
        # phi = np.round(np.linspace(0, 360, n, endpoint=False)) # evenly spaced snapped to 1deg grid
        rows += [[p, t] for p in phi]
    rows += [[0, 180]]                   # final point at south pole (0, 180)
    # a = np.array(rows) * 60              # return in arcminutes
    # there are numerical round-off artifacts in original tables at dec = 69, 78, 102, 111 degrees
    # which happen at 7+28n of 336 steps in 15/14 and 66+44n of 352 steps in 45/44
    # in the actual response matrices, this results in rounding down from *7.5 at those locations
    # note that in general, numpy will (correctly) round 0.5 to the nearest even integer
    # roundoff = [13040, 13068, 13096, 13124, 13152, 13180, 13208, 13236, 13264, 13292, 13320, \
    #             13348, 16191, 16279, 16367, 16455, 24757, 24845, 24933, 25021, 27806, 27834, \
    #             27862, 27890, 27918, 27946, 27974, 28002, 28030, 28058, 28086, 28114]
    # a[roundoff,0] -= 60 # we implement this numerical error here
    # return a
    return np.array(rows).T * np.pi/180.

# load only GBM poshist data for UTC datetime
def loadposhist(timestruc):
    import pyfits, glob
    data = dict()
    if type(timestruc) is not datetime.datetime:
        timestruc = parsetime(timestruc)
    data = dict()
    datadir = timestruc.strftime(fermidir + "%Y/%m/%d/current")
    basename = timestruc.strftime("%y%m%d")
    filename = sorted(glob.glob(datadir + '/glg_poshist_all_' + basename + '_v*.fit'))[-1]
    data['poshist'] = pyfits.open(filename)
    return data

# fit GBM data at central time t and total duration to calculate estimated background and foreground counts
# if poiss is defined, true poisson statistics will be used instead of least squares. this can be slow.
# dlist = ['n0', 'n1', ...'], data is a dictionary of pyfits data: data['n0'], etc from loaddailydata()
# injection is a tuple of (injt, injdur, injrate) where injrate[d] is the rate vector for detector d
# fitconf adds error bars for the fit
# dqshade will tag the corresponding plots where True, shape (nchan, ndetectors)
def gbmfit(data, t, duration, dlist=dlist, channels=[0,1,2,3,4,5,6,7], poiss=False, plot=False, degree=None, fitsize=10., plotsize=5.,
           fitqual=False, rebin=None, crveto=True, injection=None, fitconf=False, dqshade=None):
    import fit
    # split dlist and channels if necessary
    if type(dlist) is str:
        dlist = dlist.split()
    if type(channels) is str:
        channels = channels.split()
        channels = map(int, channels)
    if rebin is None and duration > 1.025: # auto set rebin for long durations
        rebin = True
    (nd, nc) = (len(dlist), len(channels))
    if plot and dqshade is None:
        dqshade = np.zeros((nc, nd))
    fg = np.zeros((nc, nd))     # measured counts in fg window
    bg = np.zeros((nc, nd))     # background estimate in fg window
    goodfit = np.ones((nc, nd)) # boolean True for good fit (low chisq)
    vbfit = np.zeros((nc, nd))  # variance in bg estimate due to fit parameter errors       
    vbsys = np.zeros((nc, nd))  # variance in bg estimate due to systematic error from polynomial model
    xsqdof = np.zeros((nc, nd)) # chisq DOF of bg fit
    if degree == None:
        degree = max(2, 1+np.ceil(np.log2(duration)/2.))
    if plot:
        import plots
        plots.importmpl()
        plt = plots.plt
        import itertools
        from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
        plt.clf()
        ymin = 1e6*np.ones((2,nc))
        ymax = np.zeros((2,nc))
    for (i, d) in enumerate(dlist):  # i index detectors, j index channels
        ### grab data
        if data is None:
            data = loaddailydata(fermi2utc(t))
        specdata = data[d]['SPECTRUM'].data  # look at data for detector
        tcent    = (specdata['TIME'] + specdata['ENDTIME']) / 2 - t           # relative mean time
        idx      = np.nonzero((specdata['QUALITY'] == 0) & (specdata['EXPOSURE'] > 0) & (np.abs(tcent) < fitsize*max(0.512, duration)))[0] # local data idx, at least 80 points
        if len(idx) == 0:
            return (None, None) if not fitqual else (None, None, None, None, None, None)
        tcent    = tcent[idx]
        tdur     = specdata['EXPOSURE'][idx]                                          # exposure [s]
        counts   = specdata['COUNTS'][idx][:,channels]                                # counts by channel
        if injection is not None:
            (injt, injdur, injrate) = injection
            injstart = injt - injdur/2.
            injend = injt + injdur/2.
            overlap = np.minimum(specdata['ENDTIME'][idx], injend) - np.maximum(specdata['TIME'][idx], injstart)
            overlap = overlap * (overlap > 0)
            counts += overlap[:,np.newaxis] * injrate[d][np.newaxis,:] # getting injrate for detector i, all channels, index order same as response matrix
        fgidx    = np.nonzero(np.abs(tcent) <= duration/2.)[0]                        # foreground indices
        ### cosmic ray rejection, not this removes the first two and last two points
        if crveto:
            nearcounts   = counts[:-4,:] + counts[1:-3,:] + counts[3:-1,:] + counts[4:,:]       # nearby counts
            neartdur     = tdur[:-4] + tdur[1:-3] + tdur[3:-1] + tdur[4:]
            nearflux     = nearcounts / neartdur[:,np.newaxis]
            nearexpect   = np.maximum(1, nearflux * tdur[2:-2, np.newaxis])
            nearsnr      = (counts[2:-2,:]-nearexpect) / np.sqrt(nearexpect)
            nearoutliers = np.nonzero(np.any(nearsnr > 7., axis=1))[0] + 2 # exclude 7-sigma outliers from background
            nearbigoutliers = np.nonzero(np.any(nearsnr > 500./np.sqrt(duration), axis=1))[0] + 2 # exclude 500-sigma excess from foreground
            inneridx     = np.nonzero((tcent >= -3*duration/2.) & (tcent <= 5*duration/2.))[0]
            goodidx = list(set(range(2,len(tcent)-2)) - set(np.hstack((nearbigoutliers, nearbigoutliers+1, nearbigoutliers-1))) \
                            - (set(np.hstack((nearoutliers, nearoutliers+1, nearoutliers-1))) - set(inneridx)))
            goodidx.sort()
            tcent = tcent[goodidx]
            tdur = tdur[goodidx]
            counts = counts[goodidx,:]
        bgidx = np.nonzero((tcent < -3*duration/2.) | (tcent > 5*duration/2.))[0] # background estimation indices
        fgidx    = np.nonzero(np.abs(tcent) <= duration/2.)[0]            # foreground indices
        if len(fgidx) == 0 or (len(fgidx)==1 and tdur[fgidx[0]] > 1.5*duration): # skip no foreground time, or not enough time resolution
            return (None, None) if not fitqual else (None, None, None, None, None, None)
        twin = [tcent[fgidx[0]]-tdur[fgidx[0]]/2., tcent[fgidx[-1]]+tdur[fgidx[-1]]/2.]
        ### fit each channel separately
        for j in range(nc):
            if rebin:
                binsize = max(0.256, duration / 4.)
                binedges = np.arange(twin[0]-(fitsize-0.5)*max(0.512, duration), tcent[-1], binsize)
                bcounts = np.histogram(tcent, bins=binedges, weights=counts[:,j])[0]
                btdur = np.histogram(tcent, bins=binedges, weights=tdur)[0]
                idx = np.nonzero(btdur > 0)[0]
                (bcounts, btdur) = (bcounts[idx], btdur[idx])
                bflux = bcounts / btdur
                btcent = np.histogram(tcent, bins=binedges, weights=tcent*tdur)[0][idx] / btdur
                bbgidx = np.nonzero((btcent < -duration) | (btcent > 2*duration))[0]
                bfgidx = np.nonzero(np.abs(btcent) <= duration/2.)[0]
                (ifit, tfit, cfit, dfit) = (bbgidx, btcent, bcounts, btdur)
            else:
                (ifit, tfit, cfit, dfit) = (bgidx, tcent, counts[:,j], tdur)
            if len(ifit) < degree+2: # require at least degree+1 points for background estimation
                return (None, None) if not fitqual else (None, None, None)
            # using numpy 1.7 polyfit, weights according to poisson error (1+sqrt(N)) accounting for duration since fit it on rate(t)
            # (par, cov) = fit.polyfit(tfit[ifit], cfit[ifit]/dfit[ifit], degree, cov=True, w=dfit[ifit]/(1.+np.sqrt(cfit[ifit])))
            # we DO NOT do the weights estimates above because it FAILS for low-N outliers (bad approximation)
            (par, cov) = fit.polyfit(tfit[ifit], cfit[ifit]/dfit[ifit], degree, cov=True)
            # if poiss: # poisson max-likelihood refinement to polyfit par
            #     import scipy.optimize
            #     par = scipy.optimize.fmin(fit.negloglikelihood, par, args=(np.polyval, tcent[bgidx], counts[bgidx,j], tdur[bgidx]), disp=False)
            #     cov = 0 # fmin is not going to give us parameter error estimates, so give up for now
            bgfit = dfit[ifit] * np.polyval(par, tfit[ifit])
            chisq = (cfit[ifit]-bgfit)**2/bgfit
            chisqdof = np.sum(chisq)/len(ifit)
            maxchisq = np.max(chisq)
            smallfit = False
            if(chisqdof > 2.0 or maxchisq > (4**2+2*np.log(len(ifit)))): # if fit is bad, try fitting a smaller interval
                smallfit = True
                jfit = ifit[np.abs(tfit[ifit]) <= (fitsize/2.)*max(0.512, duration)]
                if len(jfit) > degree+1: # require smaller interval to have degree+2 points
                    ifit = jfit
                    (par, cov) = fit.polyfit(tfit[ifit], cfit[ifit]/dfit[ifit], degree, cov=True)
                    # if poiss:
                    #     par = scipy.optimize.fmin(fit.negloglikelihood, par, args=(np.polyval, tcent[bgidx], counts[bgidx,j], tdur[bgidx]), disp=False)
                    #     cov = 0
                    bgfit = dfit[ifit] * np.polyval(par, tfit[ifit])
                    chisq = (cfit[ifit]-bgfit)**2/bgfit
                    chisqdof = np.sum(chisq)/len(ifit)
                    maxchisq = np.max(chisq)
            if(chisqdof > 3.0 or maxchisq > (5**2+2*np.log(len(ifit)))): # if fit is still bad give up mark bad background
                goodfit[j,i] = 0
            xsqdof[j,i] = chisqdof
            bg[j,i]  = np.sum(tdur[fgidx] * np.polyval(par, tcent[fgidx])) # expected background counts in fg region
            fg[j,i]  = np.sum(counts[fgidx,j])                             # accumulated fg counts
            lsqvars = tcent[fgidx]**np.arange(degree, -.1, -1)[:,np.newaxis] # e.g. [t**2, t, 0] dependent vars for the leastsq fit
            vbfit[j,i] = np.sum(tdur[fgidx])**2 * np.mean(lsqvars * np.dot(cov, lsqvars)) # variance contribution from parameter fit errors
            # note this is variance of the rate. to get variance of counts, we need to multiply by duration**2
            vbsys[j,i] = np.sum(tdur[fgidx]) * np.sum(cfit[ifit]) / np.sum(dfit[ifit]) * max(0., chisqdof-1.)
            # we try to estimate the contribution to variance of the bg estimate from systematic error in the fit model.
            # we don't really know this, so we use the average systematic error from the data instead.
            # we take average rate * duration in fg = representative counts in fg duration, and mulitply by typical excess stds**2.
            # for example if we have average expected counts of 100, with statistical variance of 100, but chisq is 4, then we expect
            # systematics to be contributing an excess variance of 300 on top of the statistical 100, for total variance of 400.
            if plot: # lots of code to make a nice set of plots, if desired
                width = min(plotsize, fitsize/2.) if smallfit else min(plotsize, fitsize)
                x = np.linspace(-width*duration, width*duration, 512)
                y = np.polyval(par, x) # background rate curve
                lsqvars = x[np.newaxis,:]**np.arange(degree, -.1, -1)[:,np.newaxis]
                var = np.sum(lsqvars * np.dot(cov, lsqvars), axis=0) # prediction error
                std = np.sqrt(var)
                flux = counts[:,j]/tdur
                err = np.sqrt(tdur * np.polyval(par, tcent))/tdur
                fgdur = np.sum(tdur[fgidx])
                fgt = np.mean(tcent[fgidx])
                snr = (fg[j,i]-bg[j,i]) / np.sqrt(bg[j,i])
                plt.subplot(nc, nd, 1+j*nd+i)
                if goodfit[j,i] == 0:
                    plt.gca().set_axis_bgcolor('0.8')
                if dqshade[j,i] != 0:
                    plt.gca().set_axis_bgcolor('0.6')
                plt.plot(x+duration/2., y, 'b-') # we plot x offset by duration/2 because all times are shifted to tstart=0 in plot
                plt.fill_between(x+duration/2., y+std, y-std, alpha='0.5', color='b') # 1 sigma
                if rebin:
                    berr = np.sqrt(btdur * np.polyval(par, btcent))/btdur
                    pidx = bbgidx[np.abs(btcent[bbgidx]) <= plotsize*duration]
                    plt.errorbar(btcent+duration/2., bflux, yerr=berr, fmt='.', color='gray') # again shifting to 0=tstart
                    plt.errorbar(btcent[pidx]+duration/2., bflux[pidx], yerr=berr[pidx], fmt='.', color='blue')
                    plt.errorbar(btcent[bfgidx]+duration/2., bflux[bfgidx], yerr=berr[bfgidx], fmt='.', color='green', zorder=4)
                else:
                    pidx = bgidx[np.abs(tcent[bgidx]) <= plotsize*duration]
                    plt.errorbar(tcent+duration/2., flux, yerr=err, fmt='.', color='gray')
                    if len(pidx) > 0:
                        plt.errorbar(tcent[pidx]+duration/2., flux[pidx], yerr=err[pidx], fmt='.', color='blue')
                    plt.errorbar(tcent[fgidx]+duration/2., flux[fgidx], yerr=err[fgidx], fmt='.', color='green', zorder=4)
                plt.errorbar(fgt+duration/2., fg[j,i]/fgdur, yerr=np.sqrt(bg[j,i])/fgdur, fmt='.', color='red', zorder=4)
                plt.xlim([(-plotsize+0.5)*duration, (plotsize+0.5)*duration])
                print (d, channels[j])
                # for some reason AnchoredText started needing an explicit prop dict(), toolkit bug
                at = AnchoredText("%s:%d" % (d, channels[j]), loc=2, pad=0.3, borderpad=0.3, prop=dict())
                at.patch.set_boxstyle("square,pad=0.")
                plt.gca().add_artist(at)
                # for some reason AnchoredText started needing an explicit prop dict(), toolkit bug
                at = AnchoredText("SNR = %.1f" % snr, loc=1, pad=0.3, borderpad=0.3, prop=dict())
                at.patch.set_boxstyle("square,pad=0.")
                plt.gca().add_artist(at)
                ymin[d[0] == 'n',j] = min(ymin[d[0] == 'n',j], plt.ylim()[0])
                ymax[d[0] == 'n',j] = max(ymax[d[0] == 'n',j], plt.ylim()[1])
                if i == 0: # first detector
                    plt.ylabel('flux [counts/s]')
                elif i == nd-1: # last detector
                    plt.gca().yaxis.set_label_position('right')
                    plt.gca().yaxis.set_ticks_position('right')
                    plt.gca().yaxis.set_ticks_position('both')
                    plt.ylabel('flux [counts/s]')
                else:
                    plt.gca().yaxis.set_ticklabels([])
                if j == nc-1: # last channel
                    plt.xlabel('relative time [s]')
                elif j == 0: # first channel
                    plt.gca().xaxis.set_ticks_position('top')
                    plt.gca().xaxis.set_ticks_position('both')
                else:
                    plt.gca().xaxis.set_ticklabels([])
    if plot:
        for (i, j) in itertools.product(range(nd), range(nc)):
            plt.subplot(nc, nd, 1+j*nd+i)
            plt.ylim(ymin[dlist[i][0] == 'n',j], ymax[dlist[i][0] == 'n',j])
            if j != 0: # not first channel
                tics = plt.gca().yaxis.get_ticklocs()
                if tics[-1] == plt.ylim()[-1]:
                    plt.gca().yaxis.set_ticks(tics[:-1])
        plt.suptitle('GBM detectors at ' + fermi2utc(t-duration/2.).strftime(fmtlist[4])[:-3] + ' +%0.3fs' % duration, va='center')
        # non-zero hspace and wspace to avoid matplotlib <1.0 bug for disappearing subplots
        plt.subplots_adjust(left=1./(2+nd*4), right=1-1./(2+nd*4), top=1-1./(4+nc*3), bottom=1./(4+nc*4), hspace=0.0015, wspace=0.0015)
        plt.setp(plt.gcf(), figheight=1+3*nc, figwidth=2+3*nd)
    # fg: fg counts (signal+bg), bg: bg estimate, goodfit: fit is good or not (bool)
    # vbfit: variance in bg estimate from statistical, vbsys: variance in bg estimate from systematic
    return (fg, bg) if not fitqual else (fg, bg, goodfit, vbfit, vbsys, xsqdof)

# calculate likelihood and chisq for measured foreground, estimated background, and response
# fg and bg should correspond to the final index/indices of response[spectrum, skylocation, .., channel, detector]
# it may be safer to pre-flatten all arrays to make sure they line up correctly
# returns (s, l1, l2) = (total likelihood, best estimate of amplitude, snr term, chisq term)
# beta=1.0 is scale free
def gbmlikelihood(fg, bg, response=None, nnewton=3, beta=1.0, gamma=2.5, sigref=0.05, vb=0, vr=0):
    import scipy.special
    if response == None: # no response used (only SNR term will be valid), maybe should change this to costh resp
        response = np.ones_like(fg)
    beta = np.abs(beta)
    # verify shape compatibility (avoid bugs)
    np.broadcast(response, fg)
    np.broadcast(response, bg)
    # calculate max log likelihood and estimate variance of likelihood distribution
    f    = fg.ravel()[np.newaxis, :] # flatten out everything into 2 axes: (variables, measurements)
    b    = bg.ravel()[np.newaxis, :] # same for bg mean
    b    = np.maximum(0, b) # do not allow unphysical negative background estimate
    d    = f - b            # excess above background counts
    r    = response.reshape((-1, fg.size)) # cast to 2D [spectrum*locations, detectors*channels]

    # fixed constants for faster derivative calculation
    rsq  = r**2           # square of response matrix
    if type(vb) is float or type(vb) is int: # vb in terms of fractional amount std (e.g. 10%)
        print "triggered fractional vb"
        vb = vb * b**2
    else:
        vb = vb.reshape(b.shape)
    if type(vr) is float or type(vr) is int:
        vr = vr * r**2
    else:
        vr = vr.reshape(r.shape)
    if type(sigref) == np.ndarray:
        sigref = sigref.ravel()
    # paper forumula uses (1+erf) for simplicity because reference ll does not matter yet
    lref = beta * np.log(gamma) + (1-beta) * np.log(sigref)

    # constant vf chi-sq solution initial guess for source amplitude
    vf = np.maximum(b, f) + vb # total variance of foreground counts
    vn = b + vb # total variance in noise counts (poisson + fit error)
    vfinv = 1./vf
    s = np.sum(r * d * vfinv, axis=-1) / np.sum(rsq * vfinv, axis=-1)
    spos = s > 0 # response uncertainty will only contribute for positive amplitude
    p = b + r * s[:,np.newaxis] # predicted foreground rate

    # newtonian extrapolation to find true max likelihood
    for i in range(nnewton):
        a = (f - p) * vfinv # alpha factor for simplification
        dvf = spos[:,np.newaxis] * (2*s[:,np.newaxis]*vr + r) # s derivative of vf
        asqmvfinv = a**2 - vfinv
        dl   = np.sum(r*a + 0.5 * dvf * asqmvfinv, axis=-1) # exact first and second derivatives dl/ds
        ddl  = np.sum(-vfinv * (dvf * a + r)**2 + vr * asqmvfinv + 0.5 * (dvf * vfinv)**2, axis=-1)
        s    = s - dl/ddl # new guess for source ampltiude (dL/da = 0)
        p    = b + r * s[:,np.newaxis] # second predicted foreground rate
        spos = s > 0 # only include source terms to variance for positive s
        vf    = np.maximum(b, p) + vb + (spos*s)[:,np.newaxis]**2 * vr # total foreground variance
        vfinv = 1. / vf # inverse to save on divides

    e = f - p
    # ll = np.sum(0.5 * (np.log(vn * vfinv) + d**2 / vn - e**2 * vfinv), axis=-1) # max likelihood
    ll = np.sum(0.5 * (np.log(vn * vfinv) + d**2 / vn - e**2 * vfinv), axis=-1) # max likelihood
    spos = s > (1e-6)**(1./beta) # positive indices where prior is valid, avoid numerical issues near 0
    # standard deviation of L distrib peak (for approx maginalization over s)
    sqvinv = np.sqrt(np.sum(rsq * vfinv, axis=-1))
    logsqv = -np.log(sqvinv)
    # log prior flattened at s < gamma * sigma
    logppos = np.log(1-np.exp(-(s*(1/gamma)*sqvinv)[spos]**beta)) - beta * np.log(s[spos])
    logp = -beta * (np.log(gamma) + logsqv) # log prior at s = 0, we need all logsqv indices for llmarg anyway
    np.place(logp, spos, logppos) # place logprior for positive indices
    logo = np.log(1+scipy.special.erf(s*sqvinv*(1/np.sqrt(2)))) # overlap between gaussian and s>0 region (error function)
    # technically logo should have extra term of -ln(2), but paper formula skips because lref can cancel it out
    llmarg = logsqv + logo + logp + ll # full marginalized likelihood
    # log likelihood = amplitude prior + approx marginalization over s + max log likelihood over s
    varsshape = response.shape[:response.ndim-fg.ndim] # return result in the same shape as input
    if(len(varsshape) > 1):
        return (s.reshape(varsshape), (llmarg-lref).reshape(varsshape))
    else:
        return (s, llmarg-lref)

# left matrices for coordinate transformations in xyz coords
# problems here can be due to old version of pyfits < 3 and keyword indexing
def sctransformations(poshist, t0):
    import scipy.interpolate
    scmet = poshist['SCLK_UTC']
    scquat = np.zeros((4, len(scmet)))
    scquat[0,:] = poshist['QSJ_4']
    scquat[1,:] = poshist['QSJ_1']
    scquat[2,:] = poshist['QSJ_2']
    scquat[3,:] = poshist['QSJ_3']
    q = normalize(scipy.interpolate.interp1d(scmet, scquat)(t0))
    sc2celestial = quat2rotationmatrix(q)
    celestial2sc = np.rollaxis(sc2celestial, 1) # transpose for 3x3, but works for 3x3xN
    return (sc2celestial, celestial2sc)

# get interpolated spacecraft position in celestial xyz coordinates (earth centered)
def scposition(poshist, t0):
    import scipy.interpolate
    scmet = poshist['SCLK_UTC']
    scxyz = np.zeros((3, len(scmet)))
    scxyz[0,:] = poshist['POS_X'] # (x, y, z) of SC in celestial coords (earth centered)
    scxyz[1,:] = poshist['POS_Y']
    scxyz[2,:] = poshist['POS_Z']
    return normalize(scipy.interpolate.interp1d(scmet, scxyz)(t0))

# earth position (relative to spacecraft) in celestial coordinates
def earthpositioncel(poshist, t0):
    return -scposition(poshist, t0)

# get earth position in xyz spacecraft coordinates
def earthposition(poshist, t0):
    (sc2celestial, celestial2sc) = sctransformations(poshist, t0)
    scxyz = scposition(poshist, t0)
    earthxyz = np.sum(celestial2sc * -scxyz, axis=1) # works for both single and array t0
    return earthxyz

# get approximate position for reflection point on earth atmosphere in SC coords, grbpt in (phi, theta) SC coords
def scatterpoint(poshist, t0, grbpt):
    relativealtitude = 550 / (6371.+125./2) # average fermi altitude from surface of earth using 125km of atmosphere
    earthxyz = earthposition(poshist, t0) # SC->earth in SC coords
    scxyz = -earthxyz                     # earth->SC in SC coords
    grbxyz = pt2xyz(grbpt)                # SC->GRB or earth->GRB in SC coords (GRB at infinity)
    x = np.linspace(0, .5, 256) # test 256 points along great circle
    testxyz = normalize(x * grbxyz[:,np.newaxis] + (1-x) * scxyz[:,np.newaxis]) # sample half of great circle between SC and GRB
    reflectxyz = normalize((1.+relativealtitude) * scxyz[:,np.newaxis] - testxyz) # from testxyz->SC
    ddot = np.sum((reflectxyz - grbxyz[:,np.newaxis]) * testxyz, axis=0) # dot product of test->SC minus dot product of test->GRB
    idx = np.argmin(np.abs(ddot)) # where angle between test->SC and test->GRB are the same
    return -reflectxyz[:,idx] # SC->testxyz at matching angle

# approximate sun position in celestial J2000 coordinates to ~0.01 degree
def sunposition(t0, j2000ref=utc2fermi(j2000ref), verbose=False):
    # approximate due to limited time precision, no planet perturbations, no TDB
    T  = (t0 - j2000ref) / (36525. * 86400.) # time since J2000 in units of 100 yr
    (TT, TTT) = (T**2, T**3)
    L0 = 280.46645   + 36000.76983 * T + 0.0003032 * TT # geom mean longitude
    M  = 357.52910   + 35999.05030 * T - 0.0001559 * TT - 4.8e-7 * TTT # mean anomaly
    Mrad = M * np.pi/180. # mean anomaly in radians
    e  = 0.016708617 - 0.000042037 * T - 1.236e-7  * TT # eccentricity of earth orbit
    C  = (1.914600 - 0.004817 * T - 0.000014 * TT) * np.sin(Mrad) \
       + (0.019993 - 0.000101 * T) * np.sin(2. * Mrad) \
       + 0.000290 * np.sin(3. * Mrad) # sun equation of center
    S = L0 + C # sun true longitude
    v = M + C  # sun true anomaly
    E = 23.4392911 - (46.8150 * T - 0.00059 * TT + 0.001813 * TTT) / 3600. # obliquity
    (Srad, Erad) = (S * np.pi/180., E * np.pi/180.)
    ra = np.arctan2(np.cos(Erad) * np.sin(Srad), np.cos(Srad)) % (2*np.pi) # sun RA in radians
    sindec = np.sin(Erad) * np.sin(Srad) # sun DEC in radians
    # sindec = np.maximum(sindec, -1.) # numerical precision issue
    # sindec = np.minimum(sindec,  1.) # should never really come up (E is small)
    dec = np.arcsin(sindec)
    if verbose:
        R = (1.000001018 * (1 - e**2)) / (1 + e * np.cos(v * np.pi/180.)) # radius to earth
        return {'geometric mean longitude':L0 % 360., 'mean anomaly':M % 360., 'eccentricity':e, \
                'sun equation of center':C, 'true longitude':S % 360., 'true anomaly':v % 360., \
                'distance':R, 'ecliptic obliquity':E, 'RA':ra * 180/np.pi, 'DEC':dec * 180/np.pi, \
                'centuries since J2000':T}
    else:
        return (ra, np.pi/2 - dec) # return (phi, theta) in spherical coords

# keep loudest unique events in array
# events = np.array([[tcent, duration, ..., likelihood], ...])
# overlapfactor: delete weaker event if stronger event explains overlapfactor of the SNR
def downselect(events, overlapfactor=0.2, threshold=None):
    if threshold:
        events = events[events[:,18] >= threshold]
    sortedevents = []
    uniqueevents = []
    if len(events) > 0: 
        sortedevents = events[(-events[:,18]).argsort(), :]
    for e1 in sortedevents:
        keep = True
        for e2 in uniqueevents:
            toverlap = min(e1[0] + e1[1]/2., e2[0] + e2[1]/2.) - max(e1[0] - e1[1]/2., e2[0] - e2[1]/2.)
            if toverlap > 0:
                amplitude = e1[11] / np.sqrt(e1[1])
                snrexpected = amplitude * toverlap / np.sqrt(e2[1])
                if e2[11] * overlapfactor < snrexpected:
                    keep = False
                    break
        if(keep):
            uniqueevents.append(e1)
    return np.array(uniqueevents)

# pretty print events >= threshold, optional downselection and sort by strength
# xref is subtracted from MET time in events before printed, e.g. to show GPS
# out can be an open filehandle, or STDOUT by default
def dump(events, sorted=True, downsel=False, threshold=None, xref=0, out=None):
    if out is None:
        import sys
        out = sys.stdout
    if type(events) is str:
        events = np.load(events)
    out.write("total number of win: %d\n" % len(events))
    out.write("in GTI (not SAA):    %d\n" % np.sum(events[:,2]))
    out.write("atmo resp available: %d\n" % np.sum(events[:,3]))
    out.write("able to analyze:     %d\n" % np.sum(events[:,4]))
    if len(events) == 0:
        return
    #                0          1      2    3    4    5     6     7   8    9    10    11    12    13   14     15   16    17      18      19     20    21    22
    out.write("------------------------------------------------------------------------------------------------------------------------------------------------\n")
    out.write("    tcent    duration  gti rock good  phi  theta (p,t)cele spec ampli  snr  snr0  snr1 chisq chisq+ sun  earth   logLR  coincLR  CR0   CR1   CR2\n")
    out.write("------------------------------------------------------------------------------------------------------------------------------------------------\n")
    if threshold:
        events = events[events[:,18] >= threshold]
    if downsel:
        events = downselect(events)
    elif sorted and (len(events) > 0): # no need to sort again if downselect is called
        events = events[(-events[:,18]).argsort(), :]
    for e in events:
        e = list(e)
        e[0] -= xref
        out.write("%13.3f %7.3f %3d %4d %4d  %5.3f %5.3f %5.3f %5.3f %1d %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.3f %5.3f %7.1f %7.1f %5.1f %5.1f %5.1f\n" % tuple(e))

# load in ASCII file dump of events into array
def loaddump(dumpfile):
    try:
        lines = [a.strip() for a in open(dumpfile)]
        if lines[-1][:len("saving output table:")] == "saving output table:": # strip last comment if it exists
            lines = lines[:-1]
        header = "    tcent    duration  gti rock good  phi  theta (p,t)cele spec ampli  snr  snr0  snr1 chisq chisq+ sun  earth   logLR  coincLR  CR0   CR1   CR2"
        hline = lines.index(header.strip())
        # events = np.genfromtxt(dumpfile, skiprows=hline+2, invalid_raise=False) # not going to work with ancient numpy on cluster
        events = np.array([map(float, a.split()) for a in lines[hline+2:]])
    except:
        events = []
    return events

# merge *in place* channels 3+4 for data from detectors in dlist, only keep SPECTRUM fields
# this is something of a hack to allow 50-300 keV atmo tables to be used with full-channel direct resp
def merge34(data, dlist=nlist):
    for d in dlist:
        counts = data[d]['SPECTRUM'].data['COUNTS']
        counts[:,3] += counts[:,4]
        counts[:,4:-1] = counts[:,5:]
        counts[:,-1] = 0

# return band function based on pars=[A, alpha, beta, epeak], at energies e: usually in terms of keV (relates A and epeak)
# e0 = 100 keV
def band(pars, e, e0=100):
    (A, a, b, ep) = pars
    ec = (a-b) * ep / (a+2)
    eb = (a-b) * ec
    i0 = e < eb
    i1 = e >= eb
    fe0 = A * (e / e0)**a * np.exp(-e / ec)
    fe1 = A * ((a-b)*ec/e0)**(a-b) * np.exp(b-a) * (e/e0)**b
    f = i0*fe0 + i1*fe1
    return f # number flux at e: dN / dE, in whatever units E is presented in

# number flux for 50-300 keV for band function (for normalization)
def band50300(pars, e0=100, res=0.25):
    (A, a, b, ep) = pars
    e = np.arange(50 + res/2., 300, res)
    f = band(pars, e, e0)
    n50300 = res * np.sum(f)
    return n50300

# crvars = (nsnr[0,j], nsnr[0,i], nsnr[1,j]
# cols = 20, 21, 22
def gbmcut(events, sky=2, cr1=5, cr2=1, cr2thr=8):
    if len(events) > 0:
        igti = events[:,2] > 0
        isky = events[:,19] - events[:,18] > sky
        icr1 = events[:,20]/np.maximum(0.1, events[:,21]) < cr1
        icr2 = (events[:,20]/np.maximum(0.1, events[:,22]) < cr2) | (events[:,20] < cr2thr)
        return igti & isky & icr1 & icr2
    else:
        return []

# return approximate occultation times for sources in list, returns dict
def occultationtimes(start, stop, slist=['Sun'] + strongsources, data=None):
    if data is None:
        data = precachedata(start, stop, dlist=[], poshist=True)
    poshist = data['poshist']['GLAST POS HIST'].data
    tnear = np.arange(start, stop+0.25, 0.25)
    # (s2cnear, c2snear) = sctransformations(poshist, tnear)
    earthnearxyz = earthpositioncel(poshist, tnear) # earth position in xyz celestial coordinates
    occdict['Sun'] = sunposition((start+stop)/2.) # add sun, does not change much over <1 day
    occtimes = dict()
    for x in slist:
        xnearxyz = pt2xyz(occdict[x])[:,np.newaxis]
        angles = np.arccos(np.sum(xnearxyz * earthnearxyz, axis=0)) * 180./np.pi
        zpidx  = np.where(np.diff(np.sign(angles - 69)))[0]
        zptime = (tnear[zpidx] + tnear[zpidx+1]) / 2.
        occtimes[x] = zptime
    return occtimes

# shows the angle of strong sources with respect to direction of Earth center as a function of time
# used to check for occultation steps in background
def sourceangles(data, met, dur=1.024, slist=['Sun'] + strongsources, showbg=False):
    import plots
    plots.importmpl()
    plt = plots.plt
    poshist = data['poshist']['GLAST POS HIST'].data
    tnear = np.linspace(met-10*dur, met+10*dur, 61)
    # (s2cnear, c2snear) = sctransformations(poshist, tnear)
    # earthnearxyz = earthposition(poshist, tnear) # earth position in xyz spacecraft coordinates
    earthnearxyz = earthpositioncel(poshist, tnear) # earth position in xyz celestial coordinates
    plt.fill_between(tnear - met, 0, 69, color='0.80')
    occdict['Sun'] = sunposition(met) # add sun
    for x in slist:
        xnearxyz = pt2xyz(occdict[x])[:,np.newaxis]
        angles = np.arccos(np.sum(xnearxyz * earthnearxyz, axis=0)) * 180./np.pi
        plt.plot(tnear - met, angles, label=x, linewidth=2.0)
    plt.ylim(60, 80.)
    if len(slist) > 5:
        plt.setp(plt.gcf(), figwidth=9, figheight=6)
        plt.gca().set_position([0.1, 0.1, 0.6, 0.8])
        # Put a legend to the right of the current axis
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), numpoints=1, prop={'size':10})
    else:
        plt.legend()
    if showbg:
        data = loaddailydata(fermi2utc(met))
        plt.twinx()
        for d in nlist:
            specdata = data[d]['SPECTRUM'].data  # look at data for detector
            tcent    = (specdata['TIME'] + specdata['ENDTIME']) / 2 - met           # relative mean time
            idx      = np.nonzero((specdata['QUALITY'] == 0) & (specdata['EXPOSURE'] > 0) & (np.abs(tcent) < 10*dur))[0]
            if len(idx) == 0:
                return (None, None) if not fitqual else (None, None, None, None, None, None)
            tcent    = tcent[idx]
            tdur     = specdata['EXPOSURE'][idx]
            counts   = specdata['COUNTS'][idx][:,0]
            # rebin historamming
            binedges = np.linspace(-10*dur, 10*dur, 31)
            bcounts  = np.histogram(tcent, bins=binedges, weights=counts)[0]
            btdur    = np.histogram(tcent, bins=binedges, weights=tdur)[0]
            btcent = np.histogram(tcent, bins=binedges, weights=tcent*tdur)[0] / btdur
            plt.plot(btcent, bcounts/btdur, color='gray')
            # from scipy.interpolate import UnivariateSpline
            # s = UnivariateSpline(btcent, bcounts/btdur, w=btdur/np.sqrt(bcounts))
            # plt.plot(tnear - met, s(tnear - met), color='gray')
    plt.xlim(-10 * dur, 10 * dur)
