"""plotting and data display"""

import numpy as np
from geometry import *

__version__ = "$Id: plots.py 222 2015-01-09 18:58:54Z lindy.blackburn@LIGO.ORG $"
__author__ = "Lindy Blackburn"
__email__ = "lindy.blackburn@ligo.org"

# put matplotlib and numpy in globals if not already
def importmpl():
    import os
    global matplotlib, plt, mlab
    if 'matplotlib' not in globals():
        import matplotlib
        # some reasonable ENV conditions to identify remote work (vs interactive)
        if os.environ.has_key('SSH_CONNECTION') or not os.environ.has_key('DISPLAY'):
            matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab

# extent=(left, right, bottom, top), npoints = number of points resolution in RA, skyhist[:,i] = (phi, theta, prob, cum)
# azimuth (theta=0) will ALWAYS be at top. if you want to flip y axis you must do: plt.setp(plt.gca(), ylim=plt.gca().get_ylim()[::-1]) or plt.gca().invert_yaxis(); plt.show()
# default extent is ra(h), dec(deg). also can put extent = 'radians', 'degrees', 'lonlat'
# skyhist has coords in it: skyhist[:,0] = phi, skyhist[:,1] = theta, skyhist[:,3] = probability
def ligoskymap(skyhist = 'ranked_sky_pixels.dat', coords='radec', extent=(0, 24, -90, 90), npoints=1024):
    importmpl()
    import scipy.special
    if type(skyhist) is str:
        from ligo import loadrankedskypixels
        skyhist = loadrankedskypixels(skyhist)
    if extent == 'radians':
        extent = (0, 2*np.pi, np.pi, 0)
    if extent == 'degrees':
        extent = (0, 360, 180, 0)
    if extent == 'lonlat':
        extent = (0, 360, -90, 90)
    if type(extent) is str:
        raise Exception("wrong argument for extent, must be: radians, degress, lonlat, or custom bounds (tuple)")
    # sky grid for plotting
    phi = np.linspace(0, 2*np.pi, npoints)
    theta = np.linspace(0, np.pi, int(npoints/2))
    (p, t) = np.meshgrid(phi, theta)
    # gridprob = mlab.griddata(skyhist[:,0], skyhist[:,1], skyhist[:,2], p, t)
    gridcum  = mlab.griddata(skyhist[:,0], skyhist[:,1], skyhist[:,3], p, t, interp='linear')
    # import scipy.interpolate
    # gridcum  = scipy.interpolate.griddata(skyhist[:,0:2], skyhist[:,3], np.meshgrid(phi, theta), fill_value=1., method='cubic') # does not really work any better
    # h1 = plt.contourf(gridcum.filled(1.0), scipy.special.erf(np.array([0,1])/np.sqrt(2)), origin='upper', extent=extent, colors='r', alpha=0.5)
    # h2 = plt.contourf(gridcum.filled(1.0), scipy.special.erf(np.array([1,2])/np.sqrt(2)), origin='upper', extent=extent, colors='r', alpha=0.25)
    # h3 = plt.contourf(gridcum.filled(1.0), scipy.special.erf(np.array([2,3])/np.sqrt(2)), origin='upper', extent=extent, colors='r', alpha=0.125)
    h1 = plt.contourf(gridcum.filled(1.0), scipy.special.erf(np.array([0,1])/np.sqrt(2)), origin='upper', extent=extent, colors=((1., .5, .5),(1., .5, .5)))
    h2 = plt.contourf(gridcum.filled(1.0), scipy.special.erf(np.array([1,2])/np.sqrt(2)), origin='upper', extent=extent, colors=((1., .75, .75),(1., .75, .75)))
    h3 = plt.contourf(gridcum.filled(1.0), scipy.special.erf(np.array([2,3])/np.sqrt(2)), origin='upper', extent=extent, colors=((1., .875, .875),(1., .875, .875)))
    if extent[2] > extent[3]: # flip y axis for standard spherical coordinates
        plt.setp(plt.gca(), ylim=plt.gca().get_ylim()[::-1])

# plot gbm skymap
def gbmskymap(loglikelihood, phi=None, theta=None, npoints=1024, filled=True, color='b', grbpt=None, showmax=False, earthpt=None, sunpt=None, occlimit=np.cos(67 * np.pi/180.)):
    importmpl()
    import scipy.special
    if phi == None:
        import nasaem.gbm
        (phi, theta) = nasaem.gbm.respgrid()
    ll = loglikelihood - np.max(loglikelihood)
    lprob = np.exp(ll)
    skynorm = np.sum(lprob)
    lprob = lprob / skynorm
    # this no longer works if not entire sky is calculated (e.g. skip earth facing positions)
    # lprob = lprob.reshape((-1, len(phi))).sum(axis=0).ravel() # marginalize over spectra if necessary
    if len(lprob.shape) == 2:
        lprob = lprob.sum(axis=0)
    idx = np.argsort(lprob)
    lphi = phi[idx[::-1]]
    ltheta = theta[idx[::-1]]
    lcum = np.cumsum(lprob[idx[::-1]])
    p = np.linspace(0, 2*np.pi, npoints)
    t = np.linspace(0, np.pi, int(npoints/2))
    (pp, tt) = np.meshgrid(p, t) # shape of pp, tt = (len(theta), len(phi))
    if earthpt is not None:
        earthxyz = pt2xyz(earthpt)
        eagrid = np.sum(pt2xyz((pp.ravel(), tt.ravel())).T * earthxyz, axis=1).reshape(pp.shape)
        plt.contourf(pp, tt, eagrid, [occlimit, 1.0], colors='k', alpha=0.25, label='Earth extent') 
        # plt.plot(earthpt[0], earthpt[1], marker=r'$\oplus$', ls='none', label='Earth', color='k')
        # plt.plot(earthpt[0], earthpt[1], marker='o', ls='none', color='k', markerfacecolor='none', markeredgewidth=1.5)
        plt.plot(earthpt[0], earthpt[1], marker='+', ls='none', label='Earth', color='k', markeredgewidth=1.5)
    if sunpt is not None:
        sunxyz = pt2xyz(sunpt)
        sunlimit = np.cos(0.5 * np.pi/180.)
        sungrid = np.sum(pt2xyz((pp.ravel(), tt.ravel())).T * sunxyz, axis=1).reshape(pp.shape)
        plt.contourf(pp, tt, sungrid, [sunlimit, 1.0], colors='y', alpha=0.5) 
        # plt.plot(sunpt[0], sunpt[1], marker=r'$\odot$', ls='none', label='Sun', color='k')
        plt.plot(sunpt[0], sunpt[1], marker='o', ls='none', label='Sun', color='k', markerfacecolor='y', markeredgewidth=1.5)
        # plt.plot(sunpt[0], sunpt[1], marker='.', ls='none', color='k', markersize=2)
    lgridcum = mlab.griddata(lphi, ltheta, lcum, pp, tt, interp='linear')
    if earthpt is not None: # hack because old version of matplotlib on cluster screws up interpolation
        lgridcum[eagrid > occlimit] = 1
    if filled:
        # h0 = plt.contourf(pp, tt, lgridcum.filled(1.0), scipy.special.erf(np.array([0,.5])/np.sqrt(2)), colors=color, alpha=0.75)
        h1 = plt.contourf(pp, tt, lgridcum.filled(1.0), scipy.special.erf(np.array([0,1])/np.sqrt(2)), colors=color, alpha=0.75)
        h2 = plt.contourf(pp, tt, lgridcum.filled(1.0), scipy.special.erf(np.array([1,2])/np.sqrt(2)), colors=color, alpha=0.5)
        h3 = plt.contourf(pp, tt, lgridcum.filled(1.0), scipy.special.erf(np.array([2,3])/np.sqrt(2)), colors=color, alpha=0.25)
    else:
        plt.contour(pp, tt, lgridcum.filled(1.0), scipy.special.erf(np.array([1,2,3])/np.sqrt(2)), colors=color, alpha=0.5)
    plt.xlim([0, 2*np.pi])
    plt.ylim([np.pi, 0])
    if grbpt is not None:
        plt.plot(grbpt[0], grbpt[1], 'yd')
    if showmax:
        plt.plot(lphi[0], ltheta[0], 'yx')
    # area = np.nonzero(idx[::-1] == mcrespidx(grbpt[0], grbpt[1]))[0]
    # print area
    # print np.arccos(np.sum(pt2xyz((lphi[0], ltheta[0]))*pt2xyz(grbpt))) * 180/np.pi

# make contourf skymap with basemap, (phi, theta) define locations of loglikelihood sample points
def gbmbasemap(loglikelihood, grbpt=None, phi=None, theta=None, plotconf=0.95, crop=1.1, color='b', mapproj='laea', plotmax=True, legend=True, amax=None, pt0=None, lat2str='%g', lon2str='%g'):
    importmpl()
    import gbm
    import scipy.special
    from mpl_toolkits.basemap import Basemap
    if phi is None:
        (phi, theta) = gbm.respgrid()
    steps = np.array([1, 2, 3, 5, 10, 15, 30, 60, 120]) # graticule spacing options
    (ntheta, nphi) = (5.5, 5.5) # attempt number of graticules in theta, phi
    # get values at grb and at max likelihood
    if grbpt is not None:
        grbidx = gbm.mcrespidx(grbpt)
        llgrb = loglikelihood[grbidx]
    if len(loglikelihood.shape) == 2:
        loglikelihood = loglikelihood.sum(axis=0)
    maxidx = np.argmax(loglikelihood)
    llmax = loglikelihood[maxidx]
    # calculate probability map
    ll = loglikelihood - llmax
    lprob = np.exp(ll)
    skynorm = np.sum(lprob)
    lprob = lprob / skynorm
    # order idx in order to form cumulative prob map
    idx = np.argsort(lprob)[::-1]
    lphi = phi[idx]
    ltheta = theta[idx]
    lcum = np.cumsum(lprob[idx])
    # put in xyz coords and get central value
    xyz = pt2xyz(phi, theta)
    if pt0 is None:
        xyz0 = np.average(xyz[:,idx[lcum < plotconf]], axis=1)
        (phi0, theta0) = xyz2pt(xyz0)
    else:
        xyz0 = pt2xyz(pt0)
        (phi0, theta0) = pt0
    # convert to lon and lat for basemap plotting
    lat0 = 90. - theta0 * 180./np.pi
    lon0 = phi0 * 180/np.pi
    print lon0, lat0
    # width and height of the map in radians, and degrees
    if amax is None:
        amax = np.arccos(np.min(np.sum(xyz0[:,np.newaxis] * xyz[:,idx[lcum < plotconf]], axis=0)))
    if grbpt is not None:
        grbxyz = pt2xyz(grbpt)
        amax = max(amax, np.arccos(np.sum(xyz0 * grbxyz)) * 1.05 / crop)
    wdeg = amax * 180/np.pi
    # initalize basemap with chosen parameters
    width = crop * 2*amax # physical great circle width in each direction, with crop applied
    bm = Basemap(width=width,height=width,projection=mapproj, lat_0=lat0,lon_0=lon0,rsphere=1.)
    plt.clf()
    # reverse East-West for sky picture
    plt.gca().invert_xaxis()
    # steps for graticules (in deg)
    pastep = steps[np.argmin(np.abs(wdeg / steps - ntheta))]
    mestep = steps[np.argmin(np.abs((bm(width,0,inverse=True)[0]%360.-bm(0,0,inverse=True)[0]%360.) / steps - nphi))]
    # plot graticules
    # g1 = bm.drawparallels(np.arange(-90,91,pastep), labels=[True,False,False,False], latmax=90-pastep, dashes=[1,1], color='0.0', zorder=10)
    # g2 = bm.drawmeridians(np.arange(-180,180,mestep), labels=[False,False,False,True], latmax=90-pastep, dashes=[1,1], color='0.0', zorder=10)
    # for q in g1.values() + g2.values():
    #     plt.setp(q[0][0], alpha=0.2)
    g1 = bm.drawparallels(np.arange(-90,91,pastep), labels=[True,False,False,False], latmax=90-pastep, dashes=[1,1], color='0.8', zorder=0, fmt=lat2str)
    g2 = bm.drawmeridians(np.arange(-180,180,mestep), labels=[False,False,False,True], latmax=90-pastep, dashes=[1,1], color='0.8', zorder=0, fmt=lon2str)
    # prepare grid to calculate confidence regions
    x = np.linspace(0, width, 512)
    y = np.linspace(0, width, 512)
    xy = np.meshgrid(x, y)
    # convert (theta, phi) -> (lon, lat) -> (x, y) image coords
    (xp, yp) = bm(lphi * 180/np.pi, 90. - ltheta*180/np.pi)
    z = mlab.griddata(xp, yp, lcum, x, y, interp='linear')
    import scipy.special
    # plot map countour
    # h1 = plt.contourf(x, y, z, scipy.special.erf(np.array([0,1])/np.sqrt(2)), colors=((1.,.5,.5),(1.,.5,.5)))
    # h2 = plt.contourf(x, y, z, scipy.special.erf(np.array([1,2])/np.sqrt(2)), colors=((1.,.75,.75),(1.,.75,.75)))
    # h3 = plt.contourf(x, y, z, scipy.special.erf(np.array([2,3])/np.sqrt(2)), colors=((1.,.875,.875),(1.,.875,.875)))
    h1 = plt.contourf(x, y, z, scipy.special.erf(np.array([0,1])/np.sqrt(2)), colors=color, alpha=0.5, zorder=10)
    h2 = plt.contourf(x, y, z, scipy.special.erf(np.array([1,2])/np.sqrt(2)), colors=color, alpha=0.25, zorder=10)
    h3 = plt.contourf(x, y, z, scipy.special.erf(np.array([2,3])/np.sqrt(2)), colors=color, alpha=0.15, zorder=10)
    # h4 = plt.contourf(x, y, z, scipy.special.erf(np.array([3,4])/np.sqrt(2)), colors=color, alpha=0.07, zorder=10)
    if plotmax:
        (maxx, maxy) = bm(phi[maxidx] * 180/np.pi, 90. - theta[maxidx] * 180/np.pi)
        plt.plot(maxx, maxy, 'yx', markersize=10, markeredgewidth=3, label='max likelihood = %d' % llmax, zorder=11)
    if grbpt is not None:
        (grbx, grby) = bm(grbpt[0] * 180/np.pi, 90. - grbpt[1]*180/np.pi)
        plt.plot(grbx, grby, 'yd', markersize=10, markeredgewidth=1.5, label='true position = %d' % llgrb, zorder=11)
        if plotmax:
            dtheta = np.arccos(np.dot(pt2xyz(grbpt), pt2xyz((phi[maxidx], theta[maxidx])))) * 180/np.pi
            plt.plot([maxx, grbx], [maxy, grby], 'y-', label=r'$\Delta\theta$ = %.1f deg' % dtheta, zorder=9)
    if legend:
        plt.legend(loc='best', numpoints=1)

# generic GBM sky grid plot
def gridplot(z, npoints=1024, phi=None, theta=None, cbar=True, earthpt=None, vmin=None, vmax=None, aspect='auto', rasterized=False):
    importmpl()
    if phi == None:
        import gbm
        (phi, theta) = gbm.respgrid()
    p = np.linspace(0, 2*np.pi, npoints)
    t = np.linspace(0, np.pi, int(npoints/2))
    idx = np.nonzero(phi == 0)
    pwrap = phi[idx] + 2 * np.pi
    twrap = theta[idx]
    zwrap = z[idx]
    (pp, tt) = np.meshgrid(p, t)
    zgrid = mlab.griddata(np.hstack((phi, pwrap)), np.hstack((theta, twrap)), np.hstack((z, zwrap)), pp, tt, interp='linear')
    if earthpt != None:
        occlimit = np.cos(67 * np.pi/180.)
        eagrid = np.sum(pt2xyz((pp.ravel(), tt.ravel())).T * pt2xyz(earthpt), axis=1).reshape(pp.shape)
        zgrid[eagrid > occlimit] = 0
    h = plt.imshow(zgrid, extent=(0, 2*np.pi, np.pi, 0), vmin=vmin, vmax=vmax, aspect=aspect, rasterized=rasterized)
    if cbar:
        plt.colorbar(orientation='horizontal', aspect=30, shrink=1.0)
    if earthpt != None:
        plt.contourf(pp, tt, eagrid, [occlimit, 1.0], colors='0.75', label='Earth extent')
        plt.plot(earthpt[0], earthpt[1], marker=r'$\oplus$', ls='none', label='Earth')
    # plt.subplots_adjust(top=0.875)
    return (h, plt.gca())

# put nai detector labels on skymap plot
def labelnai(color='gray', zorder=1, twins=True, **kwargs):
    importmpl()
    import gbm
    for (i, (detp, dett)) in enumerate(gbm.naipt.T):
        plt.text(detp, dett, gbm.nlist[i], horizontalalignment='center', verticalalignment='center', color=color, zorder=zorder, *kwargs)
    plt.xlabel(r'spacecraft $\phi$ [rad]')
    plt.ylabel(r'spacecraft $\theta$ [rad]')
    # plt.subplots_adjust(top=0.875)
    plt.setp(plt.gca(), xlim=[0, 2*np.pi], ylim=[np.pi, 0])
    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()
    if twins:
        ax2 = plt.twinx()
        plt.setp(ax2, ylim=[180, 0])
        ay2 = plt.twiny()
        plt.setp(ay2, xlim=[0, 360])

# put a trailing arrow on the end of a plot with given coordinate size and arrow pointiness
# x and y may be the complete trace, but only the last two points will determine the arrow
def trailingarrow(x, y, size=2.0, sharpness=2.0, **kwargs):
    importmpl()
    (dx, dy) = normalize(np.array((x[-1]-x[-2], y[-1]-y[-2])))
    (left, right, back) = np.array(((-dy,dx), (dy,-dx), (-dx,-dy)))
    (ax, ay) =  (size/np.sqrt(1+sharpness**2))*np.array(((left+sharpness*back, (0,0), right+sharpness*back))).T
    plt.plot(x[-1]+ax, y[-1]+ay, **kwargs)

# scatter plot of events
# events: array of events (see nasaem.dump for columns)
# vbars: plot dotted vertical lines at these x locations
# specnames: names for spectra indexed as 0, 1, 2, ..
# threshold: only show events with LR > threshold
def eventscatter(events, xlim=None, vbars=None, specnames=['hard', 'normal', 'soft'], threshold=None, xref=0):
    importmpl()
    if threshold is not None:
        events = events[events[:,-1] >= threshold]
    plt.clf()
    (tcent, duration, spec, snr, lr) = events[:,[0, 1, 9, 11, -1]].T
    ylim = [np.min(spec)-1, np.max(spec)+1]
    for i in np.argsort(lr):
        plt.errorbar(tcent[i]-xref, spec[i], xerr=duration[i]/2., marker='d', markersize=snr[i], color=matplotlib.cm.jet((snr[i]-1.0)/10.))
    plt.yticks(range(len(specnames)), specnames)
    if vbars is not None:
        for v in vbars:
            plt.axvline(v-xref, color='gray', linestyle=':')
    if xlim is not None:
        plt.xlim(xlim[0]-xref, xlim[1]-xref)
    plt.ylim(ylim[-1], ylim[0])

def eventsnrvst(events, xlim=None, vbars=None, threshold=None, xref=0, col=11):
    importmpl()
    if threshold is not None:
        events = events[events[:,-1] >= threshold]
    plt.clf()
    (tcent, duration, spec, snr, lr) = events[:,[0, 1, 9, col, -1]].T
    h = spec == 0
    n = spec == 1
    s = spec == 2
    plt.plot(tcent[s] - xref, snr[s], 'r.')
    plt.plot(tcent[n] - xref, snr[n], 'g.')
    plt.plot(tcent[h] - xref, snr[h], 'b.')
    if vbars is not None:
        for v in vbars:
            plt.axvline(v-xref, color='gray', linestyle=':')
    if xlim is not None:
        plt.xlim(xlim[0]-xref, xlim[1]-xref)

def eventxy(events, xcol=0, ycol=1, threshold=None, **kwargs):
    importmpl()
    if threshold is not None:
        events = events[events[:,-1] >= threshold]
    # plt.clf()
    h = events[:,9] == 0
    n = events[:,9] == 1
    s = events[:,9] == 2
    plt.plot(events[s,xcol], events[s,ycol], 'r.', **kwargs)
    plt.plot(events[n,xcol], events[n,ycol], 'g.', **kwargs)
    plt.plot(events[h,xcol], events[h,ycol], 'b.', **kwargs)

# make a plot of ASM data at t0+4d
# if data is an integer, it is interpreted as galaxy ID, and loaded automatically
def asmplot(data, t0, inject=None, fitpar=None, channel='b+c', range=[0, 4], colors='br'):
    import asm, fit
    from clock import gps2mjd
    importmpl()
    if type(data) is int:
        data = asm.loadasmdata(data, t0+range[0]*86400, t0+range[1]*86400)
    if type(data) is dict:
        data = data[channel]
    day = data[:,0] - gps2mjd(t0)
    i = (day > range[0]) & (day < range[1])
    tt = np.logspace(np.log10(100), np.log10(4*86400), 512)
    if(np.sum(i) == 0):
        return None # no data to fit
    day = day[i]
    counts = data[i, 7]
    err = data[i,8]
    var = err**2
    sec = day*86400. + 30 # avoid blow up near zero
    plt.errorbar(sec, counts, yerr=err, color=colors[0], fmt='.', label='data')
    ylim = plt.ylim()
    if inject is not None:
        # asminject(t0, tinj, par, amplitude=1e46, distance=20, normed=False, channel='sum')
        injflux = asm.asminject(0.0, sec, *inject, channel=channel)
        plt.errorbar(sec, counts+injflux, yerr=err, color=colors[1], fmt='.', label='data+injection')
        ylim = plt.ylim()
        injcurve = asm.asminject(0.0, tt, *inject, channel=channel)
        plt.plot(tt, injcurve, '--', color=colors[1], label='injection lightcurve', lw=1.5, alpha=0.5)
    if fitpar is not None:
        fitflux = asm.asminject(0.0, sec, fitpar, amplitude=1., distance=None, channel=channel)
        fitamp = fit.amplitudefit(counts, err, fitflux)
        fitcurve = asm.asminject(0.0, tt, fitpar, amplitude=fitamp, channel=channel)
        plt.plot(tt, fitcurve, label='best fit')
    plt.title("ASM data " + channel)
    plt.setp(plt.gca(), xscale='log')
    plt.xlim(min(10., sec[0]), range[1]*86400)
    plt.ylim(ylim)
    plt.grid(True, alpha=0.5)

# set y0 to something between 0 and 1 so that ylog plots work, w=weight
def stepcdf(ranks, x0=0, y0=0.1, w=1.0, **kwargs):
    importmpl()
    if type(w) is int or type(w) is float:
        w = w * np.ones_like(ranks)
    sidx = np.argsort(ranks)
    srank = ranks[sidx] # ranks from small to big
    wtot = np.sum(w)
    count = np.empty(len(w) + 1)
    count[0] = wtot
    count[1:] = wtot - np.cumsum(w[sidx]) # counts # of events between rank and previous rank
    count[-1] += w[sidx[-1]]*y0 # some slob so that logscale plots do not get eaten instead of zero
    ypt = np.vstack((count, count)).T.ravel()[:-1]
    xpt = np.hstack((x0, np.vstack((srank, srank)).T.ravel()))
    plt.plot(xpt, ypt, **kwargs)

def xlim0(x0):
    importmpl()
    xlim = plt.xlim()
    plt.xlim(x0, xlim[1])

def ylim0(y0):
    importmpl()
    ylim = plt.ylim()
    plt.ylim(y0, ylim[1])
