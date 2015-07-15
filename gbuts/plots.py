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
