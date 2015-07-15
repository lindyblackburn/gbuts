"""simple event and segment manipulation"""

__version__ = "$Id: event.py 112 2013-12-22 02:23:40Z lindy.blackburn@LIGO.ORG $"
__author__ = "Lindy Blackburn"
__email__ = "lindy.blackburn@ligo.org"

# intersect two segment lists, shorter one should go first
# from event.py
def andtwosegmentlists(list1, list2):
    newsegments = []
    index = 0
    for seg in list1:
        while(index > 0 and list2[index][0] > seg[0]):
            index -= 1
        while(index < len(list2) and list2[index][1] <= seg[0]):
            index += 1
        while(index < len(list2) and list2[index][0] < seg[1]):
            newsegments.append([max(seg[0], list2[index][0]), min(seg[1], list2[index][1])])
            index += 1
        if(index > 0):
            index -= 1
    return newsegments

# take the intersection of segments
#   - segmentlists: list of segment lists that need to be intersected
#     e.g. gpstime = segmentlists[ifo][seg_number][0(start) or 1(end)]
# from event.py
def andsegments(segmentlists, wrongsyntax = None):
    if(wrongsyntax != None):  # did not wrap two segment lists into a list
        return andsegments([segmentlists, wrongsyntax])
    if(len(segmentlists) > 0 and isinstance(segmentlists[0], (int, long, float))): # only one segment list
        return segmentlists
    if(len(segmentlists) == 1):  # only one segment list in list
        return segmentlists[0]
    elif(len(segmentlists) >= 2):
        lists = segmentlists[:]  # do not modify original list
        lists.sort(key=lambda x: len(x))  # loop over smallest lists first
        return reduce(lambda x, y: andtwosegmentlists(x, y), lists) # (((a&b)&c)&d)& ...

# take the union of segments
#   - segmentlists: list of segment lists that need to be merged.
#     e.g. gpstime = segmentlists[ifo][seg_number][0(start) or 1(end)]
# from event.py
def orsegments(segmentlists, wrongsyntax = None):
    if(wrongsyntax != None):  # did not wrap two segment lists into a list
        return orsegments([segmentlists, wrongsyntax])
    if(len(segmentlists) > 0 and isinstance(segmentlists[0], (int, long, float))): # only one segment list
        return segmentlists
    if(len(segmentlists) == 1):  # only one segment list in list
        return segmentlists[0]
    elif(len(segmentlists) >= 2):
        # fixsegments turns out to be faster for this operation than a custom O(n) routine
        # because of Python's C sort which does profiling and O(n) mergesort
        return fixsegments(reduce(lambda x, y: x + y, segmentlists)) # fixsegments(a + b + c + ...)

# sort and merge an out-of-order or overlapping segment list
# from event.py
def fixsegments(segments):
    newsegments = []
    if(segments == []):
        return newsegments
    segments.sort()
    nextseg = [a for a in segments[0]]
    for seg in segments:  # we want segments[1:] but this avoids a copy for large lists
        if(seg[0] <= nextseg[1]):
            nextseg[1] = max(seg[1], nextseg[1])
        else:
            newsegments.append(nextseg)
            nextseg = [a for a in seg]
    newsegments.append(nextseg)
    return newsegments

# cumulative livetime of segments
# from event.py
def livetime(segments):
    if(len(segments) == 0):
        return 0
    elif(isinstance(segments[0], (int, long, float))):
        segments = [segments]
    return sum(a[1] - a[0] for a in segments)

# return boolean list that says if each trigger in triggers is covered by segments
def includeidx(triggers, segments, tcol=0):
    ti = [(trg[tcol], i) for (i, trg) in enumerate(triggers)]
    ti.sort()
    (i, idx) = (0, [])
    for seg in segments:
        while(i < len(ti) and ti[i][0] < seg[0]):
            idx.append(False)
            i += 1
        while(i < len(ti) and ti[i][0] < seg[1]):
            idx.append(True)
            i += 1 
    while(i < len(ti)):
        idx.append(False)
        i += 1
    return idx

# random times from segment list
def randomtimes(ntimes, segments):
    import random
    randomtimes = []
    if(isinstance(segments[0], (int, long, float))):
        segments = [segments]
    lt = livetime(segments)
    for i in range(0, ntimes):
        bglt = lt * random.random()
        bgseg = 0
        while(bglt > segments[bgseg][1] - segments[bgseg][0]):
            bglt -= segments[bgseg][1] - segments[bgseg][0]
            bgseg += 1
        randomtimes.append(segments[bgseg][0] + bglt)
    randomtimes.sort()
    return randomtimes

# random (poisson) times corresponding to a particular rate
def randomrate(rate, segments):
    import random
    randomtimes = []
    if(isinstance(segments[0], (int, long, float))):
        segments = [segments]
    for seg in segments:
        t = seg[0]
        while(True):
            wait = random.expovariate(rate)
            t += wait
            if t < seg[-1]:
                randomtimes.append(t)
            else:
                break
    return randomtimes

# pretty print 1D or 2D array to string
def ap(a, fmt='%6.2f', stdio=True):
    import numpy as np
    if len(a.shape) == 1:
        a = a[np.newaxis,:]
    if stdio:
        import sys
        np.savetxt(sys.stdout, a, fmt)
    else:
        import StringIO
        buf = StringIO.StringIO()
        np.savetxt(buf, a, fmt)
        out = buf.getvalue()
        buf.close()
        return out

# load in ASCII table as float values, '.gz' supported
def load(file):
    f = gzopen(file, 'r')
    # table = [[float(field) for field in line.strip().split()] for line in f if line[0] != '#' and line.strip() != '' and line[-1] == '\n'] // necessary if there may be incomplete lines
    table = [[float(field) for field in line.strip().split()] for line in f if line[0] != '#' and line.strip() != '']
    f.close()
    return table

def loadsegments(file):
    f = gzopen(file, 'r')
    if(file[-4:] == ".xml" or file[-7:] == ".xml.gz"):
        from xml.dom import minidom
        xmldoc = minidom.parse(f)
        table = [map(float, line.split(',')[3:5]) for line in \
                 xmldoc.lastChild.childNodes[-2].childNodes[-2].firstChild.data.strip().split('\n')]
        return table
    else:
        table = [map(float, line.strip().split()) for line in f if line[0] != '#' and line.strip() != '']
        if(table == []):
            return table
        else:
            idx = table[0].index(max(table[0]))
            segments = [[line[idx-1], line[idx]] for line in table]
            return segments

# load in ASCII table as string values, '.gz' supported
def loadstringtable(file):
    f = gzopen(file, 'r')
    table = [line.strip().split() for line in f if line[0] != '#' and line.strip() != '']
    f.close()
    return table

# load in single string list
def loadlist(file):
    f = open(file, 'r')
    list = [line.strip() for line in f if line[0] != '#' and line.strip() != '']
    f.close()
    return list

# write ASCII table with optional formatstring (printf style)
#   - filename ending in '.gz' will create gzip file
def write(table, file, formatstring = None):
    f = gzopen(file, 'w')
    if formatstring == None:
        table = [" ".join([repr(field) for field in line]) for line in table]
    else:
        table = [formatstring % tuple(line) for line in table]
    for line in table:
        print >>f, line
    f.close()

# open file or gzip file transparently, replaces open(file, mode)
#   - 'file.gz' is automatically tried for read modes if 'file' does
#      not exist. 'file.gz' in write mode will write a gzip file
def gzopen(file, mode = 'r'):
    import os
    # explicitly defined gzip file
    if(file.endswith('.gz')):
        import gzip
        return gzip.GzipFile(file, mode)
    # if the file does not exist for read mode, try the gzip file
    # this will give a confusing error message for missing files
    elif(mode.startswith('r') and not os.path.exists(file)):
        import gzip
        return gzip.GzipFile(file + '.gz', mode)
    # this line should occur for 'r' modes where the file exists, or all 'w', 'a' modes
    return open(file, mode)

# attribute dictionary for data management
# but there is memory leak bug in python < 2.7.3
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs) # this will call and init the original dict()
        self.__dict__ = self # this will set the class internal attribute dictionary to itself
