"""time processing and conversion between time standards"""

import time
import datetime

__version__ = "$Id: clock.py 233 2015-04-10 18:24:42Z lindy.blackburn@LIGO.ORG $"
__author__ = "Lindy Blackburn"
__email__ = "lindy.blackburn@ligo.org"

ticref = time.time()

# begin stopwatch
def tic():
    global ticref
    ticref = time.time()

# end stopwatch
def toc(prefix=""):
    print prefix + repr(time.time() - ticref)

# wget http://maia.usno.navy.mil/ser7/tai-utc.dat
# leapdates = [datetime.datetime.strptime(line[:12], " %Y %b %d") for line in open(os.getenv("HOME") + '/bin/tai-utc.dat') if float(line[1:5]) >= 1970]
leapdates  = [datetime.datetime.strptime(a, "%m/%d/%y") for a in "7/1/81 7/1/82 7/1/83 7/1/85 1/1/88 1/1/90 1/1/91 7/1/92 7/1/93 7/1/94 1/1/96 7/1/97 1/1/99 1/1/06 1/1/09 7/1/12 7/1/15".split()]

# formatting for parsing time strings
fmtlist = """
%b %d %Y %H:%M:%S.%f %Z
%b %d %Y %H:%M:%S %Z
%b %d %Y %H:%M:%S.%f
%b %d %Y %H:%M:%S
%y%m%d %H:%M:%S.%f
%y%m%d %H:%M:%S
%Y%m%d %H:%M:%S
%Y%m%d
%y%m%d
%Y-%m-%d %H:%M:%S.%f
%Y-%m-%d %H:%M:%S
%m-%d-%Y %H:%M:%S.%f %Z
%m/%d/%Y %H:%M:%S.%f %Z
%m-%d-%Y %H:%M:%S %Z
%m/%d/%Y %H:%M:%S %Z
%m-%d-%Y %H:%M:%S.%f
%m/%d/%Y %H:%M:%S.%f
%m-%d-%Y %H:%M:%S
%m/%d/%Y %H:%M:%S
%m/%d/%Y
%Y%m%dT%H%M%S
%d-%b-%Y
%d-%b-%Y %H:%M:%S
%j-%Y
%j
""".strip().split('\n')

# convert string to time tuple by trying different format templates until one is found
def parsetime(timestring, fmt=None):
    if fmt is not None:
        return datetime.datetime.strptime(timestring, fmt)
    if timestring[:2] == "bn": # fermi trigger name
        return parsetime(timestring[2:8]) + datetime.timedelta(seconds=86.4 * float("0" + timestring[8:]))
    try: # speed up by trying last format that worked first
        return datetime.datetime.strptime(timestring, getattr(parsetime, 'lastfmt', fmtlist[0]))
    except:
        None
    for fmt in fmtlist:
        try:
            parsetime.lastfmt = fmt
            return datetime.datetime.strptime(timestring, fmt)
        except:
            None
    raise TypeError("no format for timestring '" + timestring + "' found in " + repr(fmtlist))

# some reference times
fermiref  = parsetime("Jan 01 2001 00:00:00 UTC")
gpsref    = parsetime("Jan 06 1980 00:00:00 UTC")
xteref    = parsetime("Feb 02 1996 00:00:00 UTC")
j2000ref  = parsetime("Jan 01 2000 11:58:55.816 UTC") # JD 2451545.0 (TT)

# convert datetime timedelta to total seconds (necessary since total_seconds() is not available in py2.6)
def td2sec(td):
    return td.seconds + td.days * 86400 + td.microseconds / 1e6

# number of leap seconds that have elapsed between start and stop (datetime objects)
# extra second occurs just before 00:00 UTC
def leapseconds(start, stop):
    return sum(1 for d in leapdates if d <= stop) - sum(1 for d in leapdates if d <= start)

# convert UTC datetime object to GPS (seconds since gpsref)
def utc2gps(timestruc):
    if type(timestruc) is not datetime.datetime:
        timestruc = parsetime(timestruc)
    return td2sec(timestruc - gpsref) + leapseconds(gpsref, timestruc)

# convert GPS seconds to UTC datetime object
def gps2utc(gps):
    leapdates_post = [timestruc for timestruc in leapdates if timestruc > gpsref]
    leapdates_gps = [td2sec(timestruc-gpsref)+1+i for (i, timestruc) in enumerate(leapdates_post)]
    # have two 00:00:00 times at leap second (same as xtime, not quite the same as the correct 23:59:60 just before)
    return gpsref + datetime.timedelta(seconds=gps - sum(map(lambda x: x <= gps, leapdates_gps)))

# convert UTC datetime object to fermi MET (seconds since fermiref)
def utc2fermi(timestruc):
    if type(timestruc) is not datetime.datetime:
        timestruc = parsetime(timestruc)
    return td2sec(timestruc - fermiref) + leapseconds(fermiref, timestruc)

# convert fermi MET to UTC datetime object
def fermi2utc(met):
    leapdates_post = [timestruc for timestruc in leapdates if timestruc > fermiref]
    leapdates_met = [td2sec(timestruc-fermiref)+1+i for (i, timestruc) in enumerate(leapdates_post)]
    # have two 00:00:00 times at leap second (same as xtime, not quite the same as the correct 23:59:60 just before)
    return fermiref + datetime.timedelta(seconds=met - sum(map(lambda x: x <= met, leapdates_met)))

# convert fermi MET to GPS seconds
def fermi2gps(met):
    return utc2gps(fermiref) + met

# convert GPS seconds to fermi MET
def gps2fermi(gps):
    return gps - utc2gps(fermiref)

# convert UTC datetime object to MJD
# assuming MJD(UTC) where MJD is aligned to UTC
# 50115. is MJD at xteref
def utc2mjd(timestruc):
    secondselapsed = td2sec(timestruc-xteref)
    return 50115. + secondselapsed/86400.

def gps2mjd(gps):
    return utc2mjd(gps2utc(gps))

# convert MJD to UTC using XTE reference time. assume MJD is aligned to UTC (for leap sec)
def mjd2utc(mjd):
    return xteref + datetime.timedelta(days=mjd-50155.)

# convert MJD to RXTE week
def mjd2week(mjd):
    return int(1. + (mjd - 50115.)/7.)

# convert RXTE week to MJD
def week2mjd(week):
    return (week - 1)*7. + 50115.

