#! /usr/bin/env python

#
# LICENSE:
# Copyright (C) 2020  Neal Patwari
#
# Open source license, please see LICENSE.md
# 
# Author: Neal Patwari, neal.patwari@gmail.com
#
# Version History:
#
# Version 1.0:  Initial Release.  27 Oct 2014.
# Version 2.0:  For Powder MWW 2019.  5 Sep
# Version 2.1:  Using command line filename, 20 May 2020.
#

import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import sys

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 


# INPUTS: coord1, coord2
#    These are (latitude, longitude) coordinates in degrees
#    As given by google maps, for example
#
# OUTPUT: distance in meters
#
# Copied from Martin Thoma:
#    https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
#
def calcDistLatLong(coord1, coord2):

    R = 6373000.0 # approximate radius of earth in meters

    lat1 = math.radians(coord1[0])
    lon1 = math.radians(coord1[1])
    lat2 = math.radians(coord2[0])
    lon2 = math.radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a    = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return dist


# INPUTS: 
#     rxpower_array: numpy array of signal strength measurements, in dB (reference arbitrary)
#     dist_array: distance / path length corresponding to each Rx Power (in same distance units as d0)
#     d0: reference distance (in same units as dist_array)
#
# OUTPUTS:
#     n_p and Rx Power0: follow model that
#       rxpower_array[i] = Rx Power0 - 10.0 * n_p * log10(dist_array[i]/d0) + e
#    
def calcPLexponent(rxpower_array, dist_array, d0=1, printOption=True):

	# Model says Rx Power is linear in dB distance
    dist_dB = -10.0*np.log10(dist_array/d0)
    #print(dist_dB)
    #print(rxpower_array)
    coeffs  = np.polyfit(dist_dB, rxpower_array, 1)
    n_p     = coeffs[0]
    RxPower0    = coeffs[1]

    # Compute the standard deviation of error
    rxpower_est = n_p*dist_dB + RxPower0
    residuals = rxpower_array - rxpower_est
    e_std   = np.sqrt(np.mean(np.abs(residuals)**2))

    #Plot the results
    if printOption:
        plt.ion()
        plt.semilogx(dist_array, rxpower_array, 'ro', dist_array, rxpower_est, 'b-')
        plt.grid('on')
        #plt.xlim(100, 2000)
        plt.xlabel('Path Length (m)', fontsize=16)
        plt.ylabel('Received Power (dB, Unknown Ref)', fontsize=16)
        plt.show()

    return n_p, RxPower0, e_std

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

# load the data with NumPy function loadtxt
if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    filename = "test.csv"
print('Using CSV file ', filename)

# Known coordinates at transmitter & receiver locations
coords = {'meb': (40.768796, -111.845994),
'browning': (40.766326, -111.847727),
'ustar': (40.768810, -111.841739),
'bes': (40.761352, -111.846297),
'fm': (40.758060, -111.853314),
'honors': (40.763975, -111.836882),
'smt': (40.767959, -111.831685),
'dentistry': (40.758183, -111.831168),
'law73': (40.761714, -111.851914),
'bookstore': (40.764059, -111.847511),
'humanities': (40.763894, -111.844020),
'web': (40.767902, -111.845535),
'ebc': (40.767611, -111.838103),
'sage': (40.762914, -111.830655),
'madsen': (40.757968, -111.836334),
'moran': (40.769830, -111.837768)}


# Read the file
data = pandas.read_csv(filename, delimiter=",", index_col=0, header=0)

# Load rxpower and distance values into 1-D numpy arrays 
rxpower_array = np.array([])
dist_array = np.array([])
for rowlabel, content in data.items(): 
    # The transmitter label is rowlabel. Look up its coordinate.
    coord1 = coords[rowlabel]
    for collabel, rxpower in content.items():
        # Only read in the floating point numbers
        if (type(rxpower) is float) or ((type(rxpower) is str) and isfloat(rxpower)):
            rxpower_array = np.append(rxpower_array, float(rxpower))
            # The receiver label is column label.  Look up its coordinate.
            coord2 = coords[collabel]
            dist_array = np.append(dist_array, calcDistLatLong(coord1, coord2) )

# Calculate the linear fit between 10*log10(distance) and rxpower
n_p, RxPower0, e_std = calcPLexponent(rxpower_array, dist_array, 1, True)
print('n_p = ', n_p)
print('RxPower0 = ', RxPower0)
print('std of error = ', e_std)
plt.tight_layout()
plt.savefig('output.png')
input()
