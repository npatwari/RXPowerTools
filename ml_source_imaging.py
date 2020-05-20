#! /usr/bin/env python

#
# Script name: ml_source_imaging
# Copyright 2020 Neal Patwari
#
# Version History:
#   Version 1.0:  Initial Release.  27 May 2014.
#   Version 2.0:  For Powder MWW 2019.  5 Sep
#   Version 2.1:  Edits, updated license, REU: 19 May 2020
#
# License: see LICENSE.md

import sys
import pandas
import numpy as np
#import scipy.stats as stats
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib
import math

plt.ion()
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

# All possible receiver/sensor coordinates
coords = {'meb': (40.768796, -111.845994),
  'browning': (40.766326, -111.847727),
  'basic': (40.766326, -111.847727), # we used browning as "basic" during MWW2019
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


# Calc the distance between lists of lat/long coordinates
# Assumes coordinates are in rows of list1 and list2
def calcDistLatLongVectors(list1, list2):

    len1 = list1.shape[0]  # How many coordinates in each list
    len2 = list2.shape[0]
    Dmat = np.zeros((len1,len2))
    for row, coord1 in enumerate(list1):
        for col, coord2 in enumerate(list2):
            Dmat[row, col] = calcDistLatLong(coord1, coord2)
    return Dmat


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



# Convert Tx, Rx, and Ch numbers to link number
def linkNumForTxRxChLists(tx, rx, ch, nodeList, channelList):
    if (nodeList.count(tx) == 0) or (nodeList.count(rx) == 0) or (channelList.count(ch) == 0):
        print("Error in linkNumForTxRx: tx, rx, or ch number invalid")
    rx_enum = nodeList.index(rx)
    tx_enum = nodeList.index(tx)
    ch_enum = channelList.index(ch)
    nodes = len(nodeList)
    links = nodes*(nodes-1)
    linknum = ch_enum*links + tx_enum*(nodes-1) + rx_enum
    if (rx_enum > tx_enum):
        linknum -= 1
    return linknum

# Convert link number to Tx and Rx numbers
def txRxForLinkNum(linknum, nodes):
    tx    = linknum / (nodes-1)
    rx    = linknum % (nodes-1)
    if (rx >= tx): 
        rx+=1
    if (tx >= nodes):
        print("Error in txRxForLinkNum: linknum too high for nodes value")
    return (tx, rx)

# Convert link number to Tx, Rx, and channel numbers
def txRxChForLinkNum(linknum, nodeList, channelList):
    nodes   = len(nodeList)
    links   = nodes*(nodes-1)
    ch_enum = linknum / links
    remLN   = linknum % links
    tx_enum = remLN / (nodes-1)
    rx_enum = remLN % (nodes-1)
    if (rx_enum >= tx_enum):
        rx_enum+=1
    if (tx_enum >= nodes) | (ch_enum >= len(channelList)):
        print("Error in txRxChForLinkNum: linknum or ch too high for nodes, channels values")
    else:
        ch = channelList[ch_enum]
        tx = nodeList[tx_enum]
        rx = nodeList[rx_enum]
    return (tx, rx, ch)

def calcGridPixelCoords(personLL, personUR, delta_p):

    xVals  = np.arange(personLL[0], personUR[0], delta_p)
    yVals  = np.arange(personLL[1], personUR[1], delta_p)
    cols   = len(xVals)
    pixels = cols * len(yVals)  # len(yVals) is the number of rows of pixels

    # fill the first row, then the 2nd row, etc.
    pixelCoords = np.array([[xVals[i%cols], yVals[i//cols]] for i in range(pixels)])

    return pixelCoords, xVals, yVals

# Plot the node/sensor locations with their node numbers
def plotLocsDictionary(coords, coordsLL, coordsUR, fignum):

    
    #Increase the axes to show full map.
    deltax          = coordsUR[1]-coordsLL[1]
    epsx            = deltax*0.1
    deltay          = coordsUR[0]-coordsLL[0]
    epsy            = deltay*0.1
    plt.figure(fignum)
    for key, value in coords.items(): 
        plt.plot(value[1]-coordsLL[1], value[0]-coordsLL[0], 'k.', markersize=14.0)
        plt.text(value[1]-coordsLL[1], value[0]-coordsLL[0]+ epsy*0.2, key[0:3], 
             horizontalalignment='center', verticalalignment='bottom', fontsize=16)
    plt.axis((-epsx, coordsUR[1]-coordsLL[1]+epsx, 
        -epsy, coordsUR[0]-coordsLL[0]+epsy))
    plt.xlabel('Relative Longitude (degrees)', fontsize=18)
    plt.ylabel('Relative Latitude (degrees)', fontsize=18)
    plt.grid()



# Plot the node/sensor locations with their node numbers
def plotLocs(sensorCoords):

    plt.plot(sensorCoords[:,0], sensorCoords[:,1], 'k.', markersize=14.0)

    #Increase the axes to show full map.
    xmin, xmax, ymin, ymax = plt.axis()
    deltay          = ymax-ymin
    epsy            = deltay*0.1
    deltax          = xmax-xmin
    epsx            = deltax*0.1
    plt.axis((xmin-epsx, xmax+epsx, ymin-epsy, ymax+epsy))
    for number, coord in enumerate(sensorCoords):
        plt.text(coord[0], coord[1]+ epsy*0.2, str(number), 
             horizontalalignment='center', verticalalignment='bottom', fontsize=16)
    plt.xlabel('X Coordinate (m)', fontsize=18)
    plt.ylabel('Y Coordinate (m)', fontsize=18)
    plt.grid()

# PURPOSE: 
#     Calculate the generalized likelihood at each pixel.
#     Assume an unknown transmit power (and thus estimate it)
#     Use a grid of lat/long coordinates 
#
# INPUTS: 
#   sensorCoords: sensors x 2 coordinate vector
#   sensorMeasdB: sensors length vector of rxpower values
#   delta_p:      pixel size in lat/long degrees
#   d0, P0, n_p, sigmadB: Channel model parameters
#   minPL:        minimum distance (meters) channel model should be used
#
# OUTPUTS:
#   imageMat:   Calculated likelihood (for each pixel)
#   estTXPower: Estimated TX power matrix (for each pixel)
#   xVals:      latitude coordinates of rows of image
#   yVals:      longitude coordinates of columns of image
#
def calcMLGrid(sensorCoords, sensorMeasdB, delta_p, d0, P0, n_p, sigmadB, minPL):

    # 1. Set up pixel locations as a grid.
    personLL      = sensorCoords.min(axis=0)
    personUR      = sensorCoords.max(axis=0)
    pixelCoords, xVals, yVals = calcGridPixelCoords(personLL, personUR, delta_p)
    pixels        = pixelCoords.shape[0]
    sensors       = sensorCoords.shape[0]
    rxpower_var       = sigmadB**2
    const         = 1./np.sqrt(2*math.pi*rxpower_var)
    
    # 2. Find distances between pixels and sensors in meters
    DistPixelAndNode = calcDistLatLongVectors(pixelCoords, sensorCoords)

    # For each pixel, calculate what the RX power would be for a 
    # source at pixel, rx at node, given the PL exponent model
    modelrxpower      = [ [] for i in range(pixels) ]
    estTXPower    = np.zeros(pixels) 
    imageMat      = np.zeros(pixels)
    for i in range(pixels):
        # Channel model rxpower given TX power is same as in model
        modelrxpower_TX0  = P0 - 10*np.log10(np.maximum(DistPixelAndNode[i,:], minPL))
        # Estimate of TX power compared to channel model (LS fit = average)
        estTXPower[i] = np.average(sensorMeasdB - modelrxpower_TX0)
        modelrxpower[i]   = modelrxpower_TX0 + estTXPower[i]
        # Probability of measured rxpower values at sensor locations if
        #  - The TX power is the estimated TX power
        #  - The channel model gives the average RX power
        #  - The rxpower measurements are iid Gaussian
        sqrDiff       = sum((sensorMeasdB - modelrxpower[i])**2)
        imageMat[i]   = const*np.exp(sqrDiff/(-2*rxpower_var))

    # Make the vector a matrix.
    estTXPower.resize(len(yVals), len(xVals))
    imageMat.resize(len(yVals), len(xVals))

    return imageMat, estTXPower, xVals, yVals



# Return the coordinate of the maximum pixel in the image
def imageMaxCoord(imageMat, xVals, yVals):    
    rowMaxInd, colMaxInd = np.unravel_index(imageMat.argmax(), imageMat.shape)
    #print("(", colMaxInd,",",rowMaxInd,")")
    #print(imageMat[rowMaxInd-2:rowMaxInd+3, colMaxInd-2:colMaxInd+3])
    return (xVals[colMaxInd], yVals[rowMaxInd])

# Plot an image matrix when the coordinates are lat/long (GPS)
# and the coordinates are listed in a Dictionary
def plotImageMat(imageMat, sensorCoordsDict, coordsLL, coordsUR, delta_p, title, fignum):
    plt.figure(fignum)
    plt.cla()
    print(imageMat.shape)


    # Plot the image on top of the map
    plt.imshow(imageMat.T, interpolation='none', origin='lower', aspect=1.42)
    #plt.imshow(imageMat.T, origin='lower')
    plt.ylabel('Index', fontsize=16)
    plt.xlabel('Index', fontsize=16)
    plt.title(title, fontsize=16)
    plt.show()


# Calculate the bounding box (rectangle) that contains all of 
#     the coords that are input
# OUTPUTS:
#     coordsLL: lower left coordinate of the rectangle
#     coordsUR: upper right coordinate of the rectangle
def calcBoundingBoxCoordsDict(coords):
    
    coordsLL = np.array(coords[list(coords)[0]] )
    coordsUR = coordsLL
    for key, value in coords.items(): 
        coordsLL = np.minimum(coordsLL, np.array(value))
        coordsUR = np.maximum(coordsUR, np.array(value))
    return coordsLL, coordsUR


# Overall main function to be called from this file.  
#
# INPUT:
#    filename: a CSV file with name, rxpower pairs in each row (comma separated)
#    an example file: 
#       bes,-65.3
#       browning,-46.6
#       ...
#
# OUTPUT:
#    coordEst: a single coordinate at the maximum of the estimated
#              image; assuming there is one transmitter
#    coordLL:  lower left coordinate (for plotting purposes)
#
def genImageFromMeasurements(filename):

    # Calculate the bounding box for the coordinates
    coordsLL, coordsUR = calcBoundingBoxCoordsDict(coords)

    measurements = pandas.read_csv(filename, delimiter=",", index_col=0, header=None)[1]
    totalSensors = len(measurements)

    # Pull out the coordinates of just the receivers used in this file.
    # And also put them in arrays (not a dictionary)
    sensorCoords = np.zeros((totalSensors, 2)) # Init 2D coords array
    sensorCoordsDict = {}
    rxpowerValues    = np.zeros(totalSensors)
    i = 0
    for key, value in measurements.items(): 
        sensorCoords[i,:]     = coords[key]
        rxpowerValues[i]          = value
        sensorCoordsDict[key] = coords[key]
        i += 1
    plotLocsDictionary(sensorCoordsDict, coordsLL, coordsUR, 1)
        
    # Maximum likelihood method parameters
    n_p           = 3.1   # set the path loss exponent 
    sigmadB       = 12.1      # dB
    d0            = 100   # meters, reference distance
    P0            = -40   # RX power at the reference distance from the source
                          # Assuming transmit power the same as during measurements
    delta_p       = 0.0005  # distance units from lat/long (GPS)coords
    minPL         = 100  # meters. minimum path length d that can be plugged into d^(-n_p)


    # Calculate the ML probability for each pixel in the grid
    imageMat, estTXPower, xVals, yVals = \
        calcMLGrid(sensorCoords, rxpowerValues, delta_p, d0, P0, n_p, sigmadB, minPL)
 
    # plot the estimated images (probability and estimated TX power)
    plotImageMat(imageMat, sensorCoordsDict, coordsLL, coordsUR, delta_p, 'Likelihood', 2)

    #xscale = imageMat.shape[0] / (coordsUR[1]-coordsLL[1])
    #yscale = imageMat.shape[1] / (coordsUR[0]-coordsLL[0])
    #xTicks = plt.xticks()[0] / xscale 
    #yTicks = plt.yticks()[0] / yscale
    

    plotImageMat(estTXPower, sensorCoordsDict, coordsLL, coordsUR, delta_p, 'Est Tx Power', 3)

    # find location estimate
    coordEst      = imageMaxCoord(imageMat, xVals, yVals)

    # Plot the estimated coordinate
    plt.figure(1)
    plt.plot(coordEst[1]-coordsLL[1], coordEst[0]-coordsLL[0], 'rx', markersize=12.0)
    plt.show()

    return coordEst, coordsLL, coordsUR


# Main
plt.figure(1)
plt.cla()
plt.ion()

# Load the rxpower measurements at the receivers
if len(sys.argv) == 1:
    actualSite    = 'basic'
    filename      = 'test_'+ actualSite + '.csv'
    actualCoord   = coords[actualSite]
else:
    filename = sys.argv[1]

# Estimate the image (and the tx location from the image max)
coordEst, coordsLL, coordsUR = genImageFromMeasurements(filename)


if len(sys.argv) == 1:
    # Relate the estimate to the actual coordinate
    coordEstError = calcDistLatLong(coordEst, actualCoord)

    # Plot the estimate, actual, and display error in meters
    plt.figure(1)
    plt.plot(actualCoord[1]-coordsLL[1], actualCoord[0]-coordsLL[0], 'go', markersize=12.0)
    plt.draw()
    plt.tight_layout()
    print("Coordinate estimate error (m): ", coordEstError)
else:
    print("Coordinate estimate: ", coordEst)

input()
