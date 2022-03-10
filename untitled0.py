# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:52:46 2021

@author: Chris
"""

import numpy as np
from numpy import (
    linspace,array,arange, log,exp,sin,cos,sqrt, pi, zeros, ones, round,
    amin, amax, mean , real, imag
    )
from datetime import datetime, date, time, timedelta
import pandas as pd
import csv
import os


def convert_data(data):
    lst = [[],[]]
    
    for i in range(len(data[0])):
        lst[0].append(datetime.strptime(data[0][i],'%Y-%m-%d %H:%M:%S'))
        lst[1].append(int(data[1][i]))
    
    lst = np.asarray(lst).T
    return lst

def removeAverage(data,davg):
    for i in range(24):
        for j in range(len(data)):
            if data[j,0].hour == i:
                data[j,1] -= davg[i]

files = os.listdir()
files.sort()

vancouver = np.loadtxt(files[0],delimiter=',',dtype='str', usecols=(0, 1), unpack=True)
wenatchee = np.loadtxt(files[2],delimiter=',',dtype='str', usecols=(0, 1), unpack=True)
aberdeen = np.loadtxt(files[3],delimiter=',',dtype='str', usecols=(0, 1), unpack=True)
portland = np.loadtxt(files[4],delimiter=',',dtype='str', usecols=(0, 1), unpack=True)
seattle = np.loadtxt(files[6],delimiter=',',dtype='str', usecols=(0, 1), unpack=True)
richland = np.loadtxt(files[5],delimiter=',',dtype='str', usecols=(0, 1), unpack=True)
chehalis = np.loadtxt(files[1],delimiter=',',dtype='str', usecols=(0, 1), unpack=True)

vancouver = convert_data(vancouver)
wenatchee = convert_data(wenatchee)
aberdeen = convert_data(aberdeen)
portland = convert_data(portland)
seattle = convert_data(seattle)
richland = convert_data(richland)
chehalis = convert_data(chehalis)

# over = [[],[]]

# for i in range(1,len(seattle)):
#     if (seattle[i,0] - seattle[i-1,0]) > timedelta(hours=1):
#         over[0].append(i-1)
#         over[1].append(i)
#     elif (seattle[i,0] - seattle[i-1,0]) < timedelta(hours=1):
#         continue       
#     elif (seattle[i,0] - seattle[i-1,0]) == timedelta(hours=1):
#         print(seattle[i-1,0])
#         print(seattle[i,0])
#         print('\n')


from scipy.fft import fft
from scipy.interpolate import interp1d

# for i in range(len(over[0])):
#     # print(seattle[over[0][i]])
#     # print(seattle[over[1][i]])
#     # print('\n')
    
#     print(seattle[over[1][i],0]-seattle[over[0][i],0])

    
    
def interpData(data):
    from scipy.interpolate import interp1d
    hold = data[:,0]
    for i in range(len(data)):     
        # print(seattle[i,0].timestamp() - seattle[i-1,0].timestamp())
        hold[i] = (hold[i].timestamp())

    f = interp1d(hold,data[:,1])

    x = arange(hold[0],hold[-1],3600)
    y = f(x)


    new = []
    for i in range(len(x)):
        new.append(datetime.fromtimestamp(x[i]))

    return new, y
    
