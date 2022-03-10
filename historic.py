# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:24:22 2021

@author: Chris
"""

from numpy import (
    linspace,array,arange, log,exp,sin,cos,sqrt, pi, zeros, ones, round,
    amin, amax, mean ,
    )
import numpy as np
import sys
from scipy import stats
from scipy.optimize import curve_fit
import datetime as dt
import calendar as cal
import os
from matplotlib.pyplot import figure,subplot,plot,legend
import matplotlib.pyplot as plt
from readCM import runData, relHum


KSEA = '../data/Daily_Hist_Thunder/KSEA_hist_thunder.txt'
GoldBar = '../data/Daily_Hist_Thunder/GoldBar_hist_thunder.txt'
SnoqualmieFalls = '../data/Daily_Hist_Thunder/SnoqualmieFalls_hist_thunder.txt'
Abderdeen = '../data/Daily_Hist_Thunder/Abderdeen_hist_thunder.txt'



data = np.loadtxt(KSEA)

stmonth = 5
stday = 1
styear = 1970
endmonth = 9
endyear = 1999
endday = cal.monthrange(endyear, endmonth)[1]

ci = 99

year = data[:,0].astype(int)
month = data[:,1].astype(int)
day = data[:,2].astype(int)
Tmax = data[:,3]
Tmin = data[:,4]
Tavg = zeros(len(data))
for i in range(len(data)):
    Tavg[i] = (data[i,3]+data[i,4])/2

hist = [[],[],[],[]] 
# hist = []     

for i in range(len(data)):
    if month[i] >= stmonth and month[i]<= endmonth:
        date = dt.datetime(int(year[i]), int(month[i]), int(day[i])).date()
        
        hist[0].append(date)
        hist[1].append(Tmax[i])
        hist[2].append(Tavg[i])
        hist[3].append(Tmin[i])
        # hist.append(data[i])

hist = np.asarray(hist).T

maxthresh = np.percentile(hist[:,1] , 99)
minthresh = np.percentile(hist[:,3] , 99)

Years = np.arange(1970 , 2021 , 1)

maxhw = []
minhw = []

for i in hist:
    if i[1] >= maxthresh:
        maxhw.append(i)
    if i[3] >= minthresh:
        minhw.append(i)
      
maxhw = np.asarray(maxhw)
minhw = np.asarray(minhw)

hot = ones(len(minhw))*32

years = linspace(1970 , 2021 , len(minhw))

plot(years, minhw[:,3],'.')
plot(years, hot)

plt.ylabel('Temperature')
plt.xlabel('Date of event')
        

        
        
        
        
        