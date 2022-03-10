# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:21:39 2021

@author: Chris
"""

import numpy as np
from numpy import (
    linspace,array,arange, log,exp,sin,cos,sqrt, pi, zeros, ones, round,
    amin, amax, mean , real, imag, diff
    )
from datetime import datetime, date, time, timedelta
import pandas as pd
import csv
import os
from hrlyavg import hrCheck, hourlyAverage, maxMinDiff
from untitled0 import interpData


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

# van_avg = hourlyAverage(vancouver)
# wen_avg = hourlyAverage(wenatchee)
# abe_avg = hourlyAverage(aberdeen)
# por_avg = hourlyAverage(portland)
# sea_avg = hourlyAverage(seattle)
# ric_avg = hourlyAverage(richland)
# che_avg = hourlyAverage(chehalis)

# removeAverage(vancouver, van_avg)
# removeAverage(wenatchee, wen_avg)
# removeAverage(aberdeen, abe_avg)
# removeAverage(portland, por_avg)
# removeAverage(seattle, sea_avg)
# removeAverage(richland, ric_avg)
# removeAverage(chehalis, che_avg)


# maxMinDiff(vancouver, 'Vancouver', all_diff=True)
# maxMinDiff(wenatchee, 'Wenatchee', all_diff=True)
# maxMinDiff(aberdeen, 'Aberdeen', all_diff=True)
# maxMinDiff(portland, 'Portland', all_diff=True)
# maxMinDiff(seattle, 'Seattle', all_diff=True)
# maxMinDiff(richland, 'Richland', all_diff=True)
# maxMinDiff(chehalis, 'Chehalis', all_diff=True)


from matplotlib.pyplot import figure,subplot,plot,legend
import matplotlib.pyplot as plt

# figure()
# plt.figure(dpi=1000)
# plot(vancouver[:,0],vancouver[:,1],'.',label='Vancouver')
# plot(wenatchee[:,0],wenatchee[:,1],'.',label='Wenatchee')
# plot(portland[:,0],portland[:,1],'.',label='Portland')
# plot(seattle[:,0],seattle[:,1],'.',label='Seattle')
# plot(aberdeen[:,0],aberdeen[:,1],'.',label='Aberdeen')
# plot(richland[:,0],richland[:,1],'.',label='Richland')
# plot(chehalis[:,0],chehalis[:,1],'.',label='Chehalis')


# plt.xticks(rotation = 45)
# legend(loc=2, fontsize='x-small')
# plt.xlabel('Hour')
# plt.ylabel('Temerature (F)')
# plt.grid()
# plt.title('Hourly Temperatures')

# start = datetime(2021,6,25,0,0)
# end = datetime(2021,6,29,23,59)
# plt.xlim(start,end)



from scipy.fft import fft

van_x,van_y = interpData(vancouver)
wen_x,wen_y = interpData(wenatchee)
por_x,por_y = interpData(portland)
sea_x,sea_y = interpData(seattle)
abe_x,abe_y = interpData(aberdeen)
ric_x,ric_y = interpData(richland)
che_x,che_y = interpData(chehalis)

seattle = np.loadtxt(files[6],delimiter=',',dtype='str', usecols=(0, 1), unpack=True)
seattle = convert_data(seattle)

# figure(dpi=400)
# plot(sea_x,sea_clean, label='Fourier')
# plot(seattle[:,0],seattle[:,1],label='Original')
# legend(loc=2, fontsize='x-small')



van_y = fft(van_y)
wen_y = fft(wen_y)
por_y = fft(por_y)
sea_y = fft(sea_y)
abe_y = fft(abe_y)
ric_y = fft(ric_y)
che_y = fft(che_y)

def fourierAn(data, hourstep=25):
    F = data
    nt = len(F)
    nfmax = int(nt/2)

    #careful, data has variable sampling rate
    dT =  3600    #data gathered at 60 minute intervals
    T =  nt*dT
    t = arange(0,T,dT)
    freqf =  1/T # Hz   fundamental frequency (lowest frequency)
    nfmax = int(nt/2) # number of frequencies resolved by FFT
    
    freqmax = freqf*nfmax # Max frequency (Nyquist)
    
    freq = arange(0,freqmax,freqf)
    
    a = 2*real(F[:nfmax])/nt # form the a coefficients
    a[0] = a[0]/2
    
    b = -2*imag(F[:nfmax])/nt # form the b coefficients
    
    p = sqrt(a**2 + b**2) # form power spectrum
    
    fclean = ones(nt)*a[0] # fill time series with constant term
    
    cutoff_freq = 1/(hourstep*3600)   #25 hours by 1 hour in seconds (25 hour period)
    
    
    for i in range(1,nfmax):
        if freq[i] <= cutoff_freq: # use only frequencies below cutoff_freq
            fclean = fclean + a[i]*cos(freq[i]*2*pi*t) + b[i]*sin(freq[i]*2*pi*t)
    
    
    # plot(t,fclean)
    return fclean

van_clean = fourierAn(van_y)
wen_clean = fourierAn(wen_y)
por_clean = fourierAn(por_y)
sea_clean = fourierAn(sea_y)
abe_clean = fourierAn(abe_y)
ric_clean = fourierAn(ric_y)
che_clean = fourierAn(che_y)


figure()
plt.figure(dpi=400)
# plot(van_x,van_clean, label='Vancouver')
plot(wen_x,wen_clean, label='Wenatchee')
plot(por_x,por_clean, label='Portland')
plot(sea_x,sea_clean, label='Seattle')
# plot(abe_x,abe_clean, label='Aberdeen')
# plot(ric_x,ric_clean, label='Richland')
# plot(che_x,che_clean, label='Chehalis')
plt.grid()
legend(loc=2, fontsize='x-small')
plt.xticks(rotation = 45)

# figure()
# plot(freq,p)
# plt.xlim(0,2*10**-5)

dx = 3600

dy_wen = diff(wen_clean)/dx
dy_por = diff(por_clean)/dx
dy_sea = diff(sea_clean)/dx


figure(dpi=400)
plot(wen_x[:-1],dy_wen)
plot(por_x[:-1],dy_por)
plot(sea_x[:-1],dy_sea)


plt.grid()

plt.xticks(rotation = 45)



# vancouver = pd.read_csv(files[0],header=None)
# vancouver[0] = pd.to_datetime(vancouver[0])

# wenatchee = pd.read_csv(files[2],header=None)
# wenatchee[0] = pd.to_datetime(wenatchee[0])

# aberdeen = pd.read_csv(files[3],header=None)
# aberdeen[0] = pd.to_datetime(aberdeen[0])

# portland = pd.read_csv(files[4],header=None)
# portland[0] = pd.to_datetime(portland[0])

# seattle = pd.read_csv(files[6],header=None)
# seattle[0] = pd.to_datetime(seattle[0])

# richland = pd.read_csv(files[5],header=None)
# richland[0] = pd.to_datetime(richland[0])

# chehalis = pd.read_csv(files[1],header=None)
# chehalis[0] = pd.to_datetime(chehalis[0])




# plot(vancouver[0],vancouver[1],label='Vancouver')
# plot(wenatchee[0],wenatchee[1],label='Wenatchee')
# plot(portland[0],portland[1],label='Portland')
# plot(seattle[0],seattle[1],label='Seattle')
# plot(aberdeen[0],aberdeen[1],label='Aberdeen')
# plot(richland[0],richland[1],label='Richland')
# plot(chehalis[0],chehalis[1],label='Chehalis')