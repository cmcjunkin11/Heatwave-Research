# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:35:24 2021

@author: Chris
"""

import numpy as np
from numpy import (
    linspace,array,arange, log,exp,sin,cos,sqrt, pi, zeros, ones, round,
    amin, amax, mean ,
    )
from datetime import datetime, date, time, timedelta



"""
1). find all entries within the span of the hour for one day

2). loop through each day within that time period appending the entries

3). average all entries (so there should be about 24, an average for each hour)



"""

def hrCheck(array, s, e):
    lst = []

    for hr in range(len(array)):
        if s <= array[hr,0] < e:
            lst.append(array[hr,1])

    return np.nanmean(lst)
    

def hourlyAverage(array):
    hrly_avgs = zeros(24)
    
    
    for h in range(24):
        
        start = array[0,0]
        start = start.replace(minute=0)
        start += timedelta(hours=h)

        
        hd = timedelta(hours=1)
        end = start + hd
        end = end.replace(minute=0)

        
        avglst = []
        while end.date() <= array[-1,0].date():
            avglst.append(hrCheck(array,start, end))
            start += timedelta(days=1)
            end += timedelta(days=1)
    
        hrly_avgs[h] = np.nanmean(avglst)
        h += 1
    return hrly_avgs


def maxMinDiff(array, name, f_max=True,f_min=True,f_diff=True,all_diff=False):
    
    from matplotlib.pyplot import figure,subplot,plot,legend
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    d = datetime(2021,6,16)
    
    test = array    
    # test_avg = a
    
    maxidx = []
    minidx = []
    
    
    while d.date() <= test[-1,0].date():
        max_index = 0
        max_value = 0
        
        min_index = 0
        min_value = 100
        
        for i in range(len(test)):
            if d.date() == test[i,0].date():
                if max_value < test[i,1]:
                    max_value = test[i,1]
                    max_index = i
                if min_value > test[i,1]:
                    min_value = test[i,1]
                    min_index = i
        maxidx.append(max_index)
        minidx.append(min_index)
        d += timedelta(days=1)
    
    tmx = [[],[]]
    tmn = [[],[]]
    
    
    for n in maxidx:
        tmx[0].append(test[n,0])
        tmx[1].append(test[n,1])
    
    for n in minidx:
        tmn[0].append(test[n,0])
        tmn[1].append(test[n,1])
    
    
    
    # for i in range(len(tmx[0])):
    #     # print(t[0][i].hour)
    #     tmx[1][i] -= test_avg[tmx[0][i].hour]
    
    # for i in range(len(tmn[0])):
    #     # print(t[0][i].hour)
    #     tmn[1][i] -= test_avg[tmn[0][i].hour]
    
    
    
    
    diff = np.subtract(tmx[1],tmn[1])
    
    diff_avg = ones(len(diff))*np.average(diff)
    diff_std_lower = np.subtract(diff_avg,ones(len(diff))*np.std(diff))
    diff_std_upper = np.add(diff_avg,ones(len(diff))*np.std(diff))
    
    tmx_avg = ones(len(tmx[1]))*np.average(tmx[1])
    tmx_std_upper = np.add(tmx_avg,ones(len(tmx[1]))*np.std(tmx[1]))
    tmx_std_lower = np.subtract(tmx_avg,ones(len(tmx[1]))*np.std(tmx[1]))
    
    tmn_avg = ones(len(tmn[1]))*np.average(tmn[1])
    tmn_std_upper = np.add(tmn_avg,ones(len(tmn[1]))*np.std(tmn[1]))
    tmn_std_lower = np.subtract(tmn_avg,ones(len(tmn[1]))*np.std(tmn[1]))
    
    if all_diff:
        f_max=False
        f_min=False
        f_diff=False
        plt.figure(dpi=1000)
        plot(tmx[0],diff)
        plot(tmx[0],diff_avg)
        plot(tmx[0],diff_std_upper)
        plot(tmx[0],diff_std_lower)
        plt.xticks(rotation = 45)
        plt.xlabel('Day')
        plt.ylabel('Temerature Difference (F)')
        plt.grid()
        plt.title(name+': Tmax-Tmin with avg and 1 sigma')
        
    
    if f_diff:
        figure()
        plt.figure(dpi=1000)

        plot(tmx[0],diff)
        plot(tmx[0],diff_avg)
        plot(tmx[0],diff_std_upper)
        plot(tmx[0],diff_std_lower)
        plt.xticks(rotation = 45)
        plt.xlabel('Day')
        plt.ylabel('Temerature Difference (F)')
        plt.grid()
        plt.title(name+': Tmax-Tmin with avg and 1 sigma')
    
    if f_max:
        figure()
        plt.figure(dpi=1000)

        plot(tmx[0],tmx[1])
        plot(tmx[0],tmx_avg)
        plot(tmx[0],tmx_std_upper)
        plot(tmx[0],tmx_std_lower)
        plt.xticks(rotation = 45)
        plt.xlabel('Day')
        plt.ylabel('Temerature (F)')
        plt.grid()
        plt.title(name+': Tmax with Tmax_avg and 1 sigma')
        
    if f_min:
        figure()
        plt.figure(dpi=1000)

        plot(tmn[0],tmn[1])
        plot(tmn[0],tmn_avg)
        plot(tmn[0],tmn_std_upper)
        plot(tmn[0],tmn_std_lower)
        plt.xticks(rotation = 45)
        plt.xlabel('Day')
        plt.ylabel('Temerature (F)')
        plt.grid()
        plt.title(name+': Tmin with Tmin_avg and 1 sigma')





# plt.hist(seattle[:,1],bins=int(np.ceil(sqrt(len(seattle)))))
# plt.hist(portland[:,1],bins=int(np.ceil(sqrt(len(portland)))))
# plt.hist(wenatchee[:,1],bins=int(np.ceil(sqrt(len(wenatchee)))))
# plt.hist(vancouver[:,1],bins=int(np.ceil(sqrt(len(vancouver)))))

# plt.hist(aberdeen[:,1],bins=int(np.ceil(sqrt(len(aberdeen)))))



"""
for the entire length of the array:
    check if timestamp is between hourly period
        if yes, then add the index to an array

for each 
"""