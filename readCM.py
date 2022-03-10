# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 11:21:53 2021

@author: Cryss
"""

from numpy import (
    linspace,array,arange, log,exp,sin,cos,sqrt, pi, zeros, ones, round,
    amin, amax, mean ,
    )
import numpy as np
import sys
import datetime as dt
import calendar as cal


def runData(data, startMonth, endMonth, startDay, endDay, ci):
    temps = [[],[],[]] #create a list to hold max, avg, and min
    percip = []
    wind = [[],[]]
    for irow in range(0,np.shape(data)[0],24):
       
        year = int(data[irow,0]) # Fill variables with each quantity in the row:
        month = int(data[irow,1])
        day = int(data[irow,2])
        
        start = dt.datetime(year, startMonth, startDay)
        date = dt.datetime(year, month, day)
        if endMonth < startMonth:
            end = dt.datetime(year+1, endMonth, endDay)
        else:
            end = dt.datetime(year, endMonth, endDay)
        
        
        if date >= start and date <= end:     #if date falls between given time period
            temps[0].append(amax(data[irow:irow+24,4])-273.15) # array of the max temp 
            temps[1].append(mean(data[irow:irow+24,4])-273.15) # array of the avg temp
            temps[2].append(amin(data[irow:irow+24,4])-273.15) # array of the min temp
            percip.append(np.sum(data[irow:irow+24,5])) # precip mm in last hour
            wind[0].append(mean(data[irow:irow+24,6])) # m/s  eastward componnent of horizontal wind 
            wind[1].append(mean(data[irow:irow+24,7])) # m/s northward componnent of horizontal wind
            
    percents = [np.percentile(temps[0], ci),
                np.percentile(temps[1], ci),
                np.percentile(temps[2], ci)]
    percents = np.asarray(percents)
    temps = np.asarray(temps).T
    wind = np.asarray(wind).T
    #percents returns the confidence intervals ordered for MAX (index 0), AVG (index 1), MIN (index 2)
    #temps returns the temperatures in columns, where column 0 is MAX, column 1 is AVG, column 2 is MIN
    return percents, temps, percip, wind

def relHum(Td, T):
    act = 6.11 * 10 ** ((7.5*Td)/(237.3+Td))
    sat = 6.11 * 10 ** ((7.5*T)/(237.3+T))
    rh = (act/sat) * 100
    return rh

def fit_line1(x, m, b):
    return m*x+b

# def parse(files):
#     shell = [[],[],[],[],[],[],[]]
#     events = []
#     windhold = [[],[],[]]
#     #summerdata = [[],[],[],[],[]]
    
#     for i in range((len(files))):
        
#         # Use numpy loadtxt to read in the datafile
#         data=np.loadtxt(files[i]) # read in the whole file to numpy array data
#         nrows = np.shape(data)[0] # get the number of rows as first index of array data
#         # Initialize quantities used below for sums and averages
#         Tavg = 0                                                                        
#         N_Tavg = 0
#         Ndays_T35 = 0
#         y = np.char.strip(files[0], 'ccsm4_KSEA_.dat')
#         int(y)
#         styear = y
#         endyear = int(data[-1,0])       #get the last year of the data
#         D = array([])
#         intervals, temps, rain, wind = runData(data, stmonth, endmonth, stday, endday, ci)
        
#         DAYTemp_thresh = np.append(DAYTemp_thresh, intervals[0])
#         NIGHTTemp_thresh = np.append(NIGHTTemp_thresh, intervals[2])
        
#         DAYAvg = np.append(DAYAvg, intervals[1])
    
#         above_hisDAY = temps[0] >= his_per[0]
#         above_hisDAY.sum() #number of times the temp exceed day time historic threshold
#         occurencesD = np.append(occurencesD, above_hisDAY.sum())
    
#         above_hisNIGHT = temps[2] >= his_per[2]
#         above_hisNIGHT.sum() #number of times the temp exceed night time historic threshold
#         occurencesN = np.append(occurencesN, above_hisNIGHT.sum())
        
        
#         Dthreshold = his_per[0] #confidence interval stats
#         for irow in range(0,nrows,24):
#             year = int(data[irow,0])
#             month =  int(data[irow,1])
#             day =  int(data[irow,2])
            
#             start = dt.datetime(year, stmonth, stday)
#             date = dt.datetime(year, month, day)
            
#             """
#             wind stuff here. grabs from every day
#             """
#             windhold[0].append(date)
#             windhold[1].append(sqrt(mean(data[irow:irow+24,5])**2+mean(data[irow:irow+24,5])**2))
#             theta = np.arctan2(mean(data[irow:irow+24,6]),mean(data[irow:irow+24,7]))
#             windhold[2].append(np.degrees(theta))
            
            
#             if endmonth < stmonth:
#                 end = dt.datetime(year+1, endmonth, endday)
#             else:
#                 end = dt.datetime(year, endmonth, endday)
            
            
#             if date >= start and date <= end:
#                 # summerdata[0].append(date)
#                 # summerdata[1].append(data[irow:irow+24,4]-273.15)
#                 # summerdata[2].append(data[irow:irow+24,5])
#                 # summerdata[3].append(data[irow:irow+24,6])
#                 # summerdata[4].append(data[irow:irow+24,7])
#                 T2max = amax(data[irow:irow+24,4])-273.15 # 2-m min temp in C 
#                 #D = np.append(D,[TTmax[i] for i in range(len(TTmax)) if TTmax[i] >= Dthreshold])
#                 if T2max > Dthreshold: #apply threshold to filter T2max
#                     D = np.append(D , irow)
        
        
#         D = D.astype(int) #allows D array to be used as indicies
#         sequences = np.split(D, np.array(np.where(np.diff(D) > 24)[0]) + 1)
#         hold = arange(len(sequences))
    
#         for a in range(len(hold)):
#             year = data[sequences[hold[a]] ,0]
#             month = data[sequences[hold[a]] ,1]
#             day = data[sequences[hold[a]] ,2]
#             d1 = dt.datetime(int(year[0]),int(month[0]),int(day[0]))
#             d2 = dt.datetime(int(year[-1]),int(month[-1]),int(day[-1]))
            
#             diff = (d2-d1).days + 1
#             #if diff >= 3:
                
#             # if diff == 1:
#             #     print("{} day,  starting on {}".format(diff,d1.date()))
#             # else:
#             #     print("{} days, starting on {}".format(diff,d1.date()))
            
#             events.append(data[sequences[hold[a]], :])
        
#         for a in D:
#             date = dt.date(int(data[a,0]),int(data[a,1]),int(data[a,2]))
#             shell[0].append(date)
#             shell[1].append(amax(data[a:a+24,4])-273.15)
#             shell[2].append(mean(data[a:a+24,4])-273.15)
#             shell[3].append(amin(data[a:a+24,4])-273.15)
#             # shell[1].append(((amax(data[a:a+24,4])-273.15))*9/5+32)
#             # shell[2].append(((mean(data[a:a+24,4])-273.15))*9.5+32)
#             # shell[3].append(((amin(data[a:a+24,4])-273.15))*9.5+32)
#             shell[4].append(sum(data[a:a+24,5])/10)
#             shell[5].append(max(data[a:a+24,6]))
#             shell[6].append(max(data[a:a+24,7]))
#         return shell, events, windhold