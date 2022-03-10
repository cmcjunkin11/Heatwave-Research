#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors:
    Adrienne Beggs
    Chris McJunkin
    Eric Salathe
    Satveer Sandhu

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


# os.chdir("../data/Hourly_ccsm4")
# files = [x for x in os.listdir() if '.dat' and not 'hrly' in x]
# # files = [x for x in os.listdir() if 'ccsm4' in x]
# files.sort()

"""
files management
1. get a list of the data files
2. make a list of the models by chopping off the stuff we don't want
3. cutdown the models list, so we don't have repeating values
"""
os.chdir("../data/Hourly")
# files = [x for x in os.listdir() if '.dat' and 'cane' in x]
files = os.listdir()
files.sort()
model_names = ['canesm2', 'ccsm4', 'csiro-mk3-6-0', 'fgoals-g2', 'gfdl-cm3',
               'giss-e2-h', 'miroc5', 'mri-cgcm3', 'noresm1-m']

"""
3 decades (historic)
--------------------------------------------------
"""

#get first year of files, and use as styear
stmonth = 5
stday = 1
styear = 1970
endmonth = 9
endyear = 1999
endday = cal.monthrange(endyear, endmonth)[1]
decades = arange(1970,2100,10)

ci = 99
datalist = []
historic_model_data = []
for f in range(0,len(files),13):
    model = files[f:f+13]
    historic_model_data.append(list(model[:3]))
    datalist.append(list(model))



thresholds = [model_names,[]]
for i in historic_model_data:
    init = np.loadtxt(i[0])
    for j in range(1,3):
        d = np.loadtxt(i[j])
        init = np.concatenate((init, d), axis=0)
    print('loading...')
    his_per, _, _, _ = runData(init, stmonth, endmonth, stday, endday, ci)
    thresholds[1].append(his_per)

#thresholds[1]
#          [0-8 for each model]
#          [0-2 for TmaxCI, TavgCI,TminCI]


"""
single decade
--------------------------------------------------
"""

big = [thresholds,[],[]]
#summerdata = [[],[],[],[],[]]
c = 0
for m in datalist:
    heat_wave_data = []
    print('Model loaded: {}'.format(m))
    shell = [[],[],[],[],[],[],[]]
    events = []
    tau = 0
    for j in m:
        print('File loaded:  {}'.format(j))
        data = np.loadtxt(j)
        nrows = np.shape(data)[0]
        styear = decades[tau]
        print(decades[tau])
        tau += 1
        endyear = int(data[-1,0])       #get the last year of the data
        D = array([])
        intervals, temps, rain, wind = runData(data, stmonth, endmonth, stday, endday, ci)
        Dthreshold = thresholds[1][c][2] #confidence interval stats
        for irow in range(0,nrows,24):
            year = int(data[irow,0])
            month =  int(data[irow,1])
            day =  int(data[irow,2])
            
            start = dt.datetime(year, stmonth, stday)
            date = dt.datetime(year, month, day)
            
            if endmonth < stmonth:
                end = dt.datetime(year+1, endmonth, endday)
            else:
                end = dt.datetime(year, endmonth, endday)
            
            
            if date >= start and date <= end:
                # summerdata[0].append(date)
                # summerdata[1].append(data[irow:irow+24,4]-273.15)
                # summerdata[2].append(data[irow:irow+24,5])
                # summerdata[3].append(data[irow:irow+24,6])
                # summerdata[4].append(data[irow:irow+24,7])
                T2min = amin(data[irow:irow+24,4])-273.15 # 2-m min temp in C 
                #D = np.append(D,[TTmax[i] for i in range(len(TTmax)) if TTmax[i] >= Dthreshold])
                if T2min > Dthreshold: #apply threshold to filter T2max
                    D = np.append(D , irow)
        
        D = D.astype(int) #allows D array to be used as indicies
        sequences = np.split(D, np.array(np.where(np.diff(D) > 24)[0]) + 1)
        hold = arange(len(sequences))
    
        for a in range(len(hold)):
            year = data[sequences[hold[a]] ,0]
            month = data[sequences[hold[a]] ,1]
            day = data[sequences[hold[a]] ,2]
            d1 = dt.datetime(int(year[0]),int(month[0]),int(day[0]))
            d2 = dt.datetime(int(year[-1]),int(month[-1]),int(day[-1]))
            
            diff = (d2-d1).days + 1
            #if diff >= 3:
                
            # if diff == 1:
            #     print("{} day,  starting on {}".format(diff,d1.date()))
            # else:
            #     print("{} days, starting on {}".format(diff,d1.date()))
            
            events.append(data[sequences[hold[a]], :])
        
        for a in D:
            date = dt.date(int(data[a,0]),int(data[a,1]),int(data[a,2]))
            shell[0].append(date)
            shell[1].append(amax(data[a:a+24,4])-273.15)
            shell[2].append(mean(data[a:a+24,4])-273.15)
            shell[3].append(amin(data[a:a+24,4])-273.15)
            shell[4].append(sum(data[a:a+24,5])/10)
            shell[5].append(mean(data[a:a+24,6]))
            shell[6].append(mean(data[a:a+24,7]))
                        
    c += 1   
    big[1].append(np.asarray(shell).T)
    big[2].append((np.asarray(events)).T)
del shell


sys.exit()
#maxloc

windhold = np.asarray(windhold).T
heat_wave_data = np.asarray(shell)
heat_wave_data = heat_wave_data.T



"""
----uses Tmin for humidity, but this is warm bias. after getting humidity from data,
    this becomes obsolete...?
"""

humidity = zeros(heat_wave_data.shape[0])

for i in range(len(humidity)):
    humidity[i] = relHum(heat_wave_data[i,3], heat_wave_data[i,1])

# humidex = zeros(humidity.shape[0])

# for i in range(len(humidex)):
#     humidex[i] = 0.5 * ((heat_wave_data[i,2]*9/5+32) + 61.0 + (((heat_wave_data[i,2]*9/5+32)-68.0)*1.2) + (humidity[i]*0.094))


for i in range(len(model_names)):
    title = big[0][i] + '-Tmin99-heat-wave'
    np.save(title,big[1][i])


#array = np.load(file,allow_pickle=True)



"""
collect wind info
"""

wind_x = heat_wave_data[:,5]
wind_y = heat_wave_data[:,6]

wind = zeros((len(wind_x),2))

for i in range(len(wind)):
    wind[i,0] = sqrt(wind_x[i]**2+wind_y[i]**2)
    theta = np.arctan2(wind_y[i],wind_x[i])
    wind[i,1] = np.degrees(theta)

#identifying easterly heatwaves. lines 265 and 266 change filter options
easterly = [[],[],[],[],[],[]]
for i in range(len(wind)):
    # if wind_x[i] < 0 and abs(wind_x[i]) > abs(wind_y[i]):
    if 135 <= wind[i,1] <= 180 or -180 <= wind[i,1] <= -135:
        date = heat_wave_data[i,0]
        temp = round(heat_wave_data[i,2]*9/5+32,2)
        mx = round(heat_wave_data[i,1]*9/5+32,2)
        mn = round(heat_wave_data[i,3]*9/5+32,2)
        easterly[0].append(date)
        easterly[1].append(mx)
        easterly[2].append(temp)
        easterly[3].append(mn)
        easterly[4].append(wind[i,0])
        easterly[5].append(round(humidity[i],0))
        
easterly = np.asarray(easterly).T

"""
determine length between each event
"""
days = array([])
for i in range(len(easterly)):
    if i == 0:
        continue
    else:
        days = np.append(days, (easterly[i,0]-easterly[i-1,0]).days)


plus135 = linspace(135,135,len(wind[:,1]))
neg135 = linspace(-135,-135,len(wind[:,1]))

figure(0)
plot(wind[:,0],wind[:,1],"bo",markersize=2)
plot(wind[:,0],plus135)
plot(wind[:,0],neg135)
plt.ylim(-185, 185)
plt.grid()
plt.ylabel('Wind Direction (Degrees)')
plt.xlabel('Wind Speed m/s')

#x axis dates, y axis average temp
figure(1)
plot(easterly[:,0],easterly[:,2],'.')
plt.ylabel('Temperature')
plt.xlabel('Date')


#plotting time between each event in years
figure(2)
plot((days[days>5]/365),'.')
plt.ylabel('Time between Previous Event (Years)')
plt.xlabel('event #')


# plot(heat_wave_data[:,0],wind[:,1],"bo",markersize=2)
# plot(windhold[:,0],windhold[:,2],'.')
# plt.xticks(fontsize=7)

# decades = arange(1970,2100,10)

# fig1 = plt.figure(1)
# fig1.suptitle("MJJAS Temperature in the {}th percentile".format(ci))
# plot(decades,DAYAvg, label='Average Day CI')
# plot(decades,DAYTemp_thresh, label='Day Temp CI')
# plot(decades,NIGHTTemp_thresh, label='Night Temp CI')
# plot(decades,ones(len(decades))*his_per[1], label='Historic Average Day CI')
# legend(loc=2, fontsize='x-small')
# plt.xlabel("Decades (1970s-2090s)")
# plt.ylabel("Temperature in C")


# fig2, (ax1, ax2) = plt.subplots(2)
# fig2.suptitle('MJJAS Temperature and Daily Rainfall in the {}th percentile'.format(ci))
# dates = heat_wave_data[:,0]
# rain = heat_wave_data[:,4]/10

# #filter out 0's
# for i in range(len(rain)):
#     if rain[i] == 0:
#         rain[i] = np.nan
        
# ax1.plot(dates,heat_wave_data[:,2])
# ax1.plot(dates,heat_wave_data[:,1])
# ax1.plot(dates,heat_wave_data[:,3])
# ax2.plot(dates,rain, 'bo')
# ax1.set_ylabel("Temperature (C)")
# ax2.set_ylabel("Rainfall (cm)")
# #legend(loc=2, fontsize='x-small')


# shell = [[],[],[]] #reset the shell
# m = stmonth
# while  m <= endmonth:
#     plt.figure()
#     plt.ylabel("Temp (C)")
#     plt.xlabel("Year")
#     plt.suptitle("Average Temperatures in Month {}".format(m))
#     for i in heat_wave_data:
#         if i[0].month == m:
#             shell[0].append(i[0])
#             shell[1].append(i[2])#*9/5 + 32)
#             #shell[2].append(sqrt(i[2]*9/5 + 32))
#     plt.scatter(shell[0],shell[1])
#     #plt.scatter(shell[0],shell[2], color="orange")
#     m += 1
#     shell = [[],[],[]] #reset the shell


# plt.figure()
# dbin = 2 # bin size
# offset=dbin
# low = 5
# high = 50
# bins = arange(low-dbin/2, high+dbin/2, dbin)
# y,x = np.histogram(heat_wave_data[:,1], bins)
# plt.bar(x[:-1]+dbin/2+dbin/8, y, dbin/4, color='red', label='Tmax')
# y,x = np.histogram(heat_wave_data[:,3], bins)
# plt.bar(x[:-1]+dbin/2-dbin/8, y, dbin/4, color='blue', label='Tmin')


#scipy stats

#from scipy import (stats, slope, intercept, r_value, p_value, std_err = stats.linregress(xfit,yfit))
