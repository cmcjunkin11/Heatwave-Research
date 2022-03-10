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

#first load files
#function to do loop. give a data set, spit out some shit.
#returns percentiles, arrays of 

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
        if endmonth < stmonth:
            end = dt.datetime(year+1, endmonth, endday)
        else:
            end = dt.datetime(year, endmonth, endday)
        
        
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

def gaussian(x, mu):
    sig = sqrt(mu)
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def relHum(Td, T):
    act = 6.11 * 10 ** ((7.5*Td)/(237.3+Td))
    sat = 6.11 * 10 ** ((7.5*T)/(237.3+T))
    rh = (act/sat) * 100
    return rh

os.chdir("../data/Hourly_ccsm4")
files = [x for x in os.listdir() if '.dat' and not 'hrly' in x]
files.sort()

init = np.loadtxt(files[0])    #save the first file of data from 1970s-1990s
init_rows = np.shape(init)[0]   #then chop it off to run the rest of the data
#temporary fix
files = np.delete(files, 0) #ignore 1970s to 1990s data


DAYTemp_thresh = array([]) 
NIGHTTemp_thresh = array([])
DAYAvg = array([])
occurencesD = array([])
occurencesN = array([])

"""
3 decades (historic)
--------------------------------------------------
"""

#get first year of files, and use as styear
stmonth = 5
stday = 1
styear = int(init[0,0])

endmonth = 9
endyear = int(init[-1,0])       #get the last year of the data
endday = cal.monthrange(endyear, endmonth)[1] 

ci = 99

his_per, his_temp, _, _ = runData(init, stmonth, endmonth, stday, endday, ci)   # python expects return values
                                                                                # but _ tells python to forget about
                                                                                #returning values

"""
single decade
--------------------------------------------------
"""

shell = [[],[],[],[],[],[],[]]
events = []
windhold = [[],[],[]]
#summerdata = [[],[],[],[],[]]

for i in range((len(files))):
    
    # Use numpy loadtxt to read in the datafile
    data=np.loadtxt(files[i]) # read in the whole file to numpy array data
    nrows = np.shape(data)[0] # get the number of rows as first index of array data

    
    # Initialize quantities used below for sums and averages
    Tavg = 0                                                                        
    N_Tavg = 0
    Ndays_T35 = 0
    
    y = np.char.strip(files[0], 'ccsm4_KSEA_.dat')
    int(y)
    styear = y
    endyear = int(data[-1,0])       #get the last year of the data
    
    D = array([])
    intervals, temps, rain, wind = runData(data, stmonth, endmonth, stday, endday, ci)
    
    DAYTemp_thresh = np.append(DAYTemp_thresh, intervals[0])
    NIGHTTemp_thresh = np.append(NIGHTTemp_thresh, intervals[2])
    
    DAYAvg = np.append(DAYAvg, intervals[1])

    above_hisDAY = temps[0] >= his_per[0]
    above_hisDAY.sum() #number of times the temp exceed day time historic threshold
    occurencesD = np.append(occurencesD, above_hisDAY.sum())

    above_hisNIGHT = temps[2] >= his_per[2]
    above_hisNIGHT.sum() #number of times the temp exceed night time historic threshold
    occurencesN = np.append(occurencesN, above_hisNIGHT.sum())
    
    
    Dthreshold = his_per[2] #confidence interval stats
    for irow in range(0,nrows,24):
        year = int(data[irow,0])
        month =  int(data[irow,1])
        day =  int(data[irow,2])
        
        start = dt.datetime(year, stmonth, stday)
        date = dt.datetime(year, month, day)
        
        """
        wind stuff here. grabs from every day
        """
        # windhold[0].append(date)
        # windhold[1].append(sqrt(mean(data[irow:irow+24,5])**2+mean(data[irow:irow+24,5])**2))
        # theta = np.arctan2(mean(data[irow:irow+24,6]),mean(data[irow:irow+24,7]))
        # windhold[2].append(np.degrees(theta))
        
        
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
            T2max = amax(data[irow:irow+24,4])-273.15 # 2-m min temp in C 
            T2min = amin(data[irow:irow+24,4])-273.15 
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
        
        # wind[i,0] = sqrt(wind_x[i]**2+wind_y[i]**2)
        # theta = np.arctan2(wind_y[i],wind_x[i])
        # wind[i,1] = np.degrees(theta)
       
        # rmax = np.argmax(data[a:a+24,4])
        # rmin = np.argmin(data[a:a+24,4])
        
        """
        grabbing the winds at Tmax of the day
        """
        # shell[5].append(data[rmin,6])
        # shell[6].append(data[rmin,7])
        """
        grabbing average of the winds
        """
        shell[5].append(mean(data[a:a+24,6]))
        shell[6].append(mean(data[a:a+24,7]))


#maxloc
#

windhold = np.asarray(windhold).T
heat_wave_data = np.asarray(shell)
heat_wave_data = heat_wave_data.T
del shell



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
    #if wind_x[i] < 0 and abs(wind_x[i]) > abs(wind_y[i]):
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
for i in range(1,len(easterly)):
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

sys.exit()

#plotting time between each event in years
figure(2)
plot(easterly[1:,0],(days[1:]/365),'.')
plt.ylabel('Time between Previous Event (Years)')
plt.xlabel('Date of event')



# m = 0.1
# b = 5

# years = zeros(len(days[1:]))
# for yr in range(len(easterly)):
#     try:
#         years[yr] = easterly[yr+1,0].year
#     except:
#         continue
# # # fit_params, fit_cov = curve_fit(fit_line, years, days/365, (m,b), sigma=None, absolute_sigma='True')
# # # deltax1_fit = np.linspace(easterly[1,0].year, easterly[-1,0].year, 500)
# # # deltay1_fit = fit_line(deltax1_fit, fit_params[0], fit_params[1])
# # # plot(deltax1_fit, deltay1_fit, '-k')

# from numpy.polynomial.polynomial import polyfit
# b, m = polyfit(years, days[1:]/365, 1)
# plot(years, b + m * years, '-')

plot(heat_wave_data[:,0],wind[:,1],"bo",markersize=2)
plot(windhold[:,0],windhold[:,2],'.')
plt.xticks(fontsize=7)

plot(heat_wave_data[:,0],heat_wave_data[:,1])
plot(heat_wave_data[:,0],(heat_wave_data[:,1]-heat_wave_data[:,3]))

decades = arange(1970,2100,10)

fig1 = plt.figure(1)
fig1.suptitle("MJJAS Temperature in the {}th percentile".format(ci))
plot(decades,DAYAvg, label='Average Day CI')
plot(decades,DAYTemp_thresh, label='Day Temp CI')
plot(decades,NIGHTTemp_thresh, label='Night Temp CI')
plot(decades,ones(len(decades))*his_per[1], label='Historic Average Day CI')
legend(loc=2, fontsize='x-small')
plt.xlabel("Decades (1970s-2090s)")
plt.ylabel("Temperature in C")


fig2, (ax1, ax2) = plt.subplots(2)
fig2.suptitle('MJJAS Temperature and Daily Rainfall in the {}th percentile'.format(ci))
dates = heat_wave_data[:,0]
rain = heat_wave_data[:,4]/10

#filter out 0's
for i in range(len(rain)):
    if rain[i] == 0:
        rain[i] = np.nan
        
ax1.plot(dates,heat_wave_data[:,2])
ax1.plot(dates,heat_wave_data[:,1])
ax1.plot(dates,heat_wave_data[:,3])
ax2.plot(dates,rain, 'bo')
ax1.set_ylabel("Temperature (C)")
ax2.set_ylabel("Rainfall (cm)")
legend(loc=2, fontsize='x-small')


shell = [[],[],[]] #reset the shell
m = stmonth
while  m <= endmonth:
    plt.figure()
    plt.ylabel("Temp (C)")
    plt.xlabel("Year")
    plt.suptitle("Average Temperatures in Month {}".format(m))
    for i in heat_wave_data:
        if i[0].month == m:
            shell[0].append(i[0])
            shell[1].append(i[2])#*9/5 + 32)
            #shell[2].append(sqrt(i[2]*9/5 + 32))
    plt.scatter(shell[0],shell[1])
    #plt.scatter(shell[0],shell[2], color="orange")
    m += 1
    shell = [[],[],[]] #reset the shell


plt.figure()
dbin = 2 # bin size
offset=dbin
low = 5
high = 50
bins = arange(low-dbin/2, high+dbin/2, dbin)
y,x = np.histogram(heat_wave_data[:,1], bins)
plt.bar(x[:-1]+dbin/2+dbin/8, y, dbin/4, color='red', label='Tmax')
y,x = np.histogram(heat_wave_data[:,3], bins)
plt.bar(x[:-1]+dbin/2-dbin/8, y, dbin/4, color='blue', label='Tmin')


#scipy stats

#from scipy import (stats, slope, intercept, r_value, p_value, std_err = stats.linregress(xfit,yfit))
