#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors:
    Chris McJunkin

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
from readCM import relHum

from matplotlib.pyplot import figure,subplot,plot,legend
import matplotlib.pyplot as plt

os.chdir('hw/Tmax99')

files = os.listdir()

tmax_dict = []
tmin_dict = []

for i in range(len(files)):
    data = np.load(files[i],allow_pickle=True)
    tmax_dict.append(data)
    
os.chdir('../Tmin99')
files = os.listdir()

for i in range(len(files)):
    data = np.load(files[i],allow_pickle=True)
    tmin_dict.append(data)


model_names = ['canesm2', 'ccsm4', 'csiro-mk3-6-0', 'fgoals-g2', 'gfdl-cm3',
               'giss-e2-h', 'miroc5', 'mri-cgcm3', 'noresm1-m']


c = 0
avg = []

for a in range(len(tmin_dict)):

    """
    ----uses Tmin for humidity, but this is warm bias. after getting humidity from data,
        this becomes obsolete...?
    """
    humidity = zeros(tmin_dict[a].shape[0])
    
    for i in range(len(humidity)):
        humidity[i] = relHum(tmin_dict[a][i,3], tmin_dict[a][i,1])


# night_humidex = zeros(humidity.shape[0])
# day_humidex = zeros(humidity.shape[0])
# for i in range(len(night_humidex)):
#     night_humidex[i] = tmin_dict[a][i,3] + (6.112 * (10 ** 7.5) * humidity[i] - 10)
#     day_humidex[i] = tmin_dict[a][i,1] + (6.112 * (10 ** 7.5) * humidity[i] - 10)

    """
    collect wind info
    """

    # wind_x = tmin_dict[a][:,5]
    # wind_y = tmin_dict[a][:,6]
    
    # wind = zeros((len(wind_x),2))
    
    # for i in range(len(wind)):
    #     wind[i,0] = sqrt(wind_x[i]**2+wind_y[i]**2)
    #     theta = np.arctan2(wind_y[i],wind_x[i])
    #     wind[i,1] = np.degrees(theta)
    
    # #identifying easterly heatwaves. lines 265 and 266 change filter options
    # easterly = [[],[],[],[],[],[]]
    # for i in range(len(wind)):
    #     #if wind_x[i] < 0 and abs(wind_x[i]) > abs(wind_y[i]):
    #     if 135 <= wind[i,1] <= 180 or -180 <= wind[i,1] <= -135:
    #         date = tmin_dict[0][i,0]
    #         temp = round(tmin_dict[a][i,2]*9/5+32,2)
    #         mx = round(tmin_dict[a][i,1]*9/5+32,2)
    #         mn = round(tmin_dict[a][i,3]*9/5+32,2)
    #         easterly[0].append(date)
    #         easterly[1].append(mx)
    #         easterly[2].append(temp)
    #         easterly[3].append(mn)
    #         easterly[4].append(wind[i,0])
    #         easterly[5].append(round(humidity[i],0))
    # easterly = np.asarray(easterly).T
    # print(model_names[a] + ': Number of Easterlies')
    # print(len(easterly))
    # avg.append(len(easterly))
    
    # print('\n')
    # """
    # determine length between each event
    # """
    # days = array([])
    # for i in range(1,len(easterly)):
    #     days = np.append(days, (easterly[i,0]-easterly[i-1,0]).days)
    
    # plus135 = linspace(135,135,len(wind[:,1]))
    # neg135 = linspace(-135,-135,len(wind[:,1]))
    
    # figure(c)
    # plot(wind[:,0],wind[:,1],"bo",markersize=2)
    # plot(wind[:,0],plus135)
    # plot(wind[:,0],neg135)
    # plt.ylim(-185, 185)
    # plt.grid()
    # plt.ylabel('Wind Direction (Degrees)')
    # plt.xlabel('Wind Speed m/s')
    # plt.title(model_names[a] + ": Direction vs Speed of Easterlies")
    # legend(model_names[a], loc=2, fontsize='x-small')
    # c+=1
    
    # #x axis dates, y axis average temp
    # figure(c)
    # plot(easterly[:,0],easterly[:,2],'.')
    # plt.ylabel('Temperature')
    # plt.xlabel('Date')
    # plt.title(model_names[a] + ": Temperature of Easterlies")
    # legend(model_names[a], loc=2, fontsize='x-small')
    # c+=1
        
    
    # #plotting time between each event in years
    # figure(c)
    # plot(easterly[1:,0],(days[:]/365),'.')
    # plt.ylabel('Time between Previous Event (Years)')
    # plt.xlabel('Date of event')
    # plt.title(model_names[a] + ": Time Delta Between Easterly Heat Events")
    # legend(model_names[a], loc=2, fontsize='x-small')
    # c+=1
   

# print('Average # of events for all models: {}'.format(mean(avg)))

events = []
for m in range(len(tmin_dict)):
    # m = 2
    model = tmin_dict[m] 
    year = ones(len(model))*365
    days = [[],[]]
    # ev = []
    # hold = []
    for i in range(1,len(model)):
        if ((model[i,0]-model[i-1,0]).days) > 1:
            days[1].append((model[i,0]-model[i-1,0]).days)
            days[0].append(model[i-1,0])
        
        # if ((model[i,0] - model[i-1,0]).days) == 1:
        #     ev.append(model[i-1])
        # else:
        #     hold.append(ev)
        #     ev = []

    # events.append(hold)  
    days = np.asarray(days).T
    events.append(days)
    year = ones(len(days))*365
    # figure()
    # plot(days[:,0], days[:,1], '.')
    # plot(days[:,0],year)

    # plt.title(model_names[m])
    plt.title('Number of Days between the Beginning and End of Events')
    plt.xlabel('Date')
    plt.ylabel('Days')
    
events_avg_length = []
for m in events:
    events

sys.exit()

for m in tmin_dict:
    figure()
    plot(m[:,0],m[:,3],'.')
    plt.title('Tmin of events')


"""
--------------------------------------------------------
"""

events = big[2]
eventsnew = []
number_of_events = []
for model in events:
    evmodel = []
    number_of_events.append(len(model))
    for ev in model:
        event = []
        
        # for day in ev:
        #     end = dt.datetime(year+1, endmonth, endday)
        #     date = dt.datetime(day[0],day[1],day)
    # eventsnew.append(evmodel)

figure()
# plt.grid(axis='y')
plt.bar(model_names, number_of_events, color=['rosybrown', 'indianred', 'brown', 
                                              'firebrick', 'coral', 'salmon', 
                                              'chocolate','peru', 'sienna'])
plt.xticks(rotation=45)
plt.title('Number of Events per Model')
plt.ylabel('Events')
ev_avg = mean(number_of_events)
ev_std = np.std(number_of_events)
std_err = ev_std / 4

plot(ones(9)*ev_avg, color='k')
plot((ones(9)*ev_avg+ev_std), color='k')
plot((ones(9)*ev_avg-ev_std), color='k')
plt.errorbar(model_names, (ones(9)*ev_avg+ev_std), yerr = std_err, capsize = 6, capthick = 1, fmt = 'ko',ecolor='k')
plt.errorbar(model_names, (ones(9)*ev_avg-ev_std), yerr = std_err, capsize = 6, capthick = 1, fmt = 'ko',ecolor='k')





dbin = 2 # bin size
offset=dbin
low = 5
high = 50
bins = arange(low-dbin/2, high+dbin/2, dbin)
y,x = np.histogram(model_names, bins)
plt.bar(x[:-1]+dbin/2+dbin/8, y, dbin/4, color='red', label='Tmax')
y,x = np.histogram(heat_wave_data[:,3], bins)
plt.bar(x[:-1]+dbin/2-dbin/8, y, dbin/4, color='blue', label='Tmin')
            
        


attr = [d.year > 2020 for d in days[:,0]]

# df['column'] = df['column'].dt.strftime('%Y-%m')


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
fig1.suptitle("MJJAS Temperature in the 99th percentile")
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
