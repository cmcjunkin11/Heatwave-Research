# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:39:59 2021

@author: Chris
"""

import requests
from numpy import (
    linspace,array,arange, log,exp,sin,cos,sqrt, pi, zeros, ones, round,
    amin, amax, mean , 
    )
import numpy as np
from datetime import datetime, date, time, timedelta
import sys
from scipy import stats
from scipy.optimize import curve_fit
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import Firefox
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.keys import Keys
import time as tm

import pandas as pd 

def scrape(u):

    url = u
    tm.sleep(10)
    driver.get(url)
    
    # I understand that at this point the data is in html format and can be
    # extracted with BeautifulSoup:
        
    
    tm.sleep(15)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Close the firefox instance started before:
    
    # hdata = soup.find_all('th')
    # data = soup.find_all('td')
    
    #then need to call get_text() on each element in data
    
    # summary = soup.find(class_='summary-table')
    
    
    obs = soup.find(class_='observation-table')
    
    
    # sumdt = summary[0].get_text()
    
    a = obs.text
    a = a[79:]
    a = a.replace('Fair',';')
    a = a.replace('Partly Cloudy',';')
    a = a.replace('Mostly Cloudy',';')
    a = a.replace('Cloudy',';')
    a = a.replace('Smoke',';')
    a = a.replace('Light Rain',';')
    a = a.replace('Haze',';')
    a = a.replace('Fair / Windy',';')
    a = a.replace('Windy',';')
    a = a.replace('Fog',';')
    a = a.split(';')
    # if len(a) <= 25:
    #     a = a[:24]
        
    # for i in range(len(a[0])-1):
    #     if a[0][i]+a[0][i-1] == 'AM':
    #         print('yep')
            
    # if 'AM' in a[0]:
    #     print('yep')
    # a[0].find('AM') + 4
    # temp = a[0][:a[0].find('AM') + 4]
    data = [[],[]]
    
    for stamp in a:
        am = stamp.find('AM')
        pm = stamp.find('PM')
        if am > 0:
            t = stamp[:am+2]
            
            #convert to 24 hr and datetime object
            in_time = datetime.strptime(t, "%I:%M %p")
            out_time = datetime.strftime(in_time, "%H:%M")
            ftime = datetime.strptime(out_time, "%H:%M")
            
            d = datetime.combine(datetime(start.year, start.month, start.day), time(ftime.hour,ftime.minute))
            
            data[0].append(d)
            temp = stamp[:am+4]
            data[1].append(temp[-2:])
        if pm > 0:
            t = stamp[:pm+2]
            
            #convert to 24 hr and datetime object
            in_time = datetime.strptime(t, "%I:%M %p")
            out_time = datetime.strftime(in_time, "%H:%M")
            ftime = datetime.strptime(out_time, "%H:%M")
            
            d = datetime.combine(datetime(start.year, start.month, start.day), time(ftime.hour,ftime.minute))
            
            data[0].append(d)
            temp = stamp[:pm+4]
            data[1].append(temp[-2:])
    
    
    data = np.asarray(data).T
    driver.close()
    return data



#date to start scraping
start = date(2021,6,20)
end = date(2021,7,10)

profile = webdriver.FirefoxProfile()
profile.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0")

driver = webdriver.Firefox(profile)

# Commands related to the webdriver (not sure what they do, but I can guess):
#bi = FirefoxBinary('C:\Program Files\Mozilla Firefox\firefox.exe')
# driver = webdriver.Firefox(executable_path = 'C:/Users/cmcju/Documents/Geckodriver/geckodriver.exe')
# driver = webdriver.Firefox(executable_path = 'D:/gecko/geckodriver.exe')


data = scrape("https://www.wunderground.com/history/daily/us/wa/seatac/KSEA/date/"+start.isoformat())
pd.DataFrame(data).to_csv("D:/1 CLIMATE/wunderground/oops.csv", index=False, header=False)
start += timedelta(1)

while start <= end:
    driver = webdriver.Firefox(profile)
    data = scrape("https://www.wunderground.com/history/daily/us/wa/seatac/KSEA/date/"+start.isoformat())
    pd.DataFrame(data).to_csv("D:/1 CLIMATE/wunderground/oops.csv", index=False, header=False, mode = 'a')
    
    start += timedelta(1)
    



# for child in a:
#     print(child[0])

"""
stations:
KSEA: https://www.wunderground.com/history/daily/us/wa/seatac/KSEA/date/
KPDX: https://www.wunderground.com/history/daily/us/or/portland/KPDX/date/
KEAT: https://www.wunderground.com/history/daily/us/wa/wenatchee/KEAT/date/

CYVR: https://www.wunderground.com/history/daily/ca/richmond/CYVR/date/
KHQM: https://www.wunderground.com/history/daily/us/wa/hoquiam/KHQM/date/


KCLS: https://www.wunderground.com/history/daily/us/wa/chehalis/KCLS/date/
KRLD: https://www.wunderground.com/history/daily/us/wa/richland/KRLD/date/

"""

#-------------------------------------------------------
#-------------------------------------------------------
#-------------------------------------------------------
#-------------------------------------------------------


"""
data format in page:
    --summary-table--
    
    table 
        thead
            tr --new row (6 of these)
                th --title of section (temp, precip, dew point...)
                    td --'actual'
                    td --'historic'
                    td --'record'
                    td --'polygon' (don't need)
        tbody
            tr
                th --title of row(high, low, average, precip last 24...)
                    td (actual for that day)
                    td (historic average over the years)
                    td (record of all time)
    
    
    
    --observation-table--
    table
        thead
            tr  (only 1)
                th (10 of these)
        tbody
            tr  (24 of these)
                td (10 to correlate to )
        
"""


#time.sleep(3)
