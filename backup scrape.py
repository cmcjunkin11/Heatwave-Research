# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 22:00:45 2021

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



#date to start scraping
start = date(2021,6,16)
end = date(2021,7,10)

profile = webdriver.FirefoxProfile()
profile.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0")

driver = webdriver.Firefox(profile)


url = "https://www.wunderground.com/history/daily/us/wa/seatac/KSEA/date/"+start.isoformat()
tm.sleep(10)
driver.get(url)

tm.sleep(15)
soup = BeautifulSoup(driver.page_source, 'html.parser')

obs = soup.find(class_='observation-table')

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

a = a.split(';')
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

pd.DataFrame(data).to_csv("D:/1 CLIMATE/wunderground/KEAT.csv", index=False, header=False, mode = 'a')
start += timedelta(1)
