# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 12:02:06 2021

@author: Cryss
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

def calib(Tdew, Tmax, Tmin, Tavg):
    Tday = 0.45*(Tmax - Tavg) + Tavg
    k = (Tmin - Tdew) / (Tday - Tmin)
    return k

temps = array([[47.79,87,59,72.83],       #in col order, Tdew, Tmax, Tmin, Tavg
               [54.75,81,63,71.83],
               [53.25,82,60,70.63],
               [54.79,84,60,71.38],
               [54.96,83,60,72.04],
               [50.08,79,53,66.13],
               [47.54,76,53,65.13]
    
    ])

calval = zeros(temps.shape[0])

for i in range(len(temps)):
    calval[i] = calib(temps[i,0],temps[i,1],temps[i,2],temps[i,3])