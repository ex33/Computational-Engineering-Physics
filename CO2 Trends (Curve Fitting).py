#AEP 5380 Homework 8
#Least Squares Curve Fitting
#Ran on MacOS with Python 3.9.6
# Eric Xue (ex33) 10-28-2023'

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, inv
import math

def func(t,i):
        if i==0:
            return t*t
        elif i==1:
          return t
        elif i==2:
            return 1
        elif i==3:
            return math.sin(2*math.pi*t)
        elif i==4:
            return math.cos(2*math.pi*t)
        elif i==5:
            return math.sin(4*math.pi*t)
        elif i==6:
            return math.cos(4*math.pi*t)

def linreg(t_span,sigma_2,m):
    b=np.zeros((m),dtype=float)
    A=np.zeros((m,m),dtype=float)
    error_a=np.zeros((m),dtype=float)
    for i in range(N):
        for l in range(m):
            for k in np.arange(l,m,1):
                A[l,k]=A[l,k]+func(t_span[i],l)*func(t_span[i],k)/(sigma_2[i])
                A[k,l]=A[l,k]
            b[l]=b[l]+co2[i]*func(t_span[i],l)/(sigma_2[i])

    a=solve( A,b )
    A_inv=inv(A)
    for k in range(m):
        print(math.sqrt(A_inv[k,k]))
        error_a[k]=math.sqrt(A_inv[k,k])
        
    return a,error_a


def best_fit(a,t_span,m):
    best_fit=np.zeros((N),dtype=float)

    for l in range(m):
        for i in range(N):
            best_fit[i]=best_fit[i]+a[l]*func(t_span[i],l)
    return best_fit


#--- read CO2 data file and separate into t,CO2 data
# data= BRW year month co2
data = np.loadtxt('co2_brw_surface-flask_1_ccgg_month.txt',skiprows=53, usecols=(1,2,3), dtype=float)
time = data[:,0] + data[:,1]/12 # year + month
co2 = data[:,2]
N=len(co2)
# print statistics and plot to verify
print('t,co2 len=',len(time),len(co2) )
print('t,co2 data range= ', min(time), max(time), min(co2), max(co2) )
plt.figure(1) # plot data vs. t
plt.plot(time, co2, 'k-' )
plt.xlabel( 'time (year)')
plt.ylabel( 'CO2 (in ppm)' )
plt.title ('original data')


t_span=time-1971
error=0.004 #0.4 percent
sigma_2=(co2*error)*(co2*error)
m=7

a,error_a=linreg(t_span,sigma_2,m)
best_fit_season=best_fit(a,t_span,m)

plt.figure(2)
plt.plot(time[-100:],best_fit_season[-100:],label='Fitted Data' )
plt.plot(time[-100:], co2[-100:], 'k-',label='Original Data' )
plt.xlabel( 'time (year)')
plt.ylabel( 'CO2 (in ppm)' )
plt.title ('Fitted Data Vs Original data')
plt.legend()
        
chi_squared=np.sum(((co2-best_fit_season)*(co2-best_fit_season))/sigma_2)

reduced_chi_squared=chi_squared/(N-m)

best_fit_no_season=a[0]*(t_span*t_span)+a[1]*t_span+a[2]

plt.figure(3)
plt.plot(time,co2,label='Original Data')
plt.plot(time,best_fit_no_season,label='No Season Variations')
plt.xlabel( 'time (year)')
plt.ylabel( 'CO2 (in ppm)' )
plt.title ('Fitted Data w/o Seasonal Variations vs Original Data')
plt.legend()

plt.figure(4)
plt.plot(time,best_fit_season-co2)
plt.xlabel( 'time (year)')
plt.ylabel( 'CO2 (in ppm)' )
plt.title ('Residuals')

a2,error_a2=linreg(t_span,sigma_2,5)
best_fit_no_harmonic=best_fit(a2,t_span,5)
chi_squared_no_harmonic=np.sum(((co2-best_fit_no_harmonic)*(co2-best_fit_no_harmonic))/sigma_2)

reduced_chi_squared_no_harmonic=chi_squared_no_harmonic/(N-5)

plt.figure(5)
plt.plot(time,co2,label='Original Data')
plt.plot(time,best_fit_no_harmonic,label='No Harmonic Fit')
plt.xlabel( 'time (year)')
plt.ylabel( 'CO2 (in ppm)' )
plt.title ('No Harmonic fit vs Original Data')
plt.legend()

