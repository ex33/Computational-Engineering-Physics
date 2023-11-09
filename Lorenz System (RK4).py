#AEP 5380 Homework 4
#Runge Kutta Method 4th Order for Lorenz system
#Ran on MacOS with Python 3.9.6
# Eric Xue (ex33) Sep-20-2023

import numpy as np 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import numba


#--------- rk4 ODE solver -----------------------
# rhs() = function to calculate the right hand sides
# y[nt,neqn] = np array to get solution
# t[nt] = array to get time
# tend = final time t
# nt, neqns = number of times steps and equations
#

@numba.jit(nopython=True)

def rk4( rhs, y, t,tend, nt, neqns):
    #Python: y is row vector
    #Matlab: z_dot is column vector
    h = tend/(nt-1)
    hh = 0.5*h
    k1 = np.zeros( neqns )
    k2= np.zeros(neqns)
    k3= np.zeros(neqns)
    k4= np.zeros(neqns)
    t[0] = 0.0
    for it in range(0,nt-1):
        rhs( y[it,:], t[it], k1) #k1=returns f(tn,yn)
        rhs( y[it,:] + hh*k1, t[it]+hh, k2 ) #returns k2=f(tn+h/2, yn+k1/2)
        rhs( y[it,:] + hh*k2, t[it]+hh, k3 ) # returns k3=f(tn+h/2, yn+k2/2)
        rhs( y[it,:] + h*k3, t[it]+h, k4 ) #returns k4=f(tn+h, yn+k3)
        y[it+1,:] = y[it,:] + h*( k1 + 2*(k2 + k3) + k4)/6.0 
        t[it+1] = (it+1)*h
    #No need for output, rk4 just updates the given y


#-------------Harmonic Oscillator RHS for Testing----------------
@numba.jit(nopython=True)
def HO_RHS(y,t,k):
    #K is just the state_dot equations evaluated with different positions. 
    #If dy/dt=f(x,y), then k can be replaced with f since k=f(x1,y1,t1...)
    #States are y=y[0] and y_dot=y[1]
    #State_dot is then y_dot=k[0]=y[1] and y_doubledot=k[1]
    #E.O.M is y_doubledot=-w^2*y 
    # Thus y_doubledot=k[1]=-w^2*y[0] 
    w=4
    k[0]=y[1]
    k[1]=-w*w*y[0] 

#--------------Harmonic Oscillator Test case------------------

#Intial conditions
neqns=2
nt=500
tend=20.0
yy=np.zeros((nt,neqns))
t=np.zeros(nt)
yy[0,:] =[4.0, 0.0] #intial values
#rk4 call
rk4(HO_RHS,yy,t,tend,nt,neqns)
#Unpack states from rk4 for easier reading
y=yy[:,0]
y_dot=yy[:,1]

# #---Plots Data---
# #Comparing RK4 with actual result
# plt.figure(1)
# plt.plot(t,y,'*',label='RK4')
# plt.plot(t,4*np.cos(4*t),label='4cos(4wt)')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('y')
# plt.title('Harmonic Oscillator: RK4 Results vs Actual')

# #y vs y_dot
# plt.figure(2)
# plt.plot(y_dot,y)
# plt.legend()
# plt.xlabel('y_dot')
# plt.ylabel('y')
# plt.title('Harmonic Oscillator: y_dot vs y')

#-------------Lorenz System RHS----------------
@numba.jit(nopython=True)
def Lorenz_RHS(State,t,k):
    #K is just the state_dot equations evaluated with different positions. 
    #If dy/dt=f(x,y), then k can be replaced with f since k=f(x1,y1,t1...)
    #States are y=y[0] and y_dot=y[1]
    #State_dot is then y_dot=k[0]=y[1] and y_doubledot=k[1]
    #E.O.M is y_doubledot=-w^2*y 
    # Thus y_doubledot=k[1]=-w^2*y[0] 
    #State=[x y z] State_dot=k=[dx dy dz]
    #E.O.M: dx=-y-z dy=x+ay dz=b+z(x-c)
    a=0.2
    b=0.2
    c=5.7
    k[0]=-State[1]-State[2]
    k[1]=State[0]+a*State[1]
    k[2]=b+State[2]*(State[0]-c)

neqns2=3
nt2=10000
tend2=200.0
State2=np.zeros((nt2,neqns2))
t2=np.zeros(nt2)
x0=0
y0=-6.78
z0=0.002
State2[0,:] =[x0, y0, z0] #intial values
#rk4 call
rk4(Lorenz_RHS,State2,t2,tend2,nt2,neqns2)
#Unpack states from rk4 for easier reading
x=State2[:,0]
y=State2[:,1]
z=State2[:,2]

#Plots Data
plt.figure(1)
plt.plot(t2,x,label='x')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Rossler System: x vs t')

plt.figure(2)
plt.plot(t2,z,label='z')
plt.xlabel('Time')
plt.ylabel('z')
plt.title('Rossler System: z vs t')

plt.figure(3)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rossler System: y vs x')

plt.figure(4)
plt.plot(x,z)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Rossler System: z vs x')


ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.scatter(x, y, z)
ax.plot3D(x, y, z)
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.set_zlabel('Z', fontsize=10)