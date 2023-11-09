#AEP 5380 Homework 5
#Three or more body
#Ran on MacOS with Python 3.9.6
# Eric Xue (ex33) Sep-29-2023'

import numpy as np 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

#---RHS for Harmonic Osillation---
def rhs_HO(t,state):
    return ( np.array( [state[1], -state[0] ]) )

#---RHS for N-Body problem---
def rhs_NBody(t,state,N):
    #Need to define a new state to return because the states are coupled.
    state_dot=np.zeros([len(state)]) 
    x0=0
    y0=1*N
    vx0=2*N
    vy0=3*N
    for i in range(N):
        state_dot[i+x0]=state[i+vx0]
        #print(state[i+x0])
        state_dot[i+y0]=state[i+vy0]
        for j in range (N):
            if not(i==j):
                dx=state[j+x0]-state[i+x0]
                dy=state[j+y0]-state[i+y0]
                dij=math.sqrt(dx*dx+dy*dy)
                state_dot[i+vx0]=state_dot[i+vx0]+G*mass[j]*(dx/(dij*dij*dij))
                state_dot[i+vy0]=state_dot[i+vy0]+G*mass[j]*(dy/(dij*dij*dij))
    return state_dot


#---Earth Intial Conditions---
E_x0=0
E_y0=0
#E_vx0=-12.593
E_vx0=0 #changing IC
E_vy0=0
mass_E=5.9742e24 #kg
R_E=6378 #km

#---Moon Intial Conditions---
M_x0=0
M_y0=3.84e8 
M_vx0=1020.0 
M_vy0=0
mass_M=0.0123*mass_E #kg
R_M=3476 #km
#---Space vehicle Intial Conditions---
SV_x0=1.10E7
SV_y0=1.00E7
#SV_vx0=7.17e3
SV_vx0=7.57e3
SV_vy0=6.003e2
mass_SV=2.0e4 #kg
R_SV=1.0 #km


#---Constants---
T_M=648 #Hours
dist_E_M=384000 #km
t_inital=0 #seconds
t_final=25*24*60*60 #seconds
G=6.674e-11 #Nm^2/kg^2
N=3

#---Form mass and radius vectors---
#Should have N masses and radius. Follows in order of states, I.E. mass[i] should refer to body i.
mass=np.array([mass_E,mass_M,mass_SV])
radius=np.array([R_E,R_M,R_SV])

#---Forms intial state vector---
State0_NBodies=[E_x0,M_x0,SV_x0,E_y0,M_y0,SV_y0,E_vx0,M_vx0,SV_vx0,E_vy0,M_vy0,SV_vy0]
State0_HO=[1,0]
sol_NBodies = solve_ivp( rhs_NBody, [t_inital, t_final], State0_NBodies, args=[N],rtol=1.0e-6)

sol_HO = solve_ivp( rhs_HO, [0, 20], State0_HO,rtol=1.0e-6)
#print(sol)

#---Plots---
plt.figure(1)
plt.title('Earth Moon SV Trajectory (25 Days)')
plt.plot(sol_NBodies.y[0,:],sol_NBodies.y[3,:],label='Earth')
plt.plot(sol_NBodies.y[1,:],sol_NBodies.y[4,:],label='Moon')
plt.plot(sol_NBodies.y[2,:],sol_NBodies.y[5,:],label='Space Vehicle')

#plt.axis([3e8, 3.5e8, -3.5e8, -1e8]) #Uncomment to zoom into SV/Moon interaction
plt.legend()
plt.xlabel('X Position(m)')
plt.ylabel('Y Position(m)')
plt. gca(). set_aspect('equal', adjustable='box')

plt.figure(2)
plt.title('Harmonic Oscillator Test')
plt.plot(sol_HO.t,sol_HO.y[0,:],'*',label='Solve_IVP Solution')
plt.plot(sol_HO.t,np.cos(sol_HO.t),label='cos(wt)')
plt.xlabel('Time(s)')
plt.ylabel('Position (m)')
plt.legend()