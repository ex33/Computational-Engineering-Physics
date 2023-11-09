#AEP 5380 Homework 7
#Finite Difference Method
#Ran on MacOS with Python 3.9.6
# Eric Xue (ex33) 10-20-2023'
import numpy as np
import numba
import matplotlib.pyplot as plt
import math 
import cmath


def tridiag(a,b,c,d,i_t):
    #---Set up b vector---
    for i_x in range(nx):
        b[i_x]=complex(-2-(del_x*del_x)*V[i_x]/(h_bar_squared_over_m),w*h_bar/h_bar_squared_over_m)
    
    #---Set up d vector---
    for i_x in np.arange(1,nx-2,1): 
        #Need nx-2 bc nx is out of bounds and nx-1 is at the boundary
        #Start at 1 since 0 is a boundary
        temp_comp=complex(2+(del_x*del_x)*V[i_x]/(h_bar_squared_over_m),w*h_bar/h_bar_squared_over_m)*wave[i_x,i_t]
        d[i_x]=-wave[i_x-1,i_t]+temp_comp-wave[i_x+1,i_t]
    #Its probably fine to approximate d at the boundaries as zero anyways but I decided to come back
    #And hardcode them in just in case, cause a wave could be right at the boundaries.
    #This is outside the forloop so the loop isn't checking two if statements that only occur once
    d[0]=complex(2+(del_x*del_x)*V[0]/(h_bar_squared_over_m),w*h_bar/h_bar_squared_over_m)*wave[0,i_t]-wave[1,i_t]
    d[nx-1]=-wave[nx-2,i_t]+complex(2+(del_x*del_x)*V[nx-1]/(h_bar_squared_over_m),w*h_bar/h_bar_squared_over_m)*wave[nx-1,i_t]
    #---Solving upper triangular diagonal matrix---
    #Step 1
    c[0]=c[0]/b[0]
    d[0]=d[0]/b[0]

    #Step 2
    for i_x in np.arange(1,nx-1,1): #i_x=1,2,3...nx-1, only need nx-1 cause x excludes boundaries
        c[i_x]=c[i_x]/(b[i_x]-a[i_x]*c[i_x-1]) 
        d[i_x]=(d[i_x]-a[i_x]*d[i_x-1])/(b[i_x]-a[i_x]*c[i_x-1])

    #Step 3
    for i_x in np.arange(nx-2,0,-1):
        if i_x==nx-2:
            wave[i_x,i_t+1]=d[i_x]
        else:
            wave[i_x,i_t+1]=d[i_x]-c[i_x]*wave[i_x+1,i_t+1]





#---Constants---
h_bar=6.5821e-16 #eV-sec
h_bar_squared_over_m =3.801 #eV- ̊A2
L=500 #Angstroms
k0=1  #Angstrom− Avg wavenumber 2ppi/lamda
x0=0.5*L
wx=5 #Angstroms
s=10 #Angstroms
V1= 2.0 #eV
t_final=3e-14 #seconds  
del_x=0.1 
del_t=0.01e-14
w=2*del_x*del_x/del_t
x=np.arange(0,L,del_x)
t=np.arange(0,t_final,del_t)
nx=len(x)
nt=len(t)

#---Set up V Vector---
#Intalize V vector
V=np.zeros((nx),dtype=np.float32 )

#Calculate V
for i_x in range(nx):
    if x[i_x]>100:
        V[i_x]=V1*(0.75-math.cos((x[i_x]-x0)/wx))
    else:
        V[i_x]=0


#---Set up wave matrix---
#Initalize wave vector
wave = np.empty((nx,nt),dtype=complex) #Intalize wave function array

#Set up boundaries
wave[nx-1,:]=0
wave[0,:]=0

#Calculate wave inbetween boundary
for i_x in np.arange(1,nx-2,1):
    #From 1 to nx-2 so boundaries stays 0
    w_real=(-((x[i_x]-0.3*L)/s)*((x[i_x]-0.3*L)/s))
    w_img=(x[i_x]*k0)
    #wave[i_x,0]=complex(w_real,w_img)
    wave[i_x,0]=cmath.exp(complex(w_real,w_img))
    #complex(math.exp(-((x[i_x]-0.3*L)/s)*((x[i_x]-0.3*L)/s)), math.exp(x[i_x]*k0))

#---Calls function to solve tri-drag matrix---
for i_t in range(nt-1):
    a=np.ones((nx),complex) #a coefficents are all ones according to finite difference form
    b=np.empty((nx),complex)
    c=np.ones((nx),complex) #c coefficents are all ones according to finite difference form (prior to transforming into upper diag)
    d=np.zeros((nx),complex )
    tridiag(a,b,c,d,i_t)


#---Plots---
plt.figure(1)
plt.plot(x,V)
plt.xlabel('x (Angstroms)')
plt.ylabel('V (eV)')
plt.title('V vs x')
plt.figure(2)
plt.plot(x,np.real(wave[:,0]))
plt.xlabel('x (Angstroms)')
plt.ylabel('Wave.Real ')
plt.title('Wave.Real vs x at t=0')
plt.figure(3)
plt.plot(x,np.imag(wave[:,0]))
plt.xlabel('x (Angstroms)')
plt.ylabel('Wave.Imag ')
plt.title('Wave.Imag vs x at t=0')
plt.figure(4)
plt.plot(x,np.abs(wave[:,0])*np.abs(wave[:,0]))
plt.xlabel('x (Angstroms)')
plt.ylabel('Wave.Mag ')
plt.title('Wave.Mag vs x at t=0')


i_t1=round(0.5e-14/del_t)
i_t2=round(1.0e-14/del_t)
i_t3=round(1.5e-14/del_t)
i_t4=round(2.0e-14/del_t)
i_t5=round(2.5e-14/del_t)
plt.figure(5)
plt.plot(x,np.abs(wave[:,i_t1])*np.abs(wave[:,i_t1]))
plt.xlabel('x (Angstroms)')
plt.ylabel('Wave.Mag ')
plt.title('Wave.Mag vs x at t=0.5e-14')

plt.figure(6)
plt.plot(x,np.abs(wave[:,i_t2])*np.abs(wave[:,i_t2]))
plt.xlabel('x (Angstroms)')
plt.ylabel('Wave.Mag ')
plt.title('Wave.Mag vs x at t=1.0e-14')

plt.figure(7)
plt.plot(x,np.abs(wave[:,i_t3])*np.abs(wave[:,i_t3]))
plt.xlabel('x (Angstroms)')
plt.ylabel('Wave.Mag ')
plt.title('Wave.Mag vs x at t=1.5e-14')

plt.figure(8)
plt.plot(x,np.abs(wave[:,i_t4])*np.abs(wave[:,i_t4]))
plt.xlabel('x (Angstroms)')
plt.ylabel('Wave.Mag ')
plt.title('Wave.Mag vs x at t=2.0e-14')

plt.figure(9)
plt.plot(x,np.abs(wave[:,i_t5])*np.abs(wave[:,i_t5]))
plt.xlabel('x (Angstroms)')
plt.ylabel('Wave.Mag ')
plt.title('Wave.Mag vs x at t=2.5e-14')
