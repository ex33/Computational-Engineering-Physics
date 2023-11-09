#AEP 5380 Homework 9
#Monte Carlo
#Ran on MacOS with Python 3.9.6
# Eric Xue (ex33) Nov-6-2023'

import random as rng
import numpy as np
import matplotlib.pyplot as plt
import numba
import math
rng.seed() # unspecified seed

#10 is good



#---Part 1---
# N=100000
# j=np.zeros([N])
# for i in range(N):
#     j[i] = rng.random()


# x=np.random.rand(1,10000)
# y=np.random.rand(1,10000)
# plt.figure(1)
# plt.hist(j,bins=100)
# plt.xlabel('Bins')
# plt.ylabel('# per bin')
# plt.title('Histogram of 100000 random numbers')

# plt.figure(2)
# plt.scatter(x,y,s=1)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Scatter of 10,000 points')

#---Part 2---

@numba.jit(nopython=True)
def dist(x1,y1,x2,y2):
    return abs(x1-x2)+abs(y1-y2)

@numba.jit(nopython=True)
def is_in(Amino_info,x,y):
    #Checks if (x,y) is a position of an amino acid
    for i in range(N_L):
        if Amino_info[i,0]==x and Amino_info[i,1]==y:
            return True
    return False

@numba.jit(nopython=True)
def move_amino(Amino_info,choice,E0):
    valid_amino=False
    temp_amino=np.copy(Amino_info)
    while not(valid_amino):
        rand_amino=rng.randint(0,44)
        #Unpack index, this is unnecessary but done for clarity
        x=Amino_info[rand_amino,0]
        y=Amino_info[rand_amino,1]
        #print(x,y)
        choice_index=rng.randint(0,3) #Randomly selects a movement, only 4 options
        movement=choice[choice_index,:]
        #Unpack index, again this is unnecessary but done for clarity
        m_x=movement[0]
        m_y=movement[1]

        
        if not(is_in(Amino_info,x+m_x,y+m_y)): #This means the grid it moves to is empty
            
            if rand_amino==0:
                #If amino 1, only need to check amino 2
                if dist(x+m_x,y+m_y,Amino_info[rand_amino+1,0],Amino_info[rand_amino+1,1])==1:
                    valid_amino=True
            elif rand_amino==N_L-1:
                #If amino 45, only need to check amino 44
                if dist(x+m_x,y+m_y,Amino_info[rand_amino-1,0],Amino_info[rand_amino-1,1])==1:
                    valid_amino=True
            else:
                if (dist(x+m_x,y+m_y,Amino_info[rand_amino+1,0],Amino_info[rand_amino+1,1])==1) and (dist(x+m_x,y+m_y,Amino_info[rand_amino-1,0],Amino_info[rand_amino-1,1])==1):
                    valid_amino=True
    temp_amino[rand_amino,0]=temp_amino[rand_amino,0]+m_x #Sets new position to previous grid
    temp_amino[rand_amino,1]=temp_amino[rand_amino,1]+m_y
    E_new=calc_energy(temp_amino)
    delta_E=E_new-E0
    if delta_E<=0:
        Amino_info[rand_amino,0]=Amino_info[rand_amino,0]+m_x 
        Amino_info[rand_amino,1]=Amino_info[rand_amino,1]+m_y
        end_dist=dist(Amino_info[0,0],Amino_info[0,1],Amino_info[N_L-1,0],Amino_info[N_L-1,1])
        return E_new,end_dist
    else:
        p=math.exp(-delta_E/(kBT))
        if p>rng.random():
            Amino_info[rand_amino,0]=Amino_info[rand_amino,0]+m_x 
            Amino_info[rand_amino,1]=Amino_info[rand_amino,1]+m_y
            end_dist=dist(Amino_info[0,0],Amino_info[0,1],Amino_info[N_L-1,0],Amino_info[N_L-1,1])
            return E_new,end_dist
        else:
            end_dist=dist(Amino_info[0,0],Amino_info[0,1],Amino_info[N_L-1,0],Amino_info[N_L-1,1])
            return E0,end_dist

@numba.jit(nopython=True)
def calc_energy(Amino_info):  # calc_energy(Grid):
    E_total=0
    for i in range(N_L):
        for j in range(i,N_L): #Ensures no double counting
            if j!=(i+1): #If this is false, it means its not a covalent bond
                if (dist(Amino_info[i,0],Amino_info[i,1],Amino_info[j,0],Amino_info[j,1])==1):
                    E_total=E_total+E[int(Amino_info[i,2]),int(Amino_info[j,2])]
    #E_total=E_total*0.5
    return E_total

kBT=1.0
Emin=-7.0
Emax=-2.0
E=np.zeros([20,20])
N_L=45
for l in range(20):
    for k in np.arange(l,20,1):
        E[l,k]=(Emax-Emin)*rng.random()+Emin #Random # between Emin and Emax
        E[k,l]=E[l,k]

choice=np.array([[1,1],[-1,1],[1,-1],[-1,-1]])

Type=np.empty(N_L)

Grid=np.zeros([N_L,N_L]) #Decrease grid from 45x45 to 45x30 to reduce time
#emp_Grid=np.empty([N_L,30])
#Intialize Protein and and type i+1 refers to the amino number. Amino number starts from 1-45 such 0 represents empty
Amino_info=np.zeros([N_L,3])
for i in range(N_L):
    Amino_info[i,0]=i
    Amino_info[i,1]=15
    Amino_info[i,2]=rng.randint(0,19)

Monte_step=10000000  
Energy=np.empty(Monte_step+1)
end_to_end=np.empty(Monte_step+1)


#plot_amino(Amino_info)
#move_amino(Amino_info,choice)
j=0
E0=calc_energy(Amino_info)
for i in np.arange(0,Monte_step+1,1):
    E_new,dist_temp=move_amino(Amino_info,choice,E0)
    #if i%20000==0:
    E0=E_new
    Energy[i]=E_new
    end_to_end[i]=dist_temp
    if i==1e4:
        plt.figure(1)
        plt.plot(Amino_info[:,0],Amino_info[:,1],'o-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Amino at timestep=1e4')
    if i==1e5:
        plt.figure(2)
        plt.plot(Amino_info[:,0],Amino_info[:,1],'o-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Amino at timestep=1e5')
    if i==1e6:
        plt.figure(3)
        plt.plot(Amino_info[:,0],Amino_info[:,1],'o-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Amino at timestep=1e6')
    if i==1e7:
        plt.figure(4)
        plt.plot(Amino_info[:,0],Amino_info[:,1],'o-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Amino at timestep=1e7')
time_step=np.arange(0,Monte_step+1,1)
plt.figure(5)
plt.plot(Energy)
plt.xlabel('Time Step')
plt.ylabel('Energy')
plt.title('Energy Vs TimeStep')
plt.figure(6)

plt.figure(6)
plt.plot(end_to_end)
plt.xlabel('Time Step')
plt.ylabel('end_to_end dist')
plt.title('End to End Distance Vs TimeStep')

#t_span=np.arange(0,500,1)
Energy_500=np.zeros([2,500])
dist_500=np.zeros([2,500])
for i in range(500):
    Energy_500[1,i]=Energy[i*20000]
    Energy_500[0,i]=time_step[i*20000]
    dist_500[1,i]=end_to_end[i*20000]
    dist_500[0,i]=time_step[i*20000]

    

plt.figure(7)
plt.grid(True)
plt.plot(Energy_500[0,:],Energy_500[1,:])
plt.xlabel('Time Step')
plt.ylabel('Energy')
plt.title('Energy Vs TimeStep (500 Points)')

plt.figure(8)
plt.grid(True)
plt.plot(dist_500[0,:],dist_500[1,:])
plt.xlabel('Time Step')
plt.ylabel('end_to_end dist')
plt.title('End to End Distance Vs TimeStep (500 Points)')






