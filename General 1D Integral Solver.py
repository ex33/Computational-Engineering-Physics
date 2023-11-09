#AEP 5380 Homework 2
#Calculate integrals using Trapezoid and Simpsons method
#Ran on MacOS with Python 3.9.6
# Eric Xue (ex33) Sep-5-2023

import numpy as np
import matplotlib.pyplot as plt
import math
from tabulate import tabulate

def func(x):
    #Need to set first if condition since x=0 will return #/0 which is a divide by zero error 
    if abs(x)<10**-10:
        return 0
    else:
        return ((x**4)*math.exp(x))/((math.exp(x)-1)**2) #this is the function within the integral

#for part 1 of homework
def integral_trapezoid(y_lower,y_upper,n_min):
    # y_lower is the lower bound (0 for Homework 2)
    # y_upper is the uppber bound (user choice)
    #chosen as minimum number of areas needed to be split up
    n=1 #want to start with two areas, but n=n*2 is inside while loop so start at n=1
    S=0.5*(func(y_lower)+func(y_upper)) #intial average between the function evaluated at the bounds
    I_new=(y_upper-y_lower)*S #Calculates intial trapizoid area
    I_old=0 #if loop doesn't start, then I_new is good enough 
    tol=10**-10 #sets tolerance
    while n <n_min: #need to both hit a minimum number of intervals and a minimum tolerance to break
        # remove abs(I_new-I_old)>abs(tol) contraint to interate by n
        n=2*n #multiply intervals by 2
        delta_x=(y_upper-y_lower)/n #sets thickness of the trapezoid
        I_old=I_new #sets the previous iteration to be compared
        for i in np.arange(1,n,2):  #no n-1 bc starting at 1 and indexing by 2 gives odd numbers from 1 to n. Doing n-1 skips n-1.
            S=S+func(y_lower+i*delta_x) #calculates the average value of the function inbetween each interval and sums them up
        I_new=delta_x*S #Since S is a sum, this is the same as multiplying average values by this thickness to get the sum of area
    return I_new
    

def integral_simpson(y_lower,y_upper,n_min):
    # y_lower is the lower bound (0 for Homework 2)
    # y_upper is the uppber bound (user choice)
    n=2 
    S1=func(y_lower)+func(y_upper) #takes the sum of the function evaluated at the bounds
    S2=0 #set to be zero at the start
    S4=func((y_lower+y_upper)/2) #function evaluated right inbetween the boundaries
    I_New=0.5*(y_upper-y_lower)*(S1+4*S4)/3 #calculates intial integral guess
    I_old=0 #set to be zero so while loops starts
    
    tol=10**-10 #sets tolerance

    while n<n_min:
        #removed abs(I_New-I_old)>abs(tol) for now so loops based on n_min
        
        temp=0 #holds the summation value in the for loop
        n=2*n #increases number of intervals
        deltax=(y_upper-y_lower)/n #calculates thickness
        S2=S2+S4 #updates S2
        I_old=I_New #updates old integral to compare for while loop condition
        for i in list(range(1,n,2)):
            temp=temp+func(y_lower+i*deltax) #this is summing the areas
        S4=temp #update S4
        I_New=deltax*(S1+2*S2+4*S4)/3 #updates new integral
        
    return I_New #return integral value once while loop is broken

#Part 1        
n=[2,4,8 ,16, 32, 64, 128 ,256 ,512 ,1024, 2048, 4096] #number of intervals requested
j=0 #index
trapezoid1=np.empty(12,float)
simpson1=np.empty(12,float)
for i in n: #loops through the intervals
    simpson1[j]=integral_simpson(0,1,i)
    trapezoid1[j]=integral_trapezoid(0,1,i)
    j=j+1
#Tabulates data
trapezoid_list=[round(elem,7) for elem in trapezoid1.tolist()] #Rounds each element so table isn't super long
simpson_list=[round(elem,7) for elem in simpson1.tolist()]
#Creates an empty list so each index can be replaced with a dictionary
list_of_dict=list(range(len(n)))
for h in range(len(n)): #updates empty list with the dictionary as follows
    list_of_dict[h]={"N":n[h],"Trapezoid Method":trapezoid_list[h],"Simpson Method":simpson_list[h]}
    #this is just a automated way to use the tabulate package, other method is just populate it by hand

print(tabulate(list_of_dict,headers='keys')) #prints table

#Part 2
N=256 #Should be enough to give ~4-5 sig fig
k=0 #index

y_values=np.arange(0.01,10,(10-0.01)/200) #sets range of y values to iterate over
trapezoid2=np.empty(len(y_values+1),float) #creates empty vector to fill
simpson2=np.empty(len(y_values+1),float) #creates empty vector to fill
for z in y_values: 
    simpson2[k]=integral_simpson(0,z,N) #evaluate integral at each y value
    trapezoid2[k]=integral_trapezoid(0,z,N) #evaluates integral at each y value
    k=k+1 #moves on to the next index



#Plots both method
plt.figure(1)
plt.plot(y_values, trapezoid2,'-', label='trapezoid method' ) #args are x-axis, y-axis, symbol/color, label
plt.legend()
plt.xlabel('y')
plt.ylabel('Integral values')
plt.figure(2)
plt.plot(y_values, simpson2,'-', label='simpson method' ) #args are x-axis, y-axis, symbol/color, label
plt.legend()
plt.xlabel('y')
plt.ylabel('Integral values')