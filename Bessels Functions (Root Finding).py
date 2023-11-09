import numpy as np 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
from scipy.special import *

#Part 1
x1=np.arange(0.0,25.0,0.5)
fjx0=j0(x1)
fjx1=j1(x1)
fjx2=jn(2,x1)

x2=np.arange(0.75,20.0,0.5)
fyx0=y0(x2)
fyx1=y1(x2)
fyx2=yn(2,x2)

plt.figure(1)
plt.plot(x1, fjx0,'-', label='fjx0') 
plt.plot(x1, fjx1,'-', label='fjx1' ) 
plt.plot(x1, fjx2,'-', label='fjx2' ) 
plt.legend()
plt.xlabel('x')
plt.ylabel('Bessels First Function')

plt.figure(2)
plt.plot(x2, fyx0,'-', label='fyx0') 
plt.plot(x2, fyx1,'-', label='fyx1' ) 
plt.plot(x2, fyx2,'-', label='fyx2' ) 
plt.legend()
plt.xlabel('x')
plt.ylabel('Bessels Second Function')

#Part 2
# def func(x):
#     return jn(2,x)*yn(2,x)-j0(x)*y0(x)

# def bisection(x1):
#     #Need to find 5 smallest positive values for x such that J0*Y0=J2*Y2
#     #To frame this as a root finding problem, need to find when J0*Y0-J2Y2=0, which is just
#     #the roots of that function f=J0*Y0-J2Y2
#     roots=np.empty(5) 
#     i=1 #will be used to increment the upper bounds throughout the function
#     j=0 #will be used to index roots
#     k=np.zeros(5) #used to track efficency
#     while j<5: #while we have found less than 5 roots
#         x2=x1+i*0.1 #increment upper bound
#         if func(x1)*(func(x2))<0: #if function evaluated at bounds are opposite sign
#             x3=0.5*(x1+x2) #evaluate function in middle
#             while abs(func(x3))>0.0000001: #sets tolerance
#                 if func(x1)*func(x3)>0: # if x3 is same sign as x1, then it is new lower bound
#                     x1=x3
#                     k[j]=k[j]+1
#                 elif func(x2)*func(x3)>0:#if x3 is same sign as x2, then it is new upper bound
#                     x2=x3
#                     k[j]=k[j]+1
#                 x3=0.5*(x1+x2) #find new middle
#             roots[j]=x3 #after hitting tolerance, set it as a root
#             x1=x2 #sets lower bound as the previous upper bound, now can find next root
#             j=j+1 #index of next loop
#             i=1 #restarts i to make sure increment isn't too big that we skip a root      
#         else:
#             i=i+1 #if the function isn't opposite sign, we can't find a root so increment upper bound
#     #plt.plot(,k,label='bisection')
#     #p1=plt.bar([1 ,2,3,4,5], k, color ='maroon',width = 0.4)
#     return roots,k


# def false_position(x1):
#     #Need to find 5 smallest positive values for x such that J0*Y0=J2*Y2
#     #To frame this as a root finding problem, need to find when J0*Y0-J2Y2=0, which is just
#     #the roots of that function f=J0*Y0-J2Y2
#     roots=np.empty(5)
#     i=1 #will be used to increment the upper bounds throughout the function
#     j=0 #will be used to index roots
#     k=np.zeros(5)
#     while j<5:#while we have found less than 5 roots
#         x2=x1+i*0.1 #increment upper bound
#         if func(x1)*(func(x2))<0:#if function evaluated at bounds are opposite sign
#             x3=0.1 #just to get the loop started
#             while abs(func(x3))>0.0000001:
#                 x3=x1-func(x1)*(x2-x1)/(func(x2)-func(x1)) #false position method
#                 if func(x1)*func(x3)>0: #if x3 is same sign as x1, it is new lower
#                     x1=x3
#                     k[j]=k[j]+1
#                 elif func(x2)*func(x3)>0: #if x3 is same sign as x2, it is new upper
#                     x2=x3
#                     k[j]=k[j]+1
#             roots[j]=x3 #after reaching tol, x3 is a root
#             x1=x2 #sets the lower as the previous upper bound so we can find next root
#             j=j+1 #increment index of root
#             i=1 #sets i back to 1 so we don't skip over a root
#         else:
#             i=i+1 #if founction doesn't have opposite signs at bounds, there is no root so increment i
#     #plt.plot([1 ,2,3,4,5],k,label='false_position')
#     return roots,k
# x1=0.01
# s=bisection(x1)
# r=false_position(x1)

# #For Debugging
# print(s[0])
# print(r[0])
# print(s[1])
# print(r[1])

# #Bar graph
# p1=plt.bar([1 ,2,3,4,5], s[1], color ='blue',width = 0.4,label='Iteration for Bisection')
# p2=plt.bar([1 ,2,3,4,5], r[1], color ='blue', alpha=0.5, label='Iteration for False-Postion')
# plt.legend()





