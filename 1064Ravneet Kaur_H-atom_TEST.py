'''DATE - 27 JULY 2022
TITLE - GARRETTE METHOD
NAME - RAVNEET KAUR
ROLL NO -2020PHY1064
ASSIGNMENT - 1'''
#Importing Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar,e
import sys
import pandas as pd

#Function for Energy
def Energy(n,m,L): 
    return n*n*(np.pi**2)*hbar**2/(2*m*L**2)

#Function for Delta
def delta(En):
    if V>En: #We are checking wheather the value of 
             #the V is positive or not
        pass
    else:
        sys.exit("V should be greater than E")
        
    delta = hbar/np.sqrt(2*m*(V - En))
    return delta

#Function for the Length
def Ln(L,delta):
    return L+2*delta

#Now Defining a new function for iterations and Tolernace

def Function(En,n,m,V,L,Tol):
    #Now we will make empty lists in order to append the values 
    En_array = [En]
    Ln_array = []
    i_array = []
    r_array = []
    Del_array = []
    for i in range(1,1000):
        i_array.append(i) #all value of i
        Delta= delta(En) #new delta w.r.t new Energy
        Del_array.append(Delta) #appended all values of new Delta
        Lnew = Ln(L,Delta) #New Value of Ln
        Ln_array.append(Lnew) #appended new length in the empty list
        En = Energy(n,m,Lnew) #Energy
        En_array.append(En) #Appended the new Energy values in En_array
        
        del_E = abs(En_array[-1]-En_array[-2])
        #Here del_E means the absolute difference b/w the last 
        #second last last value of Energy
        
        r = del_E/En_array[-2]
        #ration of Delta by the second last value of Energy
        
        r_array.append(r)
        #appended the values of r in the empty list r_array
        
        '''Now we will check the condition for iteration with the 
        tolernace 
        '''
        if r<Tol:  #Here if the value of r is less than the tolerance 
        #then it would break and the loop will again have to run
            break
        else:  #else it will continue to run the loop since the value of r>Tol
            continue
        
    return En_array,i_array,r_array,i

n = float(input("Enter the value of n: "))  
Len = float(input("Enter the value of L: "))
L = Len*10**-10
#E = mc^2
r_m = float(input("Enter the value of r_m: "))
m = (r_m*10**6*1.6*10**-19)/(3*10**8)**2 #kg
Pot = float(input("Enter the value of V: "))
V = Pot*e
T = float(input("Enter the value of Tolerance: "))
Tol = T*10**-10
'''
V = 10*e #J
#E = mc^2
r_m = 0.5
m = (r_m*10**6*e)/c**2 #kg
L = 20e-10 #m
Tol = 10e-9'''

#Plotting for different values of n
for n in [1,2,3]:
   En = Energy(n,m,L)
   En_array,i_array,r_array,i = Function(En,n,m,V,L,Tol)
   plt.plot([0]+i_array,En_array,label = "E_{0}".format(n),marker="o")
   plt.scatter([0]+i_array,En_array)
   
plt.legend()
plt.title("Energy vs No. of Iterations")
plt.grid()
plt.xlabel("No. of Iterations I")
plt.ylabel("Energy Joules $E_n$")
plt.show()

print("Given Tolerance :", Tol)
print("Total number of iteration :", i)

#PLOTTING
plt.plot(i_array,r_array, c="red")
plt.scatter(i_array,r_array)
plt.title("Ratio of Energy vs No. of Iterations")
plt.grid()
plt.xlabel("No. of Iterations I")
plt.ylabel("Ratio = $\Delta$ E/ $E_{i}$")
plt.show()

#Table
data ={"Energy (New)":En_array}
print(pd.DataFrame(data))
'''
n = 3
L = 10
r_m = 0.5
V = 4
Tol = 0.1
'''