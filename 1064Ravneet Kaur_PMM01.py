'''DATE - 24 AUGUST 2022
TITLE - PROPAGATION MATRIX METHOD 01
NAME - RAVNEET KAUR
ROLL NO -2020PHY1064
ASSIGNMENT - 03
https://www.lehman.edu/faculty/anchordoqui/400_5.pdf'''
#Importing Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar,e
def K(E,v,m):
    np.sqrt(2*m*(E - V))/hbar #Value of K
    return
def func(x,K1,K2):
    i = 0 + 1j
    K1 = np.sqrt(2*m*(E - V))/hbar
    K2 = np.sqrt(2*m*(E - V_0))/hbar
    p_free = np.array([[np.exp(-i*K1*x),0],[0,np.exp(i*K1*x)]])
    p_step = np.matrix([[1+(K2/K1),1-(K2/K1)],[1-(K2/K1),1+(K2/K1)]])/2
    p_step_1 = np.matrix([[1+(K1/K2),1-(K1/K2)],[1-(K1/K2),1+(K1/K2)]])/2
    p = np.matmul(p_step,p_free)
    p_new = np.matmul(p,p_step_1)
    A_r_out = 1/p_new[0, 0]
    A_l_in = p_new[1, 0]*A_r_out
    P_T = abs(A_r_out)**2
    P_R = abs(A_l_in)**2
    total = P_T + P_R
    return p_free,p_step,p,A_r_out,A_l_in,P_T,P_R,total
V = 0
V_0 = 1*e
E = 1.01*V_0
E2 = 3*V_0
E_array = np.linspace(E,E2,50)
m = 9.1*10**-31
x = 20e-10 
K1 = K(E,V,m)
K2 = K(E,V_0,m)
I = func(x,K1,K2) 
T_p = []
R_p = []
Total = []
for E in E_array:
    A = func(x,K1,K2)
    T_p.append(A[5])
    R_p.append(A[6])
    Total.append(A[7])
func(x,K1,K2)
#PLotting
plt.scatter(T_p,E_array,label = "Transmission Probability")
plt.scatter(R_p,E_array,label = "Reflection Probability")
plt.scatter(Total,E_array,label = "Total Probability")
plt.plot(T_p,E_array)
plt.plot(R_p,E_array)
plt.plot(Total,E_array)
plt.legend()
plt.grid()
plt.title("NUMERICAL METHOD")
plt.xlabel("Probabilities")
plt.ylabel("Energy")