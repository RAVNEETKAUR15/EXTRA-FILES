'''DATE - 31 AUGUST 2022
TITLE - PROPAGATION MATRIX METHOD
NAME - RAVNEET KAUR
ROLL NO -2020PHY1064
ASSIGNMENT - 4'''
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
    p = np.matmul(p_free,p_step)
    A_r_out = 1/p[0, 0]
    A_l_in = p[1, 0]*A_r_out
    P_T = abs(A_r_out)**2
    P_R = abs(A_l_in)**2
    Total = P_T + P_R
    ref_prob = P_R
    
    tran_prob = P_T*(K2/K1)
    total_prob = ref_prob + tran_prob
    return p_free,p_step,p,A_r_out,A_l_in,P_T,P_R,ref_prob,tran_prob,total_prob,Total  
V = 0
V_0 = 1*e
E = 1.01*V_0
E2 = 2.02*V_0
E_array = np.linspace(E,E2,100)
m = 9.1*10**-31
x = 20e-10 
K1 = K(E,V,m)
K2 = K(E,V_0,m)
I = func(x,K1,K2) 
T_p = []
R_p = []
T_d = []
Total = []
R_d = []
T_P = []
print("The value of P free :",I[0])
print()
print("The value of P step :",I[1])
print()
print("The value of P  :",I[2])
print()
print("The value of Tranmission Coefficient :",I[3])
print()
print("The value of Reflection Coefficient :",I[4])
print()
print("The value of Tranmission Probability :",I[5])
print()
print("The value of Reflection Probability :",I[6])
print()
print("The value of Transmission Probability :", I[7])
print()
print("The value of Reflection Probability :", I[8])
for E in E_array:
    A = func(x,K1,K2)
    T_p.append(A[5])
    R_p.append(A[6])
    T_d.append(A[7])
    R_d.append(A[8])
    T_P.append(A[9])
    Total.append(A[10])
func(x,K1,K2)
#PLotting
plt.plot(T_p,E_array,label = "Transmission Probability")
plt.plot(R_p,E_array,label = "Reflection Probability")
plt.plot(Total,E_array,label = "Total Probability")
plt.legend()
plt.xlabel("Probabilities")
plt.ylabel("Energy")
plt.title("ANALYTICAL METHOD")
plt.grid()
plt.show()
plt.scatter(T_d,E_array,label = "Transmission Probability")
plt.scatter(R_d,E_array,label = "Reflection Probability")
plt.scatter(T_P,E_array,label = "Total Probability")
plt.plot(T_d,E_array)
plt.plot(R_d,E_array)

plt.plot(T_P,E_array)
plt.legend()
plt.title("NUMERICAL METHOD")
plt.xlabel("Probabilities")
plt.ylabel("Energy")
plt.grid()
plt.show()
