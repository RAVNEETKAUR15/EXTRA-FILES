'''RAVNEET KAUR
2020PHY1064'''

from math import sin, pi,cos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons

def psi(t): #Calculation of wave function as a function of time
	global x
	v = alpha*k0
	w = (a**4 + ((alpha*t)**2)/4)
	wComplex = a**2 + 1j*alpha*t/4
	omega = (k0**2)*alpha/2
	probCompl = np.exp(1j*(k0*x - t*omega))*((pi/wComplex)**0.5)*np.exp(-((x - v*t)**2)/(4*wComplex))
	return probCompl 

#Functions for Plots
def plot(A,B,C,D,E,F,G,H):
     fig,ax = plt.subplots(1, 2, squeeze=False, figsize=(10,5))
     ax[0,0].set(xlabel = 'Displacement (x)',ylabel="g(k)",title = "Gaussian form of the Function g(x) v/s Displacement(x)")
     ax[0,0].plot(B, C, c='midnightblue',label = 'g(k) = 1 for a = 1')
     ax[0,0].plot(B, E,c='mediumvioletred',label = 'g(k) = 1 for a = 4')
     ax[0,0].plot(B, G,c='mediumturquoise',label = 'g(k) = 1 for a = 8')
     ax[0,0].legend(loc = 'best')
     ax[0,0].grid()
     ax[0,1].plot(A,D , c = 'magenta' , label = 'B(x) = √π/1 = 1.78 for a = 1') 
     ax[0,1].plot(A,F,c ='sienna', label = 'B(x) = √π/4 = 0.89 for a = 4') 
     ax[0,1].plot(A,H,c ='yellowgreen', label = 'B(x) = √π/8 = 0.22 for a = 8') 
     ax[0,1].set(ylabel="Amplitude of the wavepacket B(k)") 
     ax[0,1].legend(loc = 'best') 
     ax[0,1].set(xlabel = "Displacement (x)",ylabel="Amplitude of the wavepacket B(k)",title = "Gaussian Wavepacket for B(x)")
     ax[0,1].legend(loc = 'best')
     ax[0,1].set(xlabel ="Displacement (x)",ylabel="Amplitude of the wavepacket B(k)",title = "Amplitude of the wavepacket B(x) v/s Displacement(x)")
     ax[0,1].grid()
     ax[0,1].legend(loc = 'best')

def plot_1(A1,B1,C1,D1):
      fig,axe = plt.subplots(2,2)
      axe[0,0].plot(A1, B1, label = '$\operatorname{Re}(\psi)$',c='navy')
      axe[0,0].plot(A1, C1, label = '$\operatorname{Im}(\psi)$',c='pink')
      axe[0,0].plot(A1, D1, label = '$\psi \psi^{\dag} $',c='limegreen')
      axe[0,0].set_title("Gaussian Wavepacket for $\operatorname{Re}(\psi)$,$\operatorname{Im}(\psi)$ and $\psi \psi^{\dag}$")
      axe[0,0].set(xlabel = "Time (t)")
      axe[0,0].grid()
      axe[0,0].legend(loc='best')
      axe[0,1].plot(A1, B1,c='navy')
      axe[0,1].set(xlabel = "Time (t)",ylabel = '$\operatorname{Re}(\psi)$' ,title = 'For Real part of Wave Function')
      axe[0,1].grid()
      axe[1,0].plot(A1, C1,c='pink')
      axe[1,0].grid()
      axe[1,0].set(xlabel = "Time (t)",ylabel = '$\operatorname{Im}(\psi)$',title = 'For Imagine part of Wave Function')
      axe[1,1].plot(A1,D1,c='limegreen')
      axe[1,1].set(xlabel = "Time (t)",ylabel = '$\psi \psi^{\dag}$',title = 'For modulus of Wave Function $\psi \psi^{\dag}$')
      axe[1,1].grid()

if __name__ == "__main__" :   
    a = 1 
    a0 = 4  #Gaussian distribution factor
    a1 = 8
    k0 = 10 #Central wavenumber of the Gaussian packet
    alpha = 1 #alpha = hbar/m => v = alpha*k
    x = np.linspace(-20,20,1000) #Displacement
    x1 = np.linspace(0,20,1000)
    y = np.exp(-(a**2)*(k0-x1)**2) #distribution some frequencies
    y1 = np.exp(-(a0**2)*(k0-x1)**2) 
    y2 = np.exp(-(a1**2)*(k0-x1)**2)
    B_x = np.exp(-(x**2)/4*a)*np.sqrt(pi/a)  #amplitude of the wavepacket
    B_x1 = np.exp(-(x**2)/4*a0)*np.sqrt(pi/a0)
    B_x2 = np.exp(-(x**2)/4*a1)*np.sqrt(pi/a1)
    ########################################################################################################
    t=np.linspace(-1,1,1000) #time 
    probCompl = psi(t) #Wave Function

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plt.subplots_adjust(left=0.12, bottom=0.35)
is_color = False 
ax_a = plt.axes([0.1, 0.05, 0.8, 0.03])
a_slider = Slider(ax_a, '$t$', 0, 10, valinit=0) #slides the time
a_slider.label.set_size(20)

ax_b = plt.axes([0.1, 0.15, 0.8, 0.03])
b_slider = Slider(ax_b, '$a$', 0.01, 5, valinit=a) #slider for a 
b_slider.label.set_size(20)

ax_c = plt.axes([0.1, 0.25, 0.8, 0.03])
c_slider = Slider(ax_c, '$k_0$', 1, 15, valinit=k0)#slider for k0
c_slider.label.set_size(20)

rax = plt.axes([0.01, 0.45, 0.08, 0.1]) #Color control button
check = CheckButtons(rax,['color'], [False])


def update_phase(val_):
	global a
	a = b_slider.val
	y = np.exp(- (a**2)*(k0-x)**2) #distribution some frequencies
	B_x = np.exp(-(x**2)/4*a)*np.sqrt(pi/a)
	ax1.clear()
	ax1.set_title('Gaussian Wavepacket')
	ax1.plot(x, y,label='g(k)')
	ax1.plot(x,B_x,label='B(x)')
	ax1.grid()
	ax1.legend()
	fig.canvas.draw_idle()
	update_temps(0) 

def update_temps(val_): #Drawing the wave function

	probCompl = psi(a_slider.val)
	ax2.clear()
	ax2.set_xlim([-20,20])
	ax2.set_ylim([-4, 4])
	ax2.set_title('Wavefunction')

	if(is_color): #Color Display
		X = np.array([x,x])
		y0 = np.zeros(len(x))
		y = [abs(i) for i in probCompl]
		Y = np.array([y0,y])
		Z = np.array([probCompl,probCompl])
		C = np.angle(Z)
		ax2.pcolormesh(X, Y, C, cmap=cm.hsv, vmin=-np.pi, vmax=np.pi)
		ax2.plot(x, np.abs(probCompl), label = '$|\psi|$', color='black')
		ax2.grid()

	else: #Display of real and complex games
		ax2.plot(x, np.real(probCompl), label = '$\operatorname{Re}(\psi)$')
		ax2.plot(x, np.imag(probCompl), label = '$\operatorname{Im}(\psi)$')
		ax2.plot(x, np.absolute(probCompl)**2, label = '$\psi \psi^{\dag} $')
		ax2.legend(loc='best')
		ax2.grid()
	
	ax2.legend(fontsize=15)
	fig.canvas.draw_idle()

def update_k(val_): #changing k0
	global k0
	k0 = c_slider.val
	update_phase(0)
	update_temps(0)

def on_check(label): #when you click on the button
	global is_color
	is_color = not is_color
	update_temps(0)
update_phase(0) #Creation of the first frame

#Associating functions with sliders
a_slider.on_changed(update_temps)
b_slider.on_changed(update_phase)
c_slider.on_changed(update_k)

check.on_clicked(on_check)

plot(x,x1,y,B_x,y1,B_x1,y2,B_x2)
plot_1(t,np.real(probCompl),np.imag(probCompl),np.absolute(probCompl)**2)
plt.show()

#Printing Data for the results
data1 = {"Displacement(x)":x,"Gaussian form of the Function g(k) for a = 1":y,"Gaussian form of the Function g(k) for a = 4":y1,"Gaussian form of the Function g(k) for a = 8":y2}
print(pd.DataFrame(data1))
data2 = {"Displacement(x)":x1,"Amplitude of the wavepacket B(k) for a = 1":B_x,"Amplitude of the wavepacket B(k) for a = 4":B_x1,"Amplitude of the wavepacket B(k) for a = 8":B_x2}
print(pd.DataFrame(data2))
data3 = {"Time (t)":t,"Real(Ψ)":np.real(probCompl), "Imaginary(Ψ)":np.imag(probCompl),"ΨΨ+":np.absolute(probCompl)**2}
print(pd.DataFrame(data3))
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.max_rows', None)  