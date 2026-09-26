import numpy as np 
""" 
NumPy is used for: 
creating sample indices, 
mathematical calculations, 
generating sequences. 
""" 
import matplotlib.pyplot as plt 
""" 
Matplotlib is used to draw the signals. 
""" 
# Discrete sample indices from -10 to 10 
#imp :- The upper limit 11 is excluded, so the last value is 10. 
n = np.arange(-10, 11) 
# 1. Unit impulse sequence 
impulse = np.where(n == 0, 1, 0) 
# n=0 upar value 1 baaki badhey 0. 
plt.figure()# graph maate nu base plate ke bg paper(graph window) 
plt.stem(n, impulse) # for discrete time sgnal we use stem function of numpy library 
plt.title("Unit Impulse Sequence")# title che graph nu 
plt.xlabel("n")# x upar nu label 
plt.ylabel("δ[n]")# y upar nu label 
plt.grid(True)# grid show thai e maate 
# 2. Unit step sequence 
step = np.where(n >= 0, 1, 0) 
# n>=0 upar 1 baaki badhey 0. 
plt.figure() 
plt.stem(n, step)# dicrete time signal graph n ane step function vache. 
plt.title("Unit Step Sequence") 
plt.xlabel("n") 
plt.ylabel("u[n]") 
plt.grid(True)

# 3. Unit ramp sequence 
ramp = np.where(n >= 0, n, 0) 
# n>=0 upar value n , baaki badhey 0. 
plt.figure() 
plt.stem(n, ramp) 
plt.title("Unit Ramp Sequence") 
plt.xlabel("n") 
plt.ylabel("r[n]") 
plt.grid(True) 
# 4. Exponential sequence 
a = 0.8 
exponential = np.where(n >= 0, a**n, 0) 
# n>=0 upar value a^n baakio badhey 0. [x[n]=(0.8)n(u[n])] 
# as 0<a<1 -> decaying exponential sequence 
#if a=2 which is a>1 -> growing exponential sequence 
plt.figure() 
plt.stem(n, exponential) 
plt.title("Exponential Sequence: (0.8)^n u[n]") 
plt.xlabel("n") 

plt.ylabel("x[n]") 
plt.grid(True) 
# 5. Sinusoidal sequence 
frequency = 0.2 * np.pi  #(np.pi == pi(3.14)) 
sinusoidal = np.sin(frequency * n) 
plt.figure() 
plt.stem(n, sinusoidal) 
plt.title("Sinusoidal Sequence") 
plt.xlabel("n") 
plt.ylabel("x[n]") 
plt.grid(True) 
# Display all graphs 
plt.show()