import numpy as np
import matplotlib.pyplot as plt

def linear_convolution(signal1, signal2):
# Compute the linear convolution
    linear_conv = np.convolve(signal1, signal2, mode='full')
    return linear_conv

def circular_convolution(signal1, signal2):
# Compute the circular convolution
# Compute the circular convolution
    if( len(signal1) > len(signal2) ):
        fft_length=len(signal1)
    else:
        fft_length=len(signal2)
        fft_signal1 = np.fft.fft(signal1, fft_length)
        fft_signal2 = np.fft.fft(signal2, fft_length)
        circular_conv = np.fft.ifft(fft_signal1 * fft_signal2)
    return circular_conv

# Define the discrete-time signals
signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([2, 4, 6, 8, 10])
# Compute the linear convolution
linear_conv = linear_convolution(signal1, signal2)
# Compute the circular convolution
circular_conv = circular_convolution(signal1, signal2)
# Plot the linear and circular convolution results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.stem(linear_conv)
plt.title('Linear Convolution')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.stem(circular_conv)
plt.title('Circular Convolution')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()