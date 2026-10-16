from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fft import fft, ifft
import os

# Load WAV file

sample_rate, samples = wavfile.read("break.wav")

print("====================================")
print("Audio Loaded Successfully")
print("====================================")

print("Sample Rate :", sample_rate)
print("Shape       :", samples.shape)
print("Data Type   :", samples.dtype)


# Convert Stereo to Mono (if needed)

if len(samples.shape) == 2:
    print("Stereo Audio Detected")
    samples = samples.mean(axis=1).astype(np.int16)
    print("Converted to Mono")
else:
    print("Audio is already Mono")



print("\nFirst 10 Audio Samples:")

print(samples[:10])

# ---------------------------------------
# Plot Original Audio Signal
# ---------------------------------------

plt.figure(figsize=(12,4))

plt.plot(samples, color="blue")

plt.title("Original Audio Signal")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")

plt.grid(True)

plt.show()


# ---------------------------------------
# Impulse Response (Kernel)
# ---------------------------------------

kernel = np.array([1, 0, 1, 0, 1], dtype=np.float32)

print("\nImpulse Response :")
print(kernel)

# ---------------------------------------
# Audio Convolution
# ---------------------------------------

convoluted = convolve(samples, kernel, mode="same")

print("\nConvolution Completed")
print("Number of Samples :", len(convoluted))

# ---------------------------------------
# Normalize Audio
# ---------------------------------------

convoluted = convoluted / np.max(np.abs(convoluted))

convoluted = (convoluted * 32767).astype(np.int16)

print("\nNormalization Completed")


from scipy.io.wavfile import write

# ---------------------------------------
# Save Convoluted Audio
# ---------------------------------------

write("output_convoluted.wav", sample_rate, convoluted)

print("\nConvoluted Audio Saved Successfully")


# ---------------------------------------
# Plot Original and Convoluted Signals
# ---------------------------------------

plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.plot(samples, color="blue")
plt.title("Original Audio Signal")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(convoluted, color="red")
plt.title("Convoluted Audio Signal")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()