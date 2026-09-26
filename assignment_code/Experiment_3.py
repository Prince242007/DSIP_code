import numpy as np
import matplotlib.pyplot as plt


def cross_correlation(signal1, signal2):
    cross_corr = np.correlate(signal1, signal2, mode='full')
    return cross_corr

def autocorrelation(signal):
    auto_corr = np.correlate(signal, signal, mode='full')
    return auto_corr

signal1 = np.array([1, 2, 3, 4, 5])

signal2 = np.array([2, 4, 6, 8, 10])

cross_corr = cross_correlation(signal1, signal2)

auto_corr = autocorrelation(signal1)


print("Signal 1:")
print(signal1)

print("\nSignal 2:")
print(signal2)

print("\nCross Correlation:")
print(cross_corr)

print("\nAutocorrelation:")
print(auto_corr)


# ------------------------------
# Plot Graphs
# ------------------------------
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.stem(cross_corr)
plt.title("Cross Correlation")
plt.xlabel("Time Lag")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(auto_corr)
plt.title("Autocorrelation")
plt.xlabel("Time Lag")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()