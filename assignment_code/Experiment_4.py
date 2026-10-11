
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, freqz, lfilter

# --------------------------------------------------
# 1. Signal Specifications
# --------------------------------------------------

sampling_frequency = 16000  # Sampling frequency in Hz
duration = 0.01             # Duration in seconds

frequencies = [1000, 2000, 3000, 4000]  # Hz
amplitude = 1

# Generate time samples
t = np.arange(0, duration, 1 / sampling_frequency)

# --------------------------------------------------
# 2. Generate Four Discrete Sine Waves
# --------------------------------------------------

signals = []

for f in frequencies:
    signal = amplitude * np.sin(2 * np.pi * f * t)
    signals.append(signal)

# Plot all four sine waves separately
plt.figure(figsize=(12, 8))

for i, signal in enumerate(signals):
    plt.subplot(2, 2, i + 1)
    plt.plot(t, signal)
    plt.title(f"Sine Wave - {frequencies[i]} Hz")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. Add All Four Signals
# --------------------------------------------------

combined_signal = np.sum(signals, axis=0)

# Plot combined signal in the time domain
plt.figure(figsize=(10, 4))
plt.plot(t, combined_signal)
plt.title("Combined Signal - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4. Function to Plot Frequency Spectrum
# --------------------------------------------------

def plot_frequency_domain(signal, title):
    n = len(signal)

    # Calculate FFT
    fft_values = np.fft.rfft(signal)
    frequencies_axis = np.fft.rfftfreq(
        n, d=1 / sampling_frequency
    )

    # Calculate single-sided amplitude spectrum
    magnitude = np.abs(fft_values) / n

    if n > 1:
        magnitude[1:-1] *= 2

    plt.figure(figsize=(10, 4))
    plt.stem(frequencies_axis, magnitude)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot combined signal in the frequency domain
plot_frequency_domain(
    combined_signal,
    "Combined Signal - Frequency Domain"
)

# --------------------------------------------------
# 5. Design Butterworth Digital Low-Pass Filter
#    Cutoff Frequency = 2 kHz
# --------------------------------------------------

filter_order = 4
butterworth_cutoff = 2000

butter_b, butter_a = butter(
    filter_order,
    butterworth_cutoff,
    btype="low",
    fs=sampling_frequency
)

# Pass combined signal through Butterworth filter
butterworth_output = lfilter(
    butter_b, butter_a, combined_signal
)

# Plot Butterworth output in the time domain
plt.figure(figsize=(10, 4))
plt.plot(t, butterworth_output)
plt.title("Butterworth Filter Output - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Butterworth output in the frequency domain
plot_frequency_domain(
    butterworth_output,
    "Butterworth Filter Output - Frequency Domain"
)

# --------------------------------------------------
# 6. Design Chebyshev Type I Digital Low-Pass Filter
#    Cutoff Frequency = 3 kHz
# --------------------------------------------------

chebyshev_cutoff = 3000
ripple = 0.5  # Passband ripple in dB

cheby_b, cheby_a = cheby1(
    filter_order,
    ripple,
    chebyshev_cutoff,
    btype="low",
    fs=sampling_frequency
)

# Pass original combined signal through Chebyshev filter
chebyshev_output = lfilter(
    cheby_b, cheby_a, combined_signal
)

# Plot Chebyshev output in the time domain
plt.figure(figsize=(10, 4))
plt.plot(t, chebyshev_output)
plt.title("Chebyshev Filter Output - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Chebyshev output in the frequency domain
plot_frequency_domain(
    chebyshev_output,
    "Chebyshev Filter Output - Frequency Domain"
)

# --------------------------------------------------
# 7. Plot Filter Magnitude Responses
# --------------------------------------------------

butter_freq, butter_response = freqz(
    butter_b,
    butter_a,
    worN=2048,
    fs=sampling_frequency
)

cheby_freq, cheby_response = freqz(
    cheby_b,
    cheby_a,
    worN=2048,
    fs=sampling_frequency
)

plt.figure(figsize=(10, 5))

plt.plot(
    butter_freq,
    20 * np.log10(np.maximum(np.abs(butter_response), 1e-10)),
    label="Butterworth - 2 kHz"
)

plt.plot(
    cheby_freq,
    20 * np.log10(np.maximum(np.abs(cheby_response), 1e-10)),
    label="Chebyshev - 3 kHz"
)

plt.title("Digital Low-Pass Filter Frequency Responses")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Save Filter Coefficients
# --------------------------------------------------

np.savetxt(
    "butterworth_coefficients.txt",
    np.vstack((butter_b, butter_a)),
    delimiter=",",
    header="Numerator coefficients (b), Denominator coefficients (a)"
)

np.savetxt(
    "chebyshev_coefficients.txt",
    np.vstack((cheby_b, cheby_a)),
    delimiter=",",
    header="Numerator coefficients (b), Denominator coefficients (a)"
)

print("Experiment completed successfully!")
print("Butterworth cutoff frequency:", butterworth_cutoff, "Hz")
print("Chebyshev cutoff frequency:", chebyshev_cutoff, "Hz")
print("Filter coefficients saved successfully.")
