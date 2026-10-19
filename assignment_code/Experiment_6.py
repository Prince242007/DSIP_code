
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# ==========================================
# DSIP EXPERIMENT: DESIGN OF FIR FILTERS
# Sampling Frequency = 10 kHz
# Number of taps N = 31
# Window Method = Hamming Window
# ==========================================

Fs = 10000          # Sampling frequency in Hz
N = 31              # Number of filter coefficients (taps)
nyquist = Fs / 2    # Nyquist frequency = 5000 Hz

# Given cutoff frequencies
fc_low = 3000
fc_high = 6000

print("=" * 60)
print("FIR FILTER DESIGN EXPERIMENT")
print("=" * 60)
print(f"Sampling Frequency : {Fs / 1000:.1f} kHz")
print(f"Number of Taps     : {N}")
print(f"Nyquist Frequency  : {nyquist / 1000:.1f} kHz")
print("Window             : Hamming")
print()

# A digital cutoff must be strictly below Nyquist.
# 6 kHz is invalid for a 10 kHz sampling frequency.
if fc_high >= nyquist:
    print("WARNING: 6 kHz is not a valid digital cutoff!")
    print("Nyquist frequency is 5 kHz.")
    print("Using 4 kHz as a temporary valid upper cutoff")
    print("for the band-pass and band-stop demonstrations.")
    print("Please confirm this correction with your teacher.")
    print()
    fc_high_used = 4000
else:
    fc_high_used = fc_high

# ==========================================
# 1. FIR LOW-PASS FILTER
# Cutoff frequency = 3 kHz
# ==========================================

low_pass = firwin(
    N,
    cutoff=fc_low,
    fs=Fs,
    window="hamming",
    pass_zero="lowpass"
)

# ==========================================
# 2. FIR HIGH-PASS FILTER
# Cutoff frequency = 3 kHz
# ==========================================

high_pass = firwin(
    N,
    cutoff=fc_low,
    fs=Fs,
    window="hamming",
    pass_zero="highpass"
)

# ==========================================
# 3. FIR BAND-PASS FILTER
# Cutoff frequencies = 3 kHz and 4 kHz*
# ==========================================

band_pass = firwin(
    N,
    cutoff=[fc_low, fc_high_used],
    fs=Fs,
    window="hamming",
    pass_zero="bandpass"
)

# ==========================================
# 4. FIR BAND-STOP FILTER
# Cutoff frequencies = 3 kHz and 4 kHz*
# ==========================================

band_stop = firwin(
    N,
    cutoff=[fc_low, fc_high_used],
    fs=Fs,
    window="hamming",
    pass_zero="bandstop"
)

# Store all filters for analysis
filters = [
    ("Low-Pass Filter", low_pass),
    ("High-Pass Filter", high_pass),
    ("Band-Pass Filter", band_pass),
    ("Band-Stop Filter", band_stop)
]

# ==========================================
# DISPLAY FILTER COEFFICIENTS
# ==========================================

for name, coefficients in filters:
    print("-" * 60)
    print(name)
    print("-" * 60)
    print("Filter coefficients:")
    print(np.array2string(
        coefficients,
        precision=5,
        suppress_small=True,
        max_line_width=100
    ))
    print()

# ==========================================
# PLOT FREQUENCY RESPONSE OF ALL FOUR FILTERS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(
    "FIR Filter Frequency Responses (Hamming Window)",
    fontsize=15
)

for ax, (name, coefficients) in zip(axes.flat, filters):
    frequencies, response = freqz(
        coefficients,
        worN=2048,
        fs=Fs
    )

    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(response), 1e-8)
    )

    ax.plot(frequencies / 1000, magnitude_db)
    ax.set_title(name)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, Fs / 2000)
    ax.set_ylim(-80, 5)
    ax.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# CONCLUSION
# ==========================================

print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print("1. The FIR low-pass filter passes low frequencies.")
print("2. The FIR high-pass filter passes high frequencies.")
print("3. The FIR band-pass filter passes a selected frequency band.")
print("4. The FIR band-stop filter rejects a selected frequency band.")
print("All filters use 31 coefficients and a Hamming window.")
print("The band-pass and band-stop upper cutoff was adjusted")
print("to 4 kHz because 6 kHz exceeds the 5 kHz Nyquist limit.")
print("Confirm the intended cutoff with your teacher.")
