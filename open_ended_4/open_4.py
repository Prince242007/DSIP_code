
import os
import subprocess
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import imageio_ffmpeg

from scipy.signal import butter, sosfiltfilt

# ==========================================
# DSIP EXPERIMENT Q3 (a) AND (b)
# ==========================================

# 1. LOAD AUDIO FILE

BASE_DIR = Path(__file__).resolve().parent
filename = BASE_DIR / "hello.mp3"
output_dir = BASE_DIR / "output"

output_dir.mkdir(exist_ok=True)

if not filename.exists():
    raise FileNotFoundError(
        f"MP3 file not found: {filename}"
    )

y, sr = librosa.load(
    str(filename),
    sr=None,
    mono=True
)

print("Audio loaded successfully!")
print("Sample Rate:", sr, "Hz")
print("Duration:", round(len(y) / sr, 2), "seconds")


# ==========================================
# 2. SAVE AUDIO AS MP3
# ==========================================

def save_mp3(name, audio):
    output_path = output_dir / name
    wav_path = output_path.with_suffix(".wav")

    sf.write(str(wav_path), audio, sr)

    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-loglevel", "error",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-b:a", "192k",
                str(output_path)
            ],
            check=True
        )

        print("Saved:", output_path.name)

    finally:
        if wav_path.exists():
            wav_path.unlink()


# ==========================================
# Q3 (a): LOW-PASS FILTER
# ==========================================

def lowpass_filter(signal, cutoff, fs, order=6):

    sos = butter(
        order,
        cutoff,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


cutoffs = [500, 1500, 3500]

lowpass_results = {}

print("\nApplying Low-Pass Filters...")

for cutoff in cutoffs:

    if cutoff >= sr / 2:
        raise ValueError(
            f"Cutoff {cutoff} Hz must be below "
            f"Nyquist frequency {sr / 2} Hz"
        )

    filtered_audio = lowpass_filter(
        y, cutoff, sr
    )

    lowpass_results[cutoff] = filtered_audio

    save_mp3(
        f"lowpass_{cutoff}Hz.mp3",
        filtered_audio
    )


# ==========================================
# Q3 (b): BAND-PASS FILTER
# ==========================================

def bandpass_filter(
    signal, lowcut, highcut, fs, order=6
):

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


lowcut = 300
highcut = 3400

if highcut >= sr / 2:
    raise ValueError(
        "Band-pass upper cutoff must be below "
        "the Nyquist frequency."
    )

print("\nApplying Band-Pass Filter...")

# Estimate vocal-frequency components
vocal_audio = bandpass_filter(
    y, lowcut, highcut, sr
)

# Estimate remaining frequency components
background_audio = y - vocal_audio

save_mp3(
    "estimated_vocals.mp3",
    vocal_audio
)

save_mp3(
    "estimated_background.mp3",
    background_audio
)


# ==========================================
# 3. SPECTROGRAM FUNCTION
# ==========================================

def plot_spectrogram(signal, title, ax):

    stft = librosa.stft(signal)

    db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    librosa.display.specshow(
        db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        cmap="viridis",
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")


# ==========================================
# 4. LOW-PASS SPECTROGRAMS
# ==========================================

print("\nGenerating Low-Pass Spectrograms...")

fig, axes = plt.subplots(
    4, 1, figsize=(12, 14)
)

plot_spectrogram(
    y,
    "Original Audio",
    axes[0]
)

for i, cutoff in enumerate(cutoffs):

    plot_spectrogram(
        lowpass_results[cutoff],
        f"Low-Pass Filter: {cutoff} Hz",
        axes[i + 1]
    )

plt.tight_layout()

plt.savefig(
    output_dir / "lowpass_spectrogram.png",
    dpi=150
)

plt.close(fig)


# ==========================================
# 5. BAND-PASS SPECTROGRAMS
# ==========================================

print("\nGenerating Band-Pass Spectrograms...")

fig, axes = plt.subplots(
    3, 1, figsize=(12, 11)
)

plot_spectrogram(
    y,
    "Original Audio",
    axes[0]
)

plot_spectrogram(
    vocal_audio,
    "Estimated Vocal Frequencies (300-3400 Hz)",
    axes[1]
)

plot_spectrogram(
    background_audio,
    "Estimated Background Frequencies",
    axes[2]
)

plt.tight_layout()

plt.savefig(
    output_dir / "bandpass_spectrogram.png",
    dpi=150
)

plt.close(fig)


# ==========================================
# 6. FINAL OUTPUT
# ==========================================

print("\n====================================")
print("DSIP EXPERIMENT Q3 COMPLETED!")
print("====================================")

print("\nQ3 (a):")
print("1. lowpass_500Hz.mp3")
print("2. lowpass_1500Hz.mp3")
print("3. lowpass_3500Hz.mp3")

print("\nQ3 (b):")
print("4. estimated_vocals.mp3")
print("5. estimated_background.mp3")

print("\nSpectrogram Images:")
print("6. lowpass_spectrogram.png")
print("7. bandpass_spectrogram.png")

print("\nAll files saved in:")
print(output_dir)
