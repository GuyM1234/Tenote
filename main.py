import librosa
import numpy as np
import crepe
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load audio file
audio_file_path = "fur_elise.wav"
audio_data, sr = librosa.load(audio_file_path, duration=30)
 
note_in_hz = librosa.note_to_hz('g4')

stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)

freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

idx = (np.abs(freqs - note_in_hz)).argmin()

row_g = stft[idx, :]

d = np.abs(stft)

amp_values = d[idx]


threshold = np.max(amp_values) * 0.3

significant_peaks = []

for i in range(len(amp_values)):
    if amp_values[i] < threshold:
        continue
    max_amp_value = max(amp_values[i - 3:i + 3])
    if max_amp_value == amp_values[i]:
        significant_peaks.append(i)

times = librosa.frames_to_time(significant_peaks, sr=sr, n_fft=len(d))

plt.plot(librosa.frames_to_time(np.arange(len(amp_values)), sr=sr, n_fft=len(d)), amp_values)
plt.plot(times, amp_values[significant_peaks], 'ro')
plt.show()


# times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr)
# plt.plot(times, np.abs(row_440))
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()