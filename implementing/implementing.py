import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


file = "blues.00000.wav"

#waveform signal = sr * duration = 22050*30
signal ,sr = librosa.load(file,sr=22050) #sr = sample rate
librosa.display.waveshow(signal,sr=sr)
plt.title("Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.savefig("waveform.png")
plt.show()

#fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]


plt.plot(frequency,magnitude)
plt.title("Fast Fourier Transform")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.savefig("fft.png")
plt.show()

plt.plot(left_frequency,left_magnitude)
plt.title("Half Fast Fourier Transform")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.savefig("half-fft.png")
plt.show()


#short time fourier transform -> spectrogram
n_fft = 2048 #window size
hop_length = 512 #shift
stft = librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)
spectrogram = np.abs(stft)

librosa.display.specshow(spectrogram,sr=sr,hop_length=hop_length)
plt.title("Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.savefig("spectrogram.png")
plt.show()

#log-spectrogram
log_spectrogram = librosa.amplitude_to_db(spectrogram)
librosa.display.specshow(log_spectrogram,sr=sr,hop_length=hop_length)
plt.title("Log Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.savefig("log-spectrogram.png")
plt.show()
