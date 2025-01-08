import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def visualize_audio_as_complex_matrix(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)  # y is the waveform, sr is the sample rate

    # Perform Short-Time Fourier Transform (STFT)
    stft_matrix = librosa.stft(y)

    # Get magnitude and phase from the STFT matrix
    magnitude = np.abs(stft_matrix)  # Magnitude spectrogram
    phase = np.angle(stft_matrix)   # Phase spectrogram

    # Plotting
    plt.figure(figsize=(12, 8))

    # Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Magnitude spectrogram
    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                             sr=sr, x_axis='time', y_axis='log')
    plt.title("Magnitude Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    # Phase spectrogram
    plt.subplot(3, 1, 3)
    plt.imshow(phase, aspect='auto', origin='lower', extent=[0, y.shape[0] / sr, 0, sr // 2])
    plt.title("Phase Spectrogram")
    plt.colorbar(label="Phase (radians)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_file = "1khz_audio.wav"
    visualize_audio_as_complex_matrix(audio_file)

