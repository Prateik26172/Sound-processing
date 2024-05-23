import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Function to load audio files
def load_audio_files(directory, file_paths):
    audio_files = []
    for file_path in file_paths:
        full_path = os.path.join(directory, file_path)
        audio, sr = librosa.load(full_path)
        audio_files.append((audio, sr))
    return audio_files

# Function to plot spectrogram in Hz scale
def plot_spectrogram_hz(sound_names, raw_sounds):
    for i, (sound, sr) in enumerate(raw_sounds):
        plt.figure(figsize=(10, 4))
        S = librosa.stft(sound)
        S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram (Hz) of {sound_names[i]}')
        plt.tight_layout()
        plt.show()

# Function to plot spectrogram in note scale
def plot_spectrogram_note_scale(sound_names, raw_sounds):
    for i, (sound, sr) in enumerate(raw_sounds):
        plt.figure(figsize=(10, 4))
        C = librosa.cqt(sound, sr=sr)
        C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        librosa.display.specshow(C_dB, sr=sr, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram (Note) of {sound_names[i]}')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define parent directory and file paths
    parent_directory = "D:\Sound processing\myenv\Dataset"
    sound_file_paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
    sound_names = ['Sound 1', 'Sound 2', 'Sound 3']
    
    # Load audio files
    raw_sounds = load_audio_files(parent_directory, sound_file_paths)
    
    # Plot spectrograms in Hz scale
    plot_spectrogram_hz(sound_names, raw_sounds)
    
    # Plot spectrograms in note scale
    plot_spectrogram_note_scale(sound_names, raw_sounds)
