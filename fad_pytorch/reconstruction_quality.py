# %% Import

from typing import Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_files,
    load_and_organise_files,
    convert_to_mono_and_resample,
    compute_spectral_distances_librosa,
    compute_spectral_distances_essentia
)


# %% Load audio

essentia_bool = True

# Names of audio files
song_names = np.arange(1, 22, 1).astype(str).tolist()
# add zeros at beginning if only one digit
for i in range(len(song_names)):
    if len(song_names[i]) == 1:
        song_names[i] = "0" + song_names[i]
types = ["Original", "Reconstructed"]

target_sr = 22050
path = Path("../../data/reconstructions")

audio_paths = load_files(path)
assert len(audio_paths) == len(types) * len(song_names)

audios, sample_rates = load_and_organise_files(audio_paths)
audios = convert_to_mono_and_resample(audios, sample_rates, target_sr)

# Split original and reconstructed
originals = {}
reconstructions = {}
for audio_name in audios:
    if "ori" in audio_name:
        originals[audio_name] = audios[audio_name]
    else:
        reconstructions[audio_name] = audios[audio_name]


# %% Compute spectral distances based on Mel and CQT spectrograms

frame_length = 4096
hop_length = 2048
num_bands_mel = 128
num_bins_cqt = 84
num_bands_erb = 128
num_bands_bark = 27

reconstruction_errors: Dict[str, list] = {}

reconstruction_errors = compute_spectral_distances_librosa(
    reconstruction_errors,
    originals,
    reconstructions,
    frame_length,
    hop_length,
    num_bands_mel,
    num_bins_cqt,
    target_sr,
)

reconstruction_errors = compute_spectral_distances_essentia(
    reconstruction_errors,
    originals,
    reconstructions,
    frame_length,
    hop_length,
    num_bands_erb,
    num_bands_bark,
)

# normalise errors from 0 to 1 by error type
for i in range(8):
    errors = [
        reconstruction_errors[audio_name][i] for audio_name in reconstruction_errors
    ]
    errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
    for j, audio_name in enumerate(reconstruction_errors):
        reconstruction_errors[audio_name][i] = errors[j]

# plot reconstruction errors
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(
    np.arange(len(reconstruction_errors)) - 0.3,
    [reconstruction_errors[audio_name][0] for audio_name in reconstruction_errors],
    width=0.2,
    label="Cross Entropy Mel",
)
ax.bar(
    np.arange(len(reconstruction_errors)) - 0.1,
    [reconstruction_errors[audio_name][2] for audio_name in reconstruction_errors],
    width=0.2,
    label="Cross Entropy CQT",
)
ax.bar(
    np.arange(len(reconstruction_errors)) + 0.1,
    [reconstruction_errors[audio_name][6] for audio_name in reconstruction_errors],
    width=0.2,
    label="Cross Entropy ERB",
)
ax.bar(
    np.arange(len(reconstruction_errors)) + 0.3,
    [reconstruction_errors[audio_name][7] for audio_name in reconstruction_errors],
    width=0.2,
    label="Cross Entropy Bark",
)
ax.set_xticks(np.arange(len(reconstruction_errors)))
ax.set_xticklabels([audio_name for audio_name in reconstruction_errors])
ax.set_ylabel("Reconstruction Error")
ax.set_xlabel("Audio Sample")
ax.set_title("Reconstruction Errors")
plt.legend()
plt.show()
