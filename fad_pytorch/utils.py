import librosa
#import essentia.standard as es
import numpy as np
from typing import Dict, List
from pathlib import Path
from scipy.signal import deconvolve


def load_files(path: Path):
    audio_paths = list(path.glob("**/*.wav"))
    audio_paths = sorted(audio_paths)
    return audio_paths


def load_and_organise_files(audio_paths: List[Path]):
    audios: Dict[str, list] = {}
    sample_rates = []
    for audio_path in audio_paths:
        audio_name = audio_path.stem.split('.', 1)[0]
        if audio_name not in audios:
            audios[audio_name] = []
        x, sr = librosa.load(audio_path, sr=None)
        audios[audio_name].append(x)
        sample_rates.append(sr)
    return audios, sample_rates


def convert_to_mono_and_resample(
    audios: Dict[str, list], sample_rates: List[int], sr: int
):
    for audio_name in audios:
        for i in range(len(audios[audio_name])):
            if len(audios[audio_name][i].shape) > 1:
                audios[audio_name][i] = librosa.to_mono(audios[audio_name][i])
            audios[audio_name][i] = librosa.resample(
                audios[audio_name][i], orig_sr=sample_rates[i], target_sr=sr
            )
    return audios


def compute_spectral_distances_mel(
    reconstruction_errors: Dict[str, list],
    originals: Dict[str, list],
    reconstructions: Dict[str, list],
    frame_length: int,
    hop_length: int,
    num_bands_mel: int,
    target_sr: int,
):
    for audio_name in originals:
        reconstruction_errors[audio_name] = []
        for i in range(len(originals[audio_name])):
            # Mel spectrogram
            mel_spec_ori = librosa.feature.melspectrogram(
                y=originals[audio_name][i],
                sr=target_sr,
                n_fft=frame_length,
                hop_length=hop_length,
                n_mels=num_bands_mel,
            )  # define ERB bands and Bark bands
            mel_spec_rec = librosa.feature.melspectrogram(
                y=reconstructions[audio_name][i],
                sr=target_sr,
                n_fft=frame_length,
                hop_length=hop_length,
                n_mels=num_bands_mel,
            )
            # Normalise from 0 to 1
            mel_spec_ori = mel_spec_ori / np.max(mel_spec_ori)
            mel_spec_rec = mel_spec_rec / np.max(mel_spec_rec)
            # Compute MSE
            reconstruction_errors[audio_name].append(
                np.mean((mel_spec_ori - mel_spec_rec) ** 2) / mel_spec_rec.size
            )
            # Compute cross entropy
            reconstruction_errors[audio_name].append(
                -np.sum(mel_spec_ori * np.log(mel_spec_rec)) / mel_spec_rec.size
            )
    return reconstruction_errors
