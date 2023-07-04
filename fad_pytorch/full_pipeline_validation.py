import os
import json
import shutil
import numpy as np
from typing import Dict
from pathlib import Path
import pdb

from utils import (
    load_files,
    load_and_organise_files,
    convert_to_mono_and_resample,
    compute_spectral_distances_mel
)


# NOTE: "export PYTORCH_ENABLE_MPS_FALLBACK=1" command might be necessary to run clap model


# define and load variables
f = open('variables_validation.json')
variables = json.load(f)

models = variables['models']
modes = variables['modes']
mel_rec_metrics = variables['mel_rec_metrics']
steps_names = variables['steps_names']
target_sr = variables['target_sr']
frame_length = variables['frame_length']
hop_length = variables['hop_length']
num_bands_mel = variables['num_bands_mel']

# create generated directory
if os.path.exists('generated'):
    shutil.rmtree('generated')
os.mkdir('generated')

if not os.path.exists('validation_dataset'):
    OSError('validation_dataset directory does not exist')

fad_scores = []
for mode in modes:
    # create original directory
    if os.path.exists('original'):
        shutil.rmtree('original')
    os.mkdir('original')

    # load validation_dataset
    validation_dataset_path = 'validation_dataset/'
    validation_dataset_files = os.listdir(validation_dataset_path + '5000')
    validation_dataset_files = sorted(validation_dataset_files)

    # create generated directory for each steps_name and copy all files from
    # validation_dataset{steps_name} to generated if they have "gen" in their name
    if not os.path.exists('generated/' + steps_names[0]):
        for n in range(len(steps_names)):
            os.mkdir('generated/' + steps_names[n])
            for file in validation_dataset_files:
                if 'gen' in file:
                    shutil.copy(validation_dataset_path + steps_names[n] + '/' + file, 'generated/' + steps_names[n])

    if 'reconstruction_quality' in mode:
        # copy all files from validation_dataset/5000 to original if they have "orig" in their name
        for file in validation_dataset_files:
            if 'orig' in file:
                shutil.copy(validation_dataset_path + '5000/' + file, 'original')

    elif mode == 'audio_quality':
        if not os.path.exists('train_dataset'):
            OSError('train_dataset directory does not exist')
        else:
            # load train_dataset
            train_dataset_path = 'train_dataset/'
            train_dataset_files = os.listdir(train_dataset_path)
            train_dataset_files = sorted(train_dataset_files)

            if mode != 'audio_quality':
                # copy all files from train_dataset to original
                for file in train_dataset_files:
                    shutil.copy(train_dataset_path + file, 'original')

    if mode == 'reconstruction_quality_mel':
        for mel_rec_metric in mel_rec_metrics:
            # load originals
            audio_paths = load_files(Path("original"))
            originals, sample_rates = load_and_organise_files(audio_paths)
            originals = convert_to_mono_and_resample(originals, sample_rates, target_sr)

            for steps_name in steps_names:
                with open('fad_scores.txt', 'a+') as f:
                    f.write(mel_rec_metric + '_' + mode + '_' + steps_name + '\n')
                    
                # load generated
                audio_paths = load_files(Path("generated/" + steps_name))
                generated, sample_rates = load_and_organise_files(audio_paths)
                generated = convert_to_mono_and_resample(generated, sample_rates, target_sr)

                # get reconstruction errors
                reconstruction_errors: Dict[str, list] = {}
                reconstruction_errors = compute_spectral_distances_mel(
                    reconstruction_errors,
                    originals,
                    generated,
                    frame_length,
                    hop_length,
                    num_bands_mel,
                    target_sr
                )

                # calculate mean error for each audio
                for audio_name in reconstruction_errors:
                    mean_error = 0
                    if mel_rec_metric == "mse":
                            mean_error += reconstruction_errors[audio_name][0]
                    elif mel_rec_metric == "cross_entropy":
                            mean_error += reconstruction_errors[audio_name][1]
                    mean_error /= len(reconstruction_errors)

                with open('fad_scores.txt', 'a+') as f:
                    f.write(str(mean_error) + '\n')

    else:
        for model in models:
            if mode != 'audio_quality':
                os.system("python fad_embed.py " + model + " original --one_directory True")
            else:
                os.system("python fad_embed.py " + model + " train_dataset --one_directory True")

            # execute the fad_embed.py script for each steps_name
            for steps_name in steps_names:
                with open('fad_scores.txt', 'a+') as f:
                    f.write(model + '_' + mode + '_' + steps_name + '\n')
                os.system("python fad_embed.py " + model + " generated/" + steps_name + " --one_directory True")
                if mode != 'audio_quality':
                    os.system("python fad_score.py generated/" + steps_name + "_emb_" + model + " original_emb_" + model)
                else:
                    os.system("python fad_score.py generated/" + steps_name + "_emb_" + model + " train_dataset_emb_" + model)
                shutil.rmtree("generated/" + steps_name + "_emb_" + model)

        if mode == 'audio_quality':
            shutil.rmtree('train_dataset_emb_' + model)
        elif mode == 'reconstruction_quality':
            shutil.rmtree('original_emb_' + model)
