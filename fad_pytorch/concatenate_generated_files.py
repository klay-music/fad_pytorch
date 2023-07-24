import os
import json
import shutil
from scipy.io.wavfile import write
import numpy as np
from pathlib import Path

from utils import (
    load_files,
    load_and_organise_files,
    convert_to_mono_and_resample
)

# define and load variables
f = open('variables_validation.json')
variables = json.load(f)

steps_names = variables['steps_names']
sr = 32000

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

# copy all files from validation_dataset/5000 to original if they have "orig" in their name
for file in validation_dataset_files:
    if 'orig' in file:
        shutil.copy(validation_dataset_path + '5000/' + file, 'original')

if not os.path.exists('concatenated'):
    os.mkdir('concatenated')

for steps_name in steps_names:
    # load generated
    audio_paths = load_files(Path("generated/" + steps_name))
    generated, sample_rates = load_and_organise_files(audio_paths)
    generated = convert_to_mono_and_resample(generated, sample_rates, sr)

    # load originals
    audio_paths = load_files(Path("original"))
    originals, sample_rates = load_and_organise_files(audio_paths)
    originals = convert_to_mono_and_resample(originals, sample_rates, sr)

    # concatenate generated files using numpy concatenate
    concatenated_generated = []
    for audio_name in generated:
        for i in range(len(generated[audio_name])):
            concatenated_generated.append(generated[audio_name][i])
    concatenated_generated = np.concatenate(concatenated_generated)

    # concatenate original files using numpy concatenate
    concatenated_originals = []
    for audio_name in originals:
        for i in range(len(originals[audio_name])):
            concatenated_originals.append(originals[audio_name][i])
    concatenated_originals = np.concatenate(concatenated_originals)

    # check that concatenated and concatenated_originals have the same length
    assert len(concatenated_generated) == len(concatenated_originals)

    # Mode 1: mono files with concatenated and concatenated_originals in left and right channels
    write('concatenated/mono_generated_' + steps_name + '.wav', sr, concatenated_generated)
    if steps_name == '5000':
        write('concatenated/mono_original.wav', sr, concatenated_originals)

    # Mode 2: stereo file with concatenated and concatenated_originals in left and right channels
    concatenated = np.stack((concatenated_generated, concatenated_originals), axis=1)
    write('concatenated/stereo_' + steps_name + '.wav', sr, concatenated)
