import os
import json
import shutil


# NOTE: "export PYTORCH_ENABLE_MPS_FALLBACK=1" command might be necessary to run clap model

# define and load variables
num_genres = 10

f = open('variables.json')
variables = json.load(f)

models = variables['models']
nums_samples = variables['nums_samples']
noise_colors = variables['noise_colors']
snr_values = variables['snr_values']

# create fakes directory (storing all fake directories with different snr values)
if os.path.exists('fakes'):
    shutil.rmtree('fakes')
os.mkdir('fakes')

fad_scores = []
for num_samples in nums_samples:
    # create real directory
    if os.path.exists('real'):
        shutil.rmtree('real')
    os.mkdir('real')
    # create fake directory
    if os.path.exists('fake'):
        shutil.rmtree('fake')
    os.mkdir('fake')

    # copy first num_samples_value files from real_original to real
    real_original_path = 'real_original/'
    real_original_files = os.listdir(real_original_path)
    real_original_files = sorted(real_original_files)

    num_samples_per_genre = num_samples//num_genres
    step_constant = len(real_original_files)//num_genres
    for i in range(num_genres):
        for j in range(num_samples_per_genre):
            shutil.copy(real_original_path + real_original_files[i*step_constant + j], 'real')

    for model in models:
        for noise_color in noise_colors:
            for snr_value in snr_values:
                # create degradations configuration file
                json_dict = [{"name": "noise", "color": noise_color, "snr": snr_value}]
                with open("degradations.json", "w") as jsonFile:
                    json.dump(json_dict, jsonFile)

                # execute the degrade_audio.py script
                os.system("python degrade_audio.py")

                # execute the fad_embed.py script
                os.system("python fad_embed.py " + model + " real fake")

                # execute the fad_score.py script
                os.system("python fad_score.py real_emb_" + model + " fake_emb_" + model)

                # remove name_real and name_fake directories
                shutil.rmtree('real_emb_' + model)
                shutil.rmtree('fake_emb_' + model)

                # copy fake directory to fake_snr
                shutil.copytree('fake', 'fakes/fake_snr_' + model + '_' + str(num_samples) + '_' + noise_color + '_' + str(snr_value))

                # remove degradations configuration file
                os.remove("degradations.json")

                with open('fad_scores.txt', 'a+') as f:
                    f.write(model + '_' + str(num_samples) + '_' + noise_color + '_' + str(snr_value) + '\n')
