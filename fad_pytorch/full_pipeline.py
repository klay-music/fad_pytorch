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
modes = variables['modes']

# create fakes directory (storing all fake directories with different snr values)
if os.path.exists('fakes'):
    shutil.rmtree('fakes')
os.mkdir('fakes')

fad_scores = []
for mode in modes:
    for model in models:
        if mode == 'all':
            if not os.path.exists('real_original_emb_' + model):
                os.system("python fad_embed.py " + model + " real_original --one_directory True")

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
            os.system("python fad_embed.py " + model + " real --one_directory True")

            if mode == 'rem':
                # create real_rem directory
                if os.path.exists('real_rem'):
                    shutil.rmtree('real_rem')
                os.mkdir('real_rem')
                # copy all files from real_original to real_rem except the first num_samples_value files
                num_samples_per_genre = num_samples//num_genres
                step_constant = len(real_original_files)//num_genres
                forbidden_idxs = []
                for i in range(num_genres):
                    for j in range(num_samples_per_genre):
                        forbidden_idxs.append(i * step_constant + j)
                for i in range(len(real_original_files)):
                    if i not in forbidden_idxs:
                        shutil.copy(real_original_path + real_original_files[i], 'real_rem')
                # execute the fad_embed.py script
                os.system("python fad_embed.py " + model + " real_rem --one_directory True")

            for noise_color in noise_colors:
                for snr_value in snr_values:
                    with open('fad_scores.txt', 'a+') as f:
                        f.write(model + '_' + mode + '_' + str(num_samples) + '_' + noise_color + '_' + str(snr_value) + '\n')

                    # create degradations configuration file
                    json_dict = [{"name": "noise", "color": noise_color, "snr": snr_value}]
                    with open("degradations.json", "w") as jsonFile:
                        json.dump(json_dict, jsonFile)

                    # execute the degrade_audio.py script
                    os.system("python degrade_audio.py")

                    # execute the fad_embed.py and fad_score.py scripts
                    os.system("python fad_embed.py " + model + " fake --one_directory True")
                    if mode == 'all':
                        os.system("python fad_score.py real_original_emb_" + model + " fake_emb_" + model)
                    elif mode == 'rem':
                        os.system("python fad_score.py real_rem_emb_" + model + " fake_emb_" + model)
                    elif mode == 'same':
                        os.system("python fad_score.py real_emb_" + model + " fake_emb_" + model)
                    shutil.rmtree('fake_emb_' + model)

                    # copy fake directory to fake_snr if it doesn't exist
                    if not os.path.exists('fakes/fake_snr_' + model + '_' + str(num_samples) + '_' + noise_color + '_' + str(snr_value)):
                        shutil.copytree('fake', 'fakes/fake_snr_' + model + '_' + str(num_samples) + '_' + noise_color + '_' + str(snr_value))

                    # remove degradations configuration file
                    os.remove("degradations.json")

            if mode == 'rem':
                shutil.rmtree('real_rem_emb_' + model)
            elif mode == 'same':
                shutil.rmtree('real_emb_' + model)
