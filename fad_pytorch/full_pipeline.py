import os
import json
import shutil


# define variables
num_genres = 10
models = ["clap", "vggish"]
nums_samples = [20, 50, 100]
noise_colors = ["pink", "white"]
snr_values = [10, 20, 30, 40, 50]

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
    # create fakes directory (storing all fake directories with different snr values)
    if os.path.exists('fakes'):
        shutil.rmtree('fakes')
    os.mkdir('fakes')

    # copy first num_samples_value files from real_original to real
    real_original_path = 'real_original/'
    real_original_files = os.listdir(real_original_path)
    real_original_files = sorted(real_original_files)

    num_samples_per_genre = num_samples//num_genres
    step_constant = len(real_original_files)//num_genres
    for i in range(num_genres):
        for j in range(num_samples_per_genre):
            shutil.copy(real_original_path + real_original_files[i*step_constant + j], 'real')

    for noise_color in noise_colors:
        for snr_value in snr_values:
            # create degradations configuration file
            json_dict = [{"name": "noise", "color": noise_color, "snr": snr_value}]
            with open("degradations.json", "w") as jsonFile:
                json.dump(json_dict, jsonFile)

            # execute the degrade_audio.py script
            os.system("python degrade_audio.py")

            # execute the fad_embed.py script
            os.system("python fad_embed.py clap real fake")

            # execute the fad_score.py script
            os.system("python fad_score.py real_emb_clap fake_emb_clap")

            # remove name_real and name_fake directories
            shutil.rmtree('real_emb_clap')
            shutil.rmtree('fake_emb_clap')

            # copy fake directory to fake_snr
            shutil.copytree('fake', 'fakes/fake_snr_' + str(num_samples) + '_' + noise_color + '_' + str(snr_value))

            # remove degradations configuration file
            os.remove("degradations.json")