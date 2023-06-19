import os
import json
import matplotlib.pyplot as plt


f = open('variables.json')
variables = json.load(f)

models = variables['models']
nums_samples = variables['nums_samples']
noise_colors = variables['noise_colors']
snr_values = variables['snr_values']

configs = []
fad_scores = []
for num_samples in nums_samples:
    for model in models:
        for noise_color in noise_colors:
            fad_scores_config = []
            config = model + '_' + str(num_samples) + '_' + noise_color
            for snr_value in snr_values:
                query = config + '_' + str(snr_value)
                with open('fad_scores.txt') as f:
                    lines = f.readlines()
                    for idx, line in enumerate(lines):
                        if query in line:
                            fad_scores_config.append(float(lines[idx-1][:-1]))
            fad_scores.append(fad_scores_config)
            configs.append(config)

# Normalise FAD scores
for i in range(len(fad_scores)):
    fad_scores[i] = [x / max(fad_scores[i]) for x in fad_scores[i]]

# Create "plots" folder if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Plot results for each model
for idx, model in enumerate(models):
    plt.figure()
    for i in range(len(fad_scores)):
        if model in configs[i]:
            plt.plot(snr_values, fad_scores[i], label=configs[i])
    plt.xlabel('SNR')
    plt.ylabel('FAD Score')
    plt.title(model)
    plt.legend()
    plt.savefig('plots/model_' + model + '.png')
    plt.show()

# Plot results for each number of samples
for idx, num_samples in enumerate(nums_samples):
    plt.figure()
    for i in range(len(fad_scores)):
        if '_' + str(num_samples) + '_' in configs[i]:
            plt.plot(snr_values, fad_scores[i], label=configs[i])
    plt.xlabel('SNR')
    plt.ylabel('FAD Score')
    plt.title(str(num_samples) + ' samples')
    plt.legend()
    plt.savefig('plots/num_samples_' + str(num_samples) + '.png')
    plt.show()

# Plot results for each noise color
for idx, noise_color in enumerate(noise_colors):
    plt.figure()
    for i in range(len(fad_scores)):
        if noise_color in configs[i]:
            plt.plot(snr_values, fad_scores[i], label=configs[i])
    plt.xlabel('SNR')
    plt.ylabel('FAD Score')
    plt.title(noise_color + ' noise')
    plt.legend()
    plt.savefig('plots/noise_color_' + noise_color + '.png')
    plt.show()
