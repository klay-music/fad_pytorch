import os
import json
import numpy as np
import matplotlib.pyplot as plt


# load variables
f = open('variables_validation.json')
variables = json.load(f)

models = variables['models']
modes = variables['modes']
mel_rec_metrics = variables['mel_rec_metrics']
steps_names = variables['steps_names']

# Load FAD scores
configs = []
scores = []
for mode in modes:
    if mode != 'reconstruction_quality_mel':
        for model in models:
            fad_scores_config = []
            config = model + '_' + mode
            for steps_name in steps_names:
                query = config + '_' + steps_name
                with open('fad_scores.txt') as f:
                    lines = f.readlines()
                    for idx, line in enumerate(lines):
                        if line == query + '\n':
                            fad_scores_config.append(float(lines[idx+1][:-1]))
            scores.append(fad_scores_config)
            configs.append(config)
    else:
        for mel_rec_metric in mel_rec_metrics:
            fad_scores_config = []
            config = mel_rec_metric + '_' + mode
            for steps_name in steps_names:
                query = config + '_' + steps_name
                with open('fad_scores.txt') as f:
                    lines = f.readlines()
                    for idx, line in enumerate(lines):
                        if line == query + '\n':
                            fad_scores_config.append(float(lines[idx+1][:-1]))
            scores.append(fad_scores_config)
            configs.append(config)

# Load validation losses
val_losses = []
with open('val_loss_12_steps.txt') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        val_losses.append(float(lines[idx]))

# Normalise FAD scores between 0 and 1
for i in range(len(scores)):
    scores[i] = [(x - min(scores[i])) / (max(scores[i]) - min(scores[i])) for x in scores[i]]

# Normalise validation losses
val_losses = [(x - min(val_losses)) / (max(val_losses) - min(val_losses)) for x in val_losses]

# Create "plots" folder if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

for mode in modes:
    if mode != 'reconstruction_quality_mel':
        # Plot results for each model and validation losses
        for idx, model in enumerate(models):
            plt.figure()
            for i in range(len(scores)):
                if model in configs[i] and mode in configs[i]:
                    plt.plot(np.array(steps_names).astype(int), scores[i], label='Score', color='red')
            plt.plot(np.array(steps_names).astype(int), val_losses, label='Validation Loss', color='blue')
            plt.xlabel('Checkpoint')
            plt.ylabel('Normalised Magnitude (Score and Loss)')
            plt.grid(which='major', axis='both', linestyle='--')
            plt.title(mode + '_' + model)
            plt.legend()
            plt.savefig('plots/' + mode + '_' + model + '.png')
            plt.show()
    else:
        # Plot results for each mel_rec_metric and validation losses
        for idx, mel_rec_metric in enumerate(mel_rec_metrics):
            plt.figure()
            for i in range(len(scores)):
                if mel_rec_metric in configs[i] and mode in configs[i]:
                    plt.plot(np.array(steps_names).astype(int), scores[i], label='Score', color='red')
            plt.plot(np.array(steps_names).astype(int), val_losses, label='Validation Loss', color='blue')
            plt.xlabel('Checkpoint')
            plt.ylabel('Normalised Magnitude (Score and Loss)')
            plt.grid(which='major', axis='both', linestyle='--')
            plt.title(mode + '_' + mel_rec_metric)
            plt.legend()
            plt.savefig('plots/' + mode + '_' + mel_rec_metric + '.png')
            plt.show()