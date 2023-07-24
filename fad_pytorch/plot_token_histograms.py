# imports
import os
import numpy as np
import matplotlib.pyplot as plt

# define variables
dataset_names = ['maestro', 'fma']
codebook_size = 1024
load_persisted = True
persist = True

for dataset_name in dataset_names:
    token_path = "tokens/" + dataset_name + "_encodec_24khz"

    # load if persisted, else calculate
    if load_persisted:
        histogram_data = np.load(token_path + '/' + 'histogram_data.npy')
    else:
        data_files = []
        for root, dirs, files in os.walk(token_path):
            for file in files:
                if file.endswith(".npy"):
                    data_files.append(os.path.join(root, file))

        # load data
        data = []
        for f in data_files:
            data.append(np.load(f))

        # organise histogram data
        histogram_data = np.zeros((data[0].shape[1], codebook_size))
        for i in range(len(data)):
            for j in range(data[i].shape[1]):
                for k in range(data[i].shape[2]):
                    histogram_data[j, data[i][0,j,k]] += 1
        if persist:
            np.save(token_path + '/' + 'histogram_data.npy', histogram_data)
            
    entropies = np.zeros(histogram_data.shape[0])
    for n in range(histogram_data.shape[0]):
        # plot histograms with 1024 bins
        plt.figure()
        plt.bar(np.arange(codebook_size), histogram_data[n,:])
        plt.title('Token Histogram for Quantiser ' + str(n).zfill(2))
        plt.xlabel('Token Index')
        plt.ylabel('Token Count')
        plt.savefig(token_path + '/' + 'histogram_' + str(n).zfill(2) + '.png')
        plt.close()

        # calculate and print histogram entropy
        norm_counts = (histogram_data[n,:] / histogram_data[n,:].sum())
        entropies[n] = -(norm_counts*np.log(norm_counts+1e-16)).sum() * np.log2(np.e)
        print('Entropy for Quantiser ' + str(n).zfill(2) + ': ' + str(entropies[n]))

    # print mean entropy
    print('Mean Entropy: ' + str(np.mean(entropies)))

    # plot entropy
    plt.figure()
    plt.plot(np.arange(histogram_data.shape[0]), entropies)
    plt.title('Entropy of Quantisers for ' + dataset_name + ' Dataset')
    plt.xlabel('Quantiser Index')
    plt.ylabel('Entropy')
    plt.savefig(token_path + '/' + 'entropy_' + dataset_name + '.png')
    plt.close()

    # persist entropy data
    if persist:
        np.save(token_path + '/' + 'entropies.npy', entropies)

# plot persisted entropy data from both datasets in one plot
if persist:
    entropies_maestro = np.load('tokens/maestro_encodec_24khz/entropies.npy')
    entropies_fma = np.load('tokens/fma_encodec_24khz/entropies.npy')

    plt.figure()
    plt.plot(np.arange(histogram_data.shape[0]), entropies_maestro, label='maestro')
    plt.plot(np.arange(histogram_data.shape[0]), entropies_fma, label='fma')
    plt.title('Bitrate Efficiency of Quantisers')
    plt.xlabel('Codebook Index')
    plt.ylabel('Entropy (bits)')
    plt.grid()
    plt.legend()
    plt.savefig('bitrate.png')
    plt.close()
