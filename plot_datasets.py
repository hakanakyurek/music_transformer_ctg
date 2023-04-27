import matplotlib.pyplot as plt
import os
from joblib import load, dump
from collections import defaultdict
from tqdm import tqdm
import numpy as np


PATHS = {

'ARTISTDATASETPATH': 'dataset/encoded_new/random_seq/artist',
'GENREDATASETPATH': 'dataset/encoded_new/random_seq/genre',

'KEYDATASETPATH': 'dataset/encoded_new/random_seq/key',
'TARNSPOSEDKEYDATASETPATH': 'dataset/encoded_new_transposed/random_seq/key',
}

for k ,path in PATHS.items():
    print(path)
    histogram = defaultdict(int)

    for root, directories, files in tqdm(os.walk(path)):
        for file in files:
            file = load(os.path.join(root, file))
            if k != 'ARTISTDATASETPATH':
                histogram[file[0]] += 1
            else:
                histogram['.'.join([s[0] for s in file[0].split(' ')])] += 1

    dump(histogram, k)
    x = list(histogram.keys())
    y = list(histogram.values())
    plt.figure(figsize=(18, 8))
    plt.bar(x, y, color='blue', edgecolor='black')

    mean = np.mean(y)
    plt.axhline(mean, color='red', linestyle='--', label='Mean')

    plt.legend()
    
    plt.xticks(rotation=65, fontsize=10)

    plt.title('Bar Chart of Sample Data')
    plt.ylabel('Number of Pieces')

    plt.savefig(k + '.png')