from music21 import converter
from joblib import load, dump
from music21 import key
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


n_threads = 8

keys = load('./dataset/keys')
key_counts = {}
transposed_key_counts = {}

for value in keys.values():
    if value in key_counts:
        key_counts[value] += 1
    else:
        key_counts[value] = 1


total_keys = sum(key_counts.values())
avg_count = total_keys / len(key_counts)

print('Average key count: ', avg_count)

low_keys = []
high_keys = []

for k, v in key_counts.items():
    if v < avg_count:
        low_keys.append(k)
    elif v > avg_count:
        high_keys.append(k)

print('Low keys:', low_keys)
print('High keys:', high_keys)


# sort keys
key_counts = {k: v for k, v in sorted(key_counts.items(), key=lambda item: item[1], reverse=True)}
print(key_counts)

transposed_keys = {}

t_midi_path = './dataset/transposed_midi' 

for midi in tqdm(keys.keys()):
    t_midi_path = midi.replace('midi', 'transposed_midi', 1)
    t_midi_path_folders = '/'.join(t_midi_path.split('/')[:-1])
    
    try:
        os.makedirs(t_midi_path_folders)
    except:
        pass

    tonic_name = keys[midi].split(' ')[0]
    current_key = key.Key(tonic_name)
    #print('old', current_key)

    if key_counts[keys[midi]] < avg_count:
        score = converter.parse(midi)
        score.write('mid', t_midi_path)
        continue

    # Get the closest keys with the same mode
    closely_related_keys = [('p4', current_key.transpose('p4')), ('p5', current_key.transpose('p5'))]
    #print('Related keys: ', closely_related_keys)

    # choose the lowest one
    best_closely_related_key = current_key
    best_interval = 'p1'
    lowest_count = float('inf')
    for k in closely_related_keys:
        temp = str(k[1])
        if temp == 'D- major':
            temp = str(key.Key('C#'))

        
        if key_counts[temp] < lowest_count:
            lowest_count = key_counts[temp]
            best_closely_related_key = k[1]
            best_interval = k[0]


    #print('New key: ', best_closely_related_key)
    #transpose the score and save it to a new file
    score = converter.parse(midi)
    transposed_score = score.transpose(best_interval)
    transposed_score.write('mid', t_midi_path)
    
    temp = str(best_closely_related_key)
    if temp == 'D- major':
        temp = str(key.Key('C#'))
    key_counts[temp] += 1
    key_counts[str(current_key)] -= 1

   


print(key_counts)

dump(key_counts, './dataset/transposed_key_counts')
