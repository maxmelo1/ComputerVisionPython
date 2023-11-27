import numpy as np
import os

from sklearn.model_selection import train_test_split



IMAGE_DIR = '../datasets/cell_images/'

dataset = np.array([])
label = np.array([])

parasitized_images = [IMAGE_DIR + 'Parasitized/'+el for el in os.listdir(IMAGE_DIR + 'Parasitized/') if '.png' in el]
l = np.ones(( len(parasitized_images)))
dataset = np.concatenate((dataset, parasitized_images))
label = np.concatenate((label, l))

uninfected_images = [IMAGE_DIR + 'Uninfected/'+el for el in os.listdir(IMAGE_DIR + 'Uninfected/') if '.png' in el]
l = np.zeros(( len(uninfected_images)))
dataset = np.concatenate((dataset, uninfected_images))
label = np.concatenate((label, l))


X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)

y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)

if not os.path.exists('./data/'):
    os.makedirs('./data/')


with open('./data/train.txt', 'w') as f:
    f.write('\n'.join(X_train))
with open('./data/train_labels.txt', 'w') as f:
    f.write('\n'.join(y_train.astype(str)))

with open('./data/val.txt', 'w') as f:
    f.write('\n'.join(X_test))
with open('./data/val_labels.txt', 'w') as f:
    f.write('\n'.join(y_test.astype(str)))

print(f'Dataset lookup files with {len(X_train)} train files and {len(X_test)} validation files succesfully saved!')