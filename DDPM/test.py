import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle


def get_dataset():

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    all_images = []
    first = True
    for i in range(1, 6):
        train_data = unpickle(f'DDPM/cifar10/data_batch_{i}')
        print(train_data.keys())
        images = train_data[b'data']
        # CIFAR-10 data is stored as (N, 3072) where each 3072 is organized as:
        # first 1024 = red channel, next 1024 = green channel, last 1024 = blue channel
        # So we need to reshape to (N, 3, 32, 32) first, then transpose to (N, 32, 32, 3)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        if first:
            all_images = images
            first = False
        else:
            all_images = np.concatenate([all_images, images], axis=0)
        # Ensure data type is uint8 for PIL

    print("dataset size: ",all_images.shape)

    return all_images


images = get_dataset()
for i in range(10):
    image = Image.fromarray(images[i])
    plt.imshow(image)
    plt.show()