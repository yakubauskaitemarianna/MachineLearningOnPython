import os
import skimage
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                    if os.path.isdir(os.path.join(data_directory, d))]

    labels = []
    images = []

    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                        for f in os.listdir(label_directory)
                        if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))

    return np.array(images), np.array(labels)

# my directory for datasets is same to current workspace
ROOT_PATH = os.getcwd()

train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

images_dimenth = images.ndim
images_count = images.size

labels_dimenth = labels.ndim
labels_count = labels.size

# Determine the (random) indexes of the images that you want to see
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images that you defined
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)

plt.show()

# Задание (случайных) номеров изображений, которые вы хотите вывести

traffic_signs = [300, 2250, 3650, 4000]

# Заполнение графиков изображениями и вывод размеров

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                  images[traffic_signs[i]].min(),
                                                  images[traffic_signs[i]].max()))

unique_labels = set(labels)
plt.figure(figsize=(15, 15))
i = 1

for label in unique_labels:
    image = images[labels.index(label)]
    plt.subplot(8, 8, i)

    plt.axis('off')
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    i += 1
    plt.imshow(image)

plt.show()
