from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import cv2
from random import choice
import numpy as np



class ImageDatasetGenerator(keras.utils.Sequence):
    def __init__(self, files, labels, batch_size, output_dim, metadata, train_mode=True):
        self.dataset = files
        self.labels = labels
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.classes = {}
        self.metadata = metadata
        self.train_mode = train_mode
        self.random_flip_prob = 0.5
        self.load_specimens(files)
        self.image_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)


    def load_specimens(self, files):

        for i, file in enumerate(files):
            specimen_number = self.metadata[file.split('/')[-1]]['specimen_number']
            class_name = self.metadata[file.split('/')[-1]]['label']
            if class_name not in self.classes.keys():
                self.classes[class_name] = {}

            if specimen_number in self.classes[class_name].keys():
                self.classes[class_name][specimen_number].append(i)
            else:
                self.classes[class_name][specimen_number] = [i]

    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if self.train_mode:

            (width, height, depth) = self.output_dim[0], self.output_dim[1], 3

            X = np.empty((self.batch_size, width, height, depth), dtype=int)
            y = np.empty((self.batch_size), dtype=int)

            for i in range(self.batch_size):
                class_name = choice(list(self.classes.keys()))

                tree_number = choice(list(self.classes[class_name].keys()))

                index = choice(self.classes[class_name][tree_number])

                path = self.dataset[index]

                img = cv2.imread(path)

                img = cv2.resize(img, (width, height))

                #img = self.random_crop(img, (width, height))

                img = self.image_generator.random_transform(img)

                X[i] = img
                y[i] = self.labels[index]

            return X, keras.utils.to_categorical(y, num_classes=len(self.classes.keys()))
        else:
            (width, height, depth) = self.output_dim[0], self.output_dim[1], 3

            X = np.empty((self.batch_size, width, height, depth), dtype=int)
            y = np.empty((self.batch_size), dtype=int)

            j = 0

            for i in range(index * self.batch_size, (index + 1) * self.batch_size):
                path = self.dataset[i]

                img = cv2.imread(path)

                img = cv2.resize(img, (width, height))

                #img = self.random_crop(img, (width, height))

                X[j] = img
                y[j] = self.labels[i]

                j += 1

            return X, keras.utils.to_categorical(y, num_classes=len(self.classes.keys()))
