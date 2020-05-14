from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from random import choice
import numpy as np


class ImageDatasetGenerator(keras.utils.Sequence):
    def __init__(self, files, labels, batch_size, output_dim, metadata, random_crop=False, rotation = False, train_mode=True, **kwargs):
        self.dataset = files
        self.labels = labels
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.classes = {}
        self.metadata = metadata
        self.train_mode = train_mode
        self.load_specimens(files)
        self.image_generator = ImageDataGenerator(**kwargs)
        self.possible_rotations = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_90_COUNTERCLOCKWISE, 3: cv2.ROTATE_180}
        self.random_crop = random_crop
        self.rotation = rotation
        self.transformations = None
        self.create_transformations()

    def rotate_image(self, img):
        rand_int = np.random.randint(4)
        if rand_int != 0:
            img = cv2.rotate(img, self.possible_rotations[rand_int])
        return img

    def create_transformations(self):
        dim_image = (self.output_dim[0], self.output_dim[1])
        self.transformations = []
        if self.random_crop:
            first_transformation = lambda x: self.rand_crop(x, dim_image)
        else:
            first_transformation = lambda x: cv2.resize(x, dim_image)
        self.transformations.append(first_transformation)

        if self.rotation:
            self.transformations.append(self.rotate_image)

        self.transformations.append(self.image_generator.random_transform)


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

    def rand_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def transform_image(self, img):
        for transformation in self.transformations:
            img = transformation(img)
        return img

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

                img = self.transform_image(img)

                X[i] = img
                y[i] = self.labels[index]

            return X, keras.utils.to_categorical(y, num_classes=len(self.classes.keys()))
        else:
            (width, height, depth) = self.output_dim[0], self.output_dim[1], 3

            batch_data = self.dataset[index * self.batch_size: (index + 1) * self.batch_size]
            labels_data = self.labels[index * self.batch_size: (index + 1) * self.batch_size]

            len_batch = len(batch_data)
            X = np.empty((len_batch, width, height, depth), dtype=int)
            y = np.empty((len_batch), dtype=int)

            for j, (path, label) in enumerate(zip(batch_data, labels_data)):
                img = cv2.imread(path)

                ##Just executing first transformation.
                img = self.transformations[0](img)

                X[j] = img
                y[j] = label

            return X, keras.utils.to_categorical(y, num_classes=len(self.classes.keys()))
