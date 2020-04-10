import json
import os
import math
from PIL import Image
import random


class GenerateDataset:
    IGNORED = ['ERB', 'PEG', 'PID']

    def __init__(self, path, metadata):
        self.path = path
        self.specimens = {}
        self.dataset = {}
        self.classes = []
        self.train = {}
        self.test = {}
        self.metadata = metadata

    def load_dataset(self, existing_dataset, split=0.5):
        for class_name in os.listdir(self.path):
            if class_name in self.IGNORED:
                continue
            self.classes.append(class_name)

        if existing_dataset != 'None':
            dataset_file = os.path.join(existing_dataset)
            dataset_file = open(dataset_file)
            loaded_dataset = json.load(dataset_file)
            dataset_file.close()
            self.train = loaded_dataset['train']
            self.test = loaded_dataset['test']
        else:
            self.load_new_dataset(split)

    def num_classes(self):
        return len(self.classes)

    def add_ignore(self):
        files = []
        labels = []
        for class_name in os.listdir(self.path):
            if class_name not in self.IGNORED:
                continue
            self.classes.append(class_name)
            for file in os.listdir(os.path.join(self.path, class_name)):
                file_path = os.path.join(self.path, class_name, file)
                if not file.endswith('NAME.jpg'):
                    img = Image.open(file_path)
                    width, height = img.size
                    if width >= 224 and height >= 224:
                        files.append(file_path)
                        labels.append(class_name)

        self.classes.sort()

        for i, file in enumerate(files):
            self.train['files'].append(file)
            self.train['labels'].append(self.classes.index(labels[i]))

    def load_new_dataset(self, split):
        for class_name in os.listdir(self.path):
            if class_name in self.IGNORED:
                continue
            self.dataset[class_name] = []
            for file in os.listdir(os.path.join(self.path, class_name)):
                file_path = os.path.join(self.path, class_name, file)
                # Metadata is a dict with the metadata of each file.
                specimen_number = self.metadata[file.split('/')[-1]]['specimen_number']
                if specimen_number not in self.specimens.keys():
                    self.specimens[specimen_number] = []

                if specimen_number not in self.dataset[class_name]:
                    self.dataset[class_name].append(specimen_number)

                if not file.endswith('NAME.jpg'):
                    img = Image.open(file_path)
                    width, height = img.size
                    if width >= 224 and height >= 224:
                        self.specimens[specimen_number].append(file_path)

        self.classes.sort()
        dataset = self.get_dataset(split)
        self.train = dataset['train']
        self.test = dataset['test']

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    def drop_specimens_by_class(self, split):
        specimens = self.get_specimens_in_train_by_class()
        specimens_kept = []
        for class_specimens in specimens.values():
            number_of_specimens = len(class_specimens)
            indexes = random.sample(range(0, number_of_specimens), int(split * number_of_specimens))
            for i in indexes:
                specimens_kept.append(class_specimens[i])
        return specimens_kept

    def all_dataset(self, train_size=1.0, specimen_size=1.0):

        train_data = self.train

        if specimen_size < 1:
            specimens = self.drop_specimens_by_class(specimen_size)

            train = []
            labels = []

            j = 0
            for file in train_data['files']:
                specimen_number = self.metadata[file.split('/')[-1]]['specimen_number']
                if specimen_number in specimens:
                    train.append(train_data['files'][j])
                    labels.append(train_data['labels'][j])
                j += 1

            train_data = {
                'files': train,
                'labels': labels
            }

        elif train_size < 1:
            train = []
            labels = []
            data = self.get_train_data_by_class()
            for images in data.values():
                number_of_train_images = len(images)
                indexes = random.sample(range(0, number_of_train_images), int(train_size * number_of_train_images))
                for i in indexes:
                    j = images[i]
                    train.append(train_data['files'][j])
                    labels.append(train_data['labels'][j])

            train_data = {
                'files': train,
                'labels': labels
            }

        return {
            'train': train_data,
            'test': self.test
        }

    def get_trees_in_train(self):
        specimens = []
        for file in self.train['files']:
            specimen_number = self.metadata[file.split('/')[-1]]['specimen_number']
            if specimen_number not in specimens:
                specimens.append(specimen_number)

        return specimens

    def get_train_data_by_class(self):
        images_by_class = {x: [] for x in self.classes}
        i = 0
        for file in self.train['files']:
            class_name = self.metadata[file.split('/')[-1]]['label']
            images_by_class[class_name].append(i)
            i += 1

        return images_by_class

    def get_specimens_in_train_by_class(self):
        specimens_by_class = {x: [] for x in self.classes}
        for file in self.train['files']:
            file_name = file.split('/')[-1]
            specimen_number = self.metadata[file_name]['specimen_number']
            class_name = self.metadata[file_name]['label']
            if specimen_number not in specimens_by_class[class_name]:
                specimens_by_class[class_name].append(specimen_number)

        return specimens_by_class

    def get_dataset(self, split):
        train, test = self.get_dataset_specimens(split)

        train_files = self.get_files(train)
        train_labels = self.get_labels(train_files)
        test_files = self.get_files(test)
        test_labels = self.get_labels(test_files)

        return {
            'train': {
                'files': train_files,
                'labels': train_labels,
            },
            'test': {
                'files': test_files,
                'labels': test_labels
            }}

    def get_dataset_specimens(self, split=1.):
        train = []
        test = []

        for class_name in self.classes:
            specimens = self.dataset[class_name]
            random.shuffle(specimens)
            split_point = int(split * len(specimens))
            train.extend(specimens[:split_point])
            test.extend(specimens[split_point:])

        return train, test

    def get_files(self, specimen_list):
        files = []
        for specimen_number in specimen_list:
            files.extend(self.specimens[specimen_number])
        return files

    def get_labels(self, files):
        labels = []
        for file in files:
            label = self.metadata[file.split('/')[-1]]['label']
            if label not in self.classes:
                print("Label not found")
                print("File " + file)
            labels.append(self.classes.index(label))
        return labels

    def _generate_k_fold_dataset(self, k):
        folds = [[] for _ in range(k)]
        for class_name in self.classes:
            specimens = self.dataset[class_name]
            specimens.sort(key=lambda x: int(self.specimens[x][0].split('/')[-1].split('_')[2]))
            for i, tree in enumerate(specimens):
                folds[i % k].append(tree)

        labels = [[] for _ in range(k)]
        files = [[] for _ in range(k)]
        for i, fold in enumerate(folds):
            files[i].extend(self.get_files(fold))
            labels[i].extend(self.get_labels(files[i]))

        return files, labels

    def get_k_fold_dataset(self, k):
        files, labels = self._generate_k_fold_dataset(k)

        dataset = {}

        for i in range(k):
            train_files = []
            train_labels = []
            for j in range(k):
                if j != i:
                    train_files.extend(files[j])
                    train_labels.extend(labels[j])
            dataset[i] = {
                'train': {
                    'files': train_files,
                    'labels': train_labels
                },
                'test': {
                    'files': files[i],
                    'labels': labels[i]
                }
            }

        return dataset
