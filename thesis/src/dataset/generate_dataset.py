import json
import os
from PIL import Image
import random
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

class GenerateDataset:
    IGNORED = ['ERB', 'PEG', 'PID']

    def __init__(self, path, metadata):
        self.path = path
        self.specimens = {}
        self.dataset = {}
        self.classes = []
        self.train = {}
        self.test = {}
        self.val = {}
        self.metadata = metadata

    def load_dataset(self, existing_dataset, val_split=0.2, test_split=0.2):

        if existing_dataset != 'None':
            with open(existing_dataset, 'r') as dataset_file:
                loaded_dataset = json.load(dataset_file)
                self.train = loaded_dataset['train']
                self.test = loaded_dataset['test']
                self.val = loaded_dataset['val']
                self.classes = loaded_dataset['classes']
        else:
            for class_name in os.listdir(self.path):
                if class_name in self.IGNORED:
                    continue
                self.classes.append(class_name)
            self.load_new_dataset(val_split, test_split)

    def num_classes(self):
        return len(self.classes)

    def get_classes(self):
        return self.classes

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

    def load_new_dataset(self, val_split=0.2, test_split=0.2):
        for class_name in os.listdir(self.path):
            if class_name in self.IGNORED:
                continue
            self.dataset[class_name] = []
            for file in os.listdir(os.path.join(self.path, class_name)):
                file_path = Path(self.path, class_name, file)
                # Metadata is a dict with the metadata of each file.
                specimen_number = self.metadata[file_path.name]['specimen_number']
                if specimen_number not in self.specimens.keys():
                    self.specimens[specimen_number] = []
                if specimen_number not in self.dataset[class_name]:
                    self.dataset[class_name].append(specimen_number)
                if not file.endswith('NAME.jpg'):
                    img = Image.open(str(file_path))
                    width, height = img.size
                    if width >= 224 and height >= 224:
                        self.specimens[specimen_number].append(str(file_path))

        self.classes.sort()
        dataset = self.get_dataset(val_split, test_split)
        self.train = dataset['train']
        self.test = dataset['test']
        self.val = dataset['val']

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
                file_path = Path(file)
                specimen_number = self.metadata[file_path.name]['specimen_number']
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
            'val': self.val,
            'test': self.test
        }

    def get_trees_in_train(self):
        specimens = []
        for file in self.train['files']:
            file_path = Path(file)
            specimen_number = self.metadata[file_path.name]['specimen_number']
            if specimen_number not in specimens:
                specimens.append(specimen_number)

        return specimens

    def get_train_data_by_class(self):
        images_by_class = {x: [] for x in self.classes}
        i = 0
        for file in self.train['files']:
            file_path = Path(file)
            class_name = self.metadata[file_path.name]['label']
            images_by_class[class_name].append(i)
            i += 1

        return images_by_class

    def get_specimens_in_train_by_class(self):
        specimens_by_class = {x: [] for x in self.classes}
        for file in self.train['files']:
            file_name = Path(file).name
            specimen_number = self.metadata[file_name]['specimen_number']
            class_name = self.metadata[file_name]['label']
            if specimen_number not in specimens_by_class[class_name]:
                specimens_by_class[class_name].append(specimen_number)

        return specimens_by_class

    def get_dataset_specimens(self, val_split=0.2, test_split=0.2):
        train = []
        val = []
        test = []

        for class_name in self.classes:
            specimens = self.dataset[class_name]
            training_indexes, test_indexes, _, _ = train_test_split(np.arange(len(specimens)),
                                                                    np.arange(len(specimens)),
                                                                    test_size=test_split, random_state=1222)
            val_p = (val_split / (1 - test_split))

            train_indexes, val_indexes, _, _ = train_test_split(training_indexes, np.arange(len(training_indexes)),
                                                                test_size=val_p, random_state=213412)

            train.extend(specimens[specimen] for specimen in train_indexes)
            test.extend(specimens[specimen] for specimen in test_indexes)
            val.extend(specimens[specimen] for specimen in val_indexes)

        return train, val, test

    def get_dataset(self, val_split, test_split):
        train, val, test = self.get_dataset_specimens(val_split, test_split)

        train_files = self.get_files(train)
        train_labels = self.get_labels(train_files)
        val_files = self.get_files(val)
        val_labels = self.get_labels(val_files)
        test_files = self.get_files(test)
        test_labels = self.get_labels(test_files)

        return {
            'train': {
                'files': train_files,
                'labels': train_labels,
            },
            'val': {
                'files': val_files,
                'labels': val_labels,
            },
            'test': {
                'files': test_files,
                'labels': test_labels
            }}

    def get_files(self, specimen_list):
        files = []
        for specimen_number in specimen_list:
            files.extend(self.specimens[specimen_number])
        return files

    def get_labels(self, files):
        labels = []
        for file in files:
            file_path = Path(file)
            label = self.metadata[file_path.name]['label']
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
