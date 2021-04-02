#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.ImageDatasetGenerator import ImageDatasetGenerator
from dataset.generate_dataset import GenerateDataset
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from flip_gradient import flip_gradient
from utils import *
import json

# ### Preparing source and target data

# In[ ]:


source_dataset_path = '/datadrive/barknet_1.0'
target_dataset_path = '/datadrive/notebooks/downloaded_data/full_data_top_23_0.1_resize_672_672/'

with open('/datadrive/barknet.metadata') as json_file:
    source_dataset_metadata = json.load(json_file)

with open('/datadrive/bdfr.metadata') as json_file:
    target_dataset_metadata = json.load(json_file)

source_generate_dataset = GenerateDataset(source_dataset_path, source_dataset_metadata)
target_generate_dataset = GenerateDataset(target_dataset_path, target_dataset_metadata)

source_generate_dataset.load_dataset('None')
target_generate_dataset.load_dataset('None')

source_dataset_splits = source_generate_dataset.all_dataset()
target_dataset_splits = target_generate_dataset.all_dataset()

# In[ ]:


source_train_data = source_dataset_splits['train']
source_val_data = source_dataset_splits['val']
source_test_data = source_dataset_splits['test']

target_train_data = target_dataset_splits['train']
target_val_data = target_dataset_splits['val']
target_test_data = target_dataset_splits['test']

# In[ ]:


batch_size = 128
output_dim = (224, 224)

# In[ ]:


augmentation_params = {'vertical_flip': True, 'rotation': True, 'channel_shift_range': 10, 'shear_range': 0.2,
                       'horizontal_flip': True, 'random_crop': True}

source_train_image_generator = ImageDatasetGenerator(source_train_data['files'], source_train_data['labels'],
                                                     batch_size // 2, output_dim,
                                                     source_dataset_metadata, train_mode=True, **augmentation_params)

source_val_image_generator = ImageDatasetGenerator(source_val_data['files'], source_val_data['labels'], batch_size // 2,
                                                   output_dim,
                                                   source_dataset_metadata, train_mode=False, random_crop=False)

target_train_image_generator = ImageDatasetGenerator(target_train_data['files'], target_train_data['labels'],
                                                     batch_size // 2, output_dim,
                                                     target_dataset_metadata, train_mode=True, **augmentation_params)

target_val_image_generator = ImageDatasetGenerator(target_val_data['files'], target_val_data['labels'], batch_size // 2,
                                                   output_dim,
                                                   target_dataset_metadata, train_mode=False, random_crop=False)

# In[ ]:


source_ds_train = tf.data.Dataset.from_generator(lambda: source_train_image_generator,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=([batch_size // 2, output_dim[0], output_dim[1], 3],
                                                                [batch_size // 2, 20])
                                                 )

target_ds_train = tf.data.Dataset.from_generator(lambda: source_train_image_generator,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=([batch_size // 2, output_dim[0], output_dim[1], 3],
                                                                [batch_size // 2, 20])
                                                 )

source_gen_train = iter(source_ds_train)
target_gen_train = iter(target_ds_train)

# In[ ]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, InputLayer, GlobalAveragePooling2D, Flatten, Dense, \
    BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Model


class MNISTModel(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self, extractor_model=None):
        self.extractor_model = extractor_model
        self._build_model()

    def _build_model(self):
        input_shape = (224, 224, 3)
        output_pooling = False
        output_shape = 20

        # Feature extractor definition.

        input_features = Input(input_shape)

        if self.extractor_model is None:
            output_1 = Conv2D(filters=32, kernel_size=5, padding='SAME')(input_features)
            output_2 = MaxPooling2D(2, strides=2, padding='SAME')(output_1)
            output_3 = Conv2D(filters=48, kernel_size=5, padding='SAME')(output_2)
            output_4 = MaxPooling2D(2, strides=2, padding='SAME')(output_3)
            if output_pooling:
                final_output = GlobalAveragePooling2D()(output_4)
            else:
                final_output = Flatten()(output_4)
        else:

            base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
            output_1 = preprocess_input(input_features)
            output_2 = base_model(output_1, training=False)
            final_output = GlobalAveragePooling2D()(output_2)

        self.feature_extractor_model = Model(inputs=input_features, outputs=final_output)
        print("Feature extractor model summary: ")
        self.feature_extractor_model.summary()

        # Source classifier definition.
        source_output_1 = Dense(units=100)(final_output)
        source_output_2 = Dense(units=100)(source_output_1)
        source_final_output = Dense(units=output_shape, activation='softmax')(source_output_2)
        self.source_classifier_model = Model(inputs=input_features, outputs=source_final_output)
        print("Source classifier model summary: ")
        print(self.source_classifier_model.summary())

        # Domain classifier definition.
        # kernel_regularizer=tf.keras.regularizers.l2()
        domain_output_1 = Dense(units=100)(final_output)
        domain_final_output = Dense(units=1,
                                    activation='sigmoid')(domain_output_1)
        self.domain_classifier_model = Model(inputs=input_features, outputs=domain_final_output)
        print("Domain classifier model summary: ")
        print(self.domain_classifier_model.summary())

        # Combined model definition (source classifier + domain classifier).
        self.combined_model = Model(inputs=input_features, outputs=[source_final_output, domain_final_output])

        print("Combined model summary: ")
        print(self.combined_model.summary())


# In[ ]:


model = MNISTModel(extractor_model='resnet_50')

# In[ ]:


# Build optimizers
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, SparseCategoricalAccuracy

domain_optimizer = SGD(momentum=0.9)
# domain_optimizer = Adam(1e-4)
combined_model_optimizer = SGD(momentum=0.9)
# combined_model_optimizer = Adam(1e-4)

metrics = ["accuracy"]


# model.domain_classifier_model.compile(loss='binary_crossentropy', optimizer=domain_optimizer, metrics=metrics)

# model.combined_model.compile(loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
#                             optimizer=combined_model_optimizer, metrics=metrics)


# In[ ]:


def custom_train_on_batch(X, y, model, optimizer, loss_function, metrics_function=None, trainable_variables=None):
    with tf.GradientTape() as tape:
        pred = model(X)
        loss = loss_function(y, pred)
        metrics_results = {}
        if metrics_function:
            metrics_results = metrics_function(y, pred)
    if not trainable_variables:
        trainable_variables = model.trainable_variables
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    results = {'loss': loss}
    results.update(metrics_results)
    return results


# In[ ]:


num_steps = 8600

domain_labels = np.vstack([np.tile([1.], [batch_size // 2, 1]),
                           np.tile([0.], [batch_size // 2, 1])])

opposite_domain_labels = np.vstack([np.tile([0.], [batch_size // 2, 1]),
                                    np.tile([1.], [batch_size // 2, 1])])

sample_weights_class = np.array(([1] * (batch_size // 2) + [0] * (batch_size // 2)))

sample_weights_adversarial = np.ones((batch_size,))

weights = [sample_weights_class, sample_weights_adversarial]

domain_binary_metric = BinaryAccuracy()
source_classifier_metric = SparseCategoricalAccuracy()
source_classifier_cum_loss = tf.keras.metrics.SparseCategoricalCrossentropy()
target_classifier_cum_loss = tf.keras.metrics.SparseCategoricalCrossentropy()
target_classifier_metric = SparseCategoricalAccuracy()


def custom_loss_function(y_true, y_pred):
    sample_weight_class = weights[0]
    sample_weight_domain = weights[1]
    y_class = y_true[0]
    y_domain = y_true[1]
    y_pred_class = y_pred[0]
    y_pred_domain = y_pred[1]
    classifier_loss = CategoricalCrossentropy()(y_class, y_pred_class, sample_weight=sample_weight_class)
    domain_loss = BinaryCrossentropy()(y_domain, y_pred_domain, sample_weight=sample_weight_domain)
    return classifier_loss + -1 * domain_loss


def domain_metrics_functions(y, pred):
    domain_binary_metric.update_state(y, pred)
    result = domain_binary_metric.result()
    return {domain_binary_metric.name: result}


def combine_model_metrics_functions(y, pred):
    source_classifier_metric.update_state(y[0], pred[0], weights[0])
    source_result = source_classifier_metric.result()

    # target_classifier_metric.update_state(y[0], pred[0], 1 - weights[0])
    # target_result = target_classifier_metric.result()

    source_classifier_cum_loss.update_state(y[0], pred[0], weights[0])
    source_loss_result = source_classifier_cum_loss.result()

    # target_classifier_cum_loss.update_state(y[0], pred[0], 1 - weights[0])
    # target_loss_result = target_classifier_cum_loss.result()

    return {f'source_{source_classifier_metric.name}': source_result,
            # f'target_{target_classifier_metric.name}': target_result,
            f'source_{source_classifier_cum_loss.name}': source_loss_result, }
    # f'target_{target_classifier_cum_loss.name}': target_loss_result}


loss_print = 1

# Training in adversarial manner.
for i in range(num_steps):

    p = float(i) / num_steps
    lr = 0.01 / (1. + 10 * p) ** 0.75
    domain_optimizer.lr = lr
    combined_model_optimizer.lr = lr

    source_X, source_y = next(source_gen_train)
    target_X, target_y = next(target_gen_train)

    X = np.vstack([source_X, target_X])
    y = np.vstack([source_y, target_y])

    # Training domain classifier.

    domain_classifier_stats = custom_train_on_batch(X,
                                                    domain_labels,
                                                    model.domain_classifier_model,
                                                    domain_optimizer,
                                                    BinaryCrossentropy(),
                                                    metrics_function=domain_metrics_functions)

    # Training source classifier and feature extractor together to fool domain classifier.
    # We expect 0.5 accuracy for domain classifier.

    labels = [y, domain_labels]

    combined_model_stats = custom_train_on_batch(X,
                                                 labels,
                                                 model.combined_model,
                                                 combined_model_optimizer,
                                                 custom_loss_function,
                                                 metrics_function=combine_model_metrics_functions)

    if i % loss_print == 0:
        print(combined_model_stats)
        domain_loss = domain_classifier_stats['loss']
        domain_accuracy = domain_classifier_stats['binary_accuracy']
        combined_model_loss = combined_model_stats['loss']
        source_accuracy = combined_model_stats['source_sparse_categorical_accuracy']
        # target_accuracy = combined_model_stats['target_sparse_categorical_accuracy']
        source_cum_loss = combined_model_stats['source_sparse_categorical_crossentropy']
        # target_cum_loss = combined_model_stats['target_sparse_categorical_crossentropy']
        print('---------------------------------------------------')
        print(f'Iteration: {i}')
        print(f'Domain Classifier Loss: {domain_loss}')
        print(f'Domain Classifier Accuracy: {domain_accuracy}')
        print(f'Combined model Loss: {combined_model_loss}')
        print(f'Source Classifier Loss: {source_cum_loss}')
        # print(f'Target Classifier Loss: {target_cum_loss}')
        print(f'Source Accuracy: {source_accuracy}')
        # print(f'Target Accuracy: {target_accuracy}')
        print(f'Actual learning rate: {lr}')
        print('---------------------------------------------------')
        print()

# In[ ]:


# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
# dann_tsne = tsne.fit_transform(dann_emb)
# dann_tsne.shape


# In[ ]:


# plot_embedding(source_only_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Source only')
# plot_embedding(dann_tsne, combined_test_labels.flatten(), combined_test_domain[:, 0], 'Domain Adaptation')


# In[ ]:


# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
# source_only_tsne = tsne.fit_transform(source_only_emb)

# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
# dann_tsne = tsne.fit_transform(dann_emb)

# plot_embedding(source_only_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Source only')
# plot_embedding(dann_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Domain Adaptation')
