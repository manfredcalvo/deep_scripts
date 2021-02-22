#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import json
from sklearn.utils.class_weight import compute_class_weight
from dataset.ImageDatasetGenerator import ImageDatasetGenerator
from dataset.generate_dataset import GenerateDataset
from pathlib import Path
import math


# In[2]:


def calculate_steps(dataset_size, batch_size):
    return math.ceil(dataset_size / batch_size)


# In[3]:


guyana_dataset_path = '/datadrive/notebooks/downloaded_data/full_data_top_23_0.1_resize_672_672'

with open('/datadrive/bdfr.metadata') as json_file:
        guyana_dataset_metadata = json.load(json_file)
        
generate_dataset_guyana = GenerateDataset(guyana_dataset_path, guyana_dataset_metadata)

generate_dataset_guyana.load_dataset('None')

dataset_guyana = generate_dataset_guyana.all_dataset()


# In[4]:


train_data_guyana = dataset_guyana['train']
val_data_guyana = dataset_guyana['val']
test_data_guyana = dataset_guyana['test']


# In[5]:


batch_size = 64
output_dim = (224, 224)


# In[6]:


train_image_generator = ImageDatasetGenerator(train_data_guyana['files'], train_data_guyana['labels'], batch_size, output_dim,
                                        guyana_dataset_metadata, train_mode=True, random_crop=True)
val_image_generator = ImageDatasetGenerator(val_data_guyana['files'], val_data_guyana['labels'], batch_size, output_dim,
                                        guyana_dataset_metadata, train_mode=False, random_crop=False)


# In[7]:


len(train_image_generator), len(val_image_generator)


# In[8]:


n_classes = generate_dataset_guyana.num_classes()


# In[3]:


np.bincount(train_flow.classes)


# In[4]:


class_weights = {i: weight for i, weight in enumerate(compute_class_weight('balanced', np.unique(train_flow.classes), train_flow.classes))}
n_classes = len(class_weights)
class_weights, n_classes


# In[9]:



'''
img_generator = image_dataset_from_directory('C:\\Users\\calvom\\ThesisMaster\\notebooks\\downloaded_data\\training_top_23_0.1_resize_672_672', 
                                             batch_size=32, 
                                             image_size=(224, 224), 
                                             shuffle=True, 
                                             label_mode='categorical')
'''

base_model = ResNet50(weights='imagenet', include_top=False)

input_img = Input((224, 224,3))

x = preprocess_input(input_img)

encoded_features = base_model(x)

x = GlobalAveragePooling2D()(encoded_features)
# let's add a fully-connected layer
x = Dense(2048)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(n_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=input_img, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional resnet50 layers
for layer in base_model.layers:
    layer.trainable = False

model.summary()


# In[10]:



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit_generator(train_image_generator, 
                    steps_per_epoch=calculate_steps(len(train_image_generator), batch_size),
                    epochs=20,
                    validation_steps=calculate_steps(len(val_image_generator), batch_size),
                    validation_data=val_image_generator)
#model.fit(train_flow, epochs=10, class_weight=class_weights)
model.save('/datadrive/bdfr_models/step_1_model')


# In[11]:


model = tf.keras.models.load_model('step_1_model')


# In[12]:


base_model = model.layers[3]


# In[13]:


for i, layer in enumerate(base_model.layers):
    print(i, layer.name)


# In[14]:


for layer in base_model.layers[:143]:
    layer.trainable = False
for layer in base_model.layers[143:]:
    layer.trainable = True


# In[15]:


from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[16]:


#model.fit(train_flow, epochs=10, validation_data=validation_flow, class_weight=class_weights)
model.fit(train_image_generator, 
                    steps_per_epoch=calculate_steps(len(train_image_generator), batch_size),
                    epochs=20,
                    validation_steps=calculate_steps(len(val_image_generator), batch_size),
                    validation_data=val_image_generator)

model.save('/datadrive/bdfr_models/step_2_model')

