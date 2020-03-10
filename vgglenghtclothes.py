

from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow==2.0

"""
Created on Mon Oct  7 16:24:24 2019

@author: Artur Maia Pereira
"""

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

import pandas as pd
import numpy as np

#VGG to predict the skirt length (short, midi, long, other(not identified)) using a personal database annotated manually with 634 images
dataset = pd.read_csv('LabeledData.csv', encoding='latin-1')

#The dataframe has several columns, we need to select the length column
df_lenght = (dataset.iloc[:634, [0,2]])

num_classes = 3
INP = 224 #Image dimension default
BATCHSIZE = 128

#Copy images to a virtual machine, improving google colab performance 
!cp -r "/content/drive/My Drive/CNNClothes/" ./img

#Load Images
#data_directory = 'drive/My Drive/CNNClothes/'
data_directory = './img'
datagen=ImageDataGenerator(validation_split=0.30,
                           rescale=1./255)
            
#70% of images
train_generator=datagen.flow_from_dataframe(
        dataframe = df_lenght,
        directory = data_directory,
        x_col ='images__path',
        y_col ='lenght',
        batch_size =BATCHSIZE,
        class_mode="categorical",
        color_mode="rgb",
        classes=["short", "long", "midi"],
        target_size=(INP,INP),
        subset='training')

#30% of images
test_generator=datagen.flow_from_dataframe(
        dataframe = df_lenght,
        directory = data_directory,
        x_col ='images__path',
        y_col ='lenght',
        batch_size =BATCHSIZE,
        class_mode="categorical",
        color_mode="rgb",
        classes=["short", "long", "midi"],
        target_size=(INP,INP),
        subset='validation')


#%% VGG19

#Load Vgg without input layer
model_vgg19_conv = VGG19(weights='imagenet', include_top=False, classes = num_classes)
model_vgg19_conv.trainable = False

print(model_vgg19_conv.summary())

#Create your own input format    
keras_input = Input(shape=(INP, INP, 3), name = 'image_input')
    
#Use the generated model 
output_vgg19_conv = model_vgg19_conv(keras_input)
    
#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg19_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
#Create your own model 
pretrained_model = Model(inputs=keras_input, outputs=x)

#optimizer with default values
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

pretrained_model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy', metrics.categorical_accuracy])
es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, restore_best_weights = True, patience = 4)

history = pretrained_model.fit_generator(train_generator,
                                   steps_per_epoch= (train_generator.samples) // BATCHSIZE,
                                   epochs = 50,
                                   validation_data=test_generator,
                                   workers=4,
                                   callbacks = [es]
                         )


#%% VGG16

model_vgg16_conv = VGG16(weights='imagenet', include_top=False, classes = 4)

model_vgg16_conv.trainable = False

#Create your own input format
keras_input = Input(shape=(INP, INP, 3), name = 'image_input')
    
#Use the generated model 
output_vgg16_conv = model_vgg16_conv(keras_input)
    
#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
#Create your own model 
pretrained_model = Model(inputs=keras_input, outputs=x)    

print(pretrained_model.summary())

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

pretrained_model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy', metrics.categorical_accuracy])
es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, restore_best_weights = True, patience = 3)

history = pretrained_model.fit_generator(train_generator,
                                   steps_per_epoch= (train_generator.samples) // BATCHSIZE,
                                   epochs = 50,
                                   validation_data=test_generator,
                                   workers=4,
                                   callbacks = [es]
                         )

