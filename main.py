

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import tensorflow
import os
viral_pneumonia_path="COVID-19 Radiography Database/Viral Pneumonia/"
normal_path="COVID-19 Radiography Database/NORMAL/"
covid_path="COVID-19 Radiography Database/COVID-19/"
classes_label=[]
total_image_list=[]
for img in os.listdir(covid_path):
    temp_img=(cv2.imread(os.path.join(covid_path,img)))
    temp_img=cv2.resize(temp_img,(299,299))
    total_image_list.append((temp_img.astype('float16'))/255.0)

    classes_label.append(2) 
i=0
for img in os.listdir(normal_path):
    i=i+1
    if i<=209:
        temp_img=(cv2.imread(os.path.join(normal_path,img)))
        temp_img=cv2.resize(temp_img,(299,299))
        
        total_image_list.append((temp_img.astype('float16'))/255.0)

        classes_label.append(0) 
    else:
        break
  
i=0
for img in os.listdir(viral_pneumonia_path):
    i=i+1
    if i<=209:
        temp_img=(cv2.imread(os.path.join(viral_pneumonia_path,img)))
        temp_img=cv2.resize(temp_img,(299,299))
        
        total_image_list.append((temp_img.astype('float16'))/255.0)
        classes_label.append(1) 
    else:
        break

import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',input_shape=(299, 299, 3))
base_model.summary()
total_image_list=np.array(total_image_list)


classes_label=np.array(classes_label)
X, Y = shuffle(total_image_list, classes_label, random_state=0)
x_train ,x_test,y_train,y_test=train_test_split(X, Y, shuffle=False)

def get_images(x_train,y_train, augment=True, augment_size=200):

    train_size = x_train.shape[0]
    X=np.array(x_train)
    Y=np.array(y_train)
    if augment:

        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.05, 
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False, 
            data_format="channels_last")
        image_generator.fit(x_train, augment=True)

        randidx = np.random.randint(train_size, size=augment_size)
        x_augmented = x_train[randidx]
        y_augmented = y_train[randidx]
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                    batch_size=augment_size, shuffle=False).next()[0]
        # append augment data to trainset
        X = np.concatenate((X, x_augmented))
        Y = np.concatenate((Y, y_augmented))
        print(X.shape, Y.shape)
    return X, Y
X_train, Y_train = get_images(x_train,y_train, augment_size=800)

base_model_mod = base_model.output
base_model_mod = tf.keras.layers.GlobalAveragePooling2D()(base_model_mod)
base_model_mod=tf.keras.layers.Flatten()(base_model_mod)
# Add fully-connected layer
base_model_mod = tf.keras.layers.Dense(1024, activation='relu')(base_model_mod)
base_model_mod = tf.keras.layers.Dense(256, activation='relu')(base_model_mod)
base_model_mod = tf.keras.layers.Dense(3, activation='softmax')(base_model_mod)

model = tf.keras.Model(inputs=base_model.input, outputs=base_model_mod)
#model.summary()

model.compile(
loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#checkpoint = tf.keras.ModelCheckpoint('model_covid.h5', monitor='val_loss', save_best_only=True, verbose=1)
Y_train=tf.keras.utils.to_categorical(
    Y_train, num_classes=None, dtype='float32'
)
y_test=tf.keras.utils.to_categorical(
    y_test, num_classes=None, dtype='float32'
)
logOnCSV=tf.keras.callbacks.CSVLogger("C:/Users/VRLAB4_A/Documents/COVID19DetectAPP/log.csv")
easStop=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False
)
redLROnPlateau=tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0
)
saveBestOnly=tf.keras.callbacks.ModelCheckpoint(
    "C:/Users/VRLAB4_A/Documents/COVID19DetectAPP/best/", monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch'
)
tBoardMonitor=tf.keras.callbacks.TensorBoard(
    log_dir='C:/Users/VRLAB4_A/Documents/COVID19DetectAPP/logs/', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)
callbacksList=[logOnCSV,tBoardMonitor,saveBestOnly,redLROnPlateau,easStop]
history = model.fit(x=X_train, y=Y_train, epochs = 40, validation_data = (x_test, y_test),callbacks=callbacksList)

from joblib import dump, load
dump(history, "history.joblib")
dump(model, "model.joblib")