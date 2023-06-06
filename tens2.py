import numpy as np 
import os 
import pandas as pd 
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
  
a = pd.read_csv('Dataset\chinese_mnist.csv')
filename = a[['suite_id', 'sample_id', 'code']].values
img = [cv2.imread(f"Dataset\data\input_{suite_id}_{sample_id}_{code}.jpg") for suite_id, sample_id, code in filename]
lbl = [ [x - 1] for x in a['code'].values ]
img = np.array(img)[:,:,:,0]
lbl = np.array(lbl)
img = img.reshape(15000,64,64,1)
img = img.astype("float32")
img /= 255
img_train, img_test, lbl_train, lbl_test = train_test_split(img, lbl)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(64, 64, 1)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(15))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
eval = model.fit(img_train,  lbl_train, epochs=8,validation_data=(img_test, lbl_test))
loss, accuracy = model.evaluate(img_test,lbl_test)
print(accuracy)