import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import Sequence
import sys
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.applications import ResNet50V2
from keras._tf_keras.keras.callbacks import EarlyStopping,ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True 
)

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('usage_subset.csv')
df.dropna(inplace=True)

df = pd.get_dummies(df, columns=['usage']).astype(int)
label_counts = df.drop(columns=['id']).sum()
print(label_counts)

def preprocess_image(image_id, target_size=(224,224)):
    image_path = f'usage_subset/{image_id}.jpg'
    img = load_img(image_path, target_size=target_size, keep_aspect_ratio=True)
    img_array = img_to_array(img)
    img_array = img_array/255.0 

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_jittered = tf.image.random_brightness(img_tensor, max_delta=0.1)
    img_jittered = tf.image.random_contrast(img_jittered, lower=0.9, upper=1.1)
    img_jittered = tf.image.random_saturation(img_jittered, lower=0.9, upper=1.1)
    img_jittered = tf.image.random_hue(img_jittered, max_delta=0.1)
    img_normalized = tf.clip_by_value(img_jittered, 0.0, 1.0)
    return img_normalized

class DataGenerator(Sequence):
    def __init__(self, image_ids, labels, batch_size=64, target_size=(224,224),augment=False, datagen=None, **kwargs):
        super().__init__(**kwargs)
        self.image_ids = image_ids
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.datagen = datagen

    def __len__(self):
        return len(self.image_ids)//self.batch_size

    def __getitem__(self, index):
        batch_x = self.image_ids[index*self.batch_size:(index+1)*self.batch_size]
        curr_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        images = np.array([preprocess_image(image_id, self.target_size) for image_id in batch_x])
        if self.augment and self.datagen:
            images = np.array([self.datagen.random_transform(image) for image in images])

        return images, curr_labels


X = np.stack(df['id'].values)
y = df['usage_Formal'].values

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest',
)

train_generator = DataGenerator(x_train, y_train, batch_size=64, augment=True, datagen=datagen)
test_generator = DataGenerator(x_test, y_test, batch_size=64)

base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224,224, 3))

base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['binary_accuracy']
              )

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_lr=0.000001
)

history = model.fit(
    train_generator, 
    epochs=100,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

model.save('usage_model_50_2.keras')