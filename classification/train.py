import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import Sequence
import sys

np.set_printoptions(threshold=sys.maxsize)
df = pd.read_csv('fashion-dataset/styles.csv', nrows = 1024)
df = df[df['masterCategory'] == 'Apparel']
df = df[['id', 'articleType', 'baseColour', 'season', 'usage']]
df.dropna(inplace=True)
num_drop = len(df)-(len(df)//32*32)
df = df[:-num_drop]

df = pd.get_dummies(df, columns=['articleType', 'baseColour', 'season', 'usage']).astype(int)
def preprocess_image(image_id, target_size=(135,180)):
    image_path = f'fashion-dataset/images/{image_id}.jpg'
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array/255.0  
    return img_array

class DataGenerator(Sequence):
    def __init__(self, image_ids, labels, batch_size=32, target_size=(135, 180), **kwargs):
        super().__init__(**kwargs)
        self.image_ids = image_ids
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return len(self.image_ids)//self.batch_size

    def __getitem__(self, index):
        batch_x = self.image_ids[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        images = np.array([preprocess_image(image_id, self.target_size) for image_id in batch_x])
        return images, labels

X = np.stack(df['id'].values)
y = df.drop(columns=['id']).values
print(y)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

train_generator = DataGenerator(x_train, y_train, batch_size=32)
test_generator = DataGenerator(x_test, y_test, batch_size=32)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(135, 180, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(df.shape[1] - 1,activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10, verbose = 2)

#test_loss, test_acc = model.evaluate(test_generator, verbose=2)

#print('\nTest accuracy:', test_acc)

'''
tyring to understand what each of these methods are doing to understand
which method works best

'''