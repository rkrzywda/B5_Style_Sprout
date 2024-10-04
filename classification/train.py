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

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('fashion-dataset/styles.csv')
df = df[df['masterCategory'] == 'Apparel']
df = df[['id', 'articleType', 'baseColour']] 
df.dropna(inplace=True)
num_drop = len(df)-(len(df)//32*32)
df = df[:-num_drop]

df = pd.get_dummies(df, columns=['articleType', 'baseColour']).astype(int)

unwanted_labels = [
'articleType_Rain Jacket',
'articleType_Rain Trousers',
'articleType_Jeggings',
'articleType_Baby Dolls',
'articleType_Bath Robe',
'articleType_Belts',
'articleType_Booties',
'articleType_Boxers',
'articleType_Bra',
'articleType_Briefs',
'articleType_Churidar',
'articleType_Jumpsuit',
'articleType_Kurta Sets',
'articleType_Lehenga Choli',
'articleType_Lounge Tshirts',
'articleType_Nehru Jackets',
'articleType_Patiala',
'articleType_Robe',
'articleType_Rompers',
'articleType_Salwar',
'articleType_Salwar and Dupatta',
'articleType_Sarees',
'articleType_Shapewear',
'articleType_Shrug',
'articleType_Stockings',
'articleType_Suspenders',
'articleType_Swimwear',
'articleType_Tights',
'baseColour_Burgundy',
'baseColour_Coffee Brown',
'baseColour_Fluorescent Green',
'baseColour_Gold',
'baseColour_Khaki',
'baseColour_Lime Green',
'baseColour_Mauve',
'baseColour_Mushroom Brown',
'baseColour_Mustard',
'baseColour_Nude',
'baseColour_Rose',
'baseColour_Rust',
'baseColour_Sea Green',
'baseColour_Silver',
'baseColour_Skin',
'baseColour_Tan',
'baseColour_Taupe',
'baseColour_Teal',
'baseColour_Turquoise Blue',
'usage_Ethnic',
'usage_Party',
'usage_Travel',
'articleType_Dupatta',
'usage_Smart Casual',
'articleType_Kurtas',
'articleType_Kurtis',
'articleType_Clothing Set',
'articleType_Waistcoat'
]
df = df.drop(columns=unwanted_labels, errors='ignore')
df = df[(df.drop(columns=['id']).sum(axis=1) > 0)]
label_counts = df.drop(columns=['id']).sum()
print(label_counts)


def preprocess_image(image_id, target_size=(135,180)):
    image_path = f'fashion-dataset/images/{image_id}.jpg'
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array/255.0  
    return img_array

class DataGenerator(Sequence):
    def __init__(self, image_ids, labels, batch_size=32, target_size=(135, 180),augment=False, datagen=None, **kwargs):
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
        labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        images = np.array([preprocess_image(image_id, self.target_size) for image_id in batch_x])
        if self.augment and self.datagen:
            images = np.array([self.datagen.random_transform(image) for image in images])

        return images, labels
    
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

X = np.stack(df['id'].values)
y = df.drop(columns=['id']).values

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


train_generator = DataGenerator(x_train, y_train, batch_size=32)
test_generator = DataGenerator(x_test, y_test, batch_size=32)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 5), activation='relu', input_shape=(135, 180, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (3, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (3, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(256, (3, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(df.shape[1] - 1,activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=50, verbose = 2)

#test_loss, test_acc = model.evaluate(test_generator, verbose=2)

#print('\nTest accuracy:', test_acc)

