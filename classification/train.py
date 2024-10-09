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
from keras.applications import ResNet50
from keras._tf_keras.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True 
)

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('fashion-dataset/styles.csv')
df = df[df['masterCategory'] == 'Apparel']
df = df[['id', 'articleType', 'baseColour', 'usage']] 
df1 = pd.read_csv('archive/blazers.csv')
df1.dropna(inplace=True)
df = pd.concat([df,df1])
df.dropna(inplace=True)
num_drop = len(df)-(len(df)//64*64)
df['articleType'] = df['articleType'].replace('Shirts', 'Tops')
df['articleType'] = df['articleType'].replace('Jeggings', 'Jeans')
df['articleType'] = df['articleType'].replace('Camisoles', 'Innerwear Vests')
df['articleType'] = df['articleType'].replace('Lounge Shorts', 'Shorts')
df['articleType'] = df['articleType'].replace('Track Pants', 'Lounge Pants')
df['articleType'] = df['articleType'].replace('Innerwear Vests', 'Tank Tops')
df['articleType'] = df['articleType'].replace('Tunics', 'Tops')
df['baseColour'] = df['baseColour'].replace('Off White', 'Beige')
df['baseColour'] = df['baseColour'].replace('Cream', 'Beige')
df['baseColour'] = df['baseColour'].replace('Khaki', 'Beige')
df['baseColour'] = df['baseColour'].replace('Charcoal', 'Grey')
df['baseColour'] = df['baseColour'].replace('Olive', 'Green')
df['baseColour'] = df['baseColour'].replace('Lavender', 'Purple')
df['baseColour'] = df['baseColour'].replace('Mauve', 'Purple')
df['baseColour'] = df['baseColour'].replace('Maroon', 'Red')
df['baseColour'] = df['baseColour'].replace('Burgundy', 'Red')
df['baseColour'] = df['baseColour'].replace('Magenta', 'Pink')
df['baseColour'] = df['baseColour'].replace('Peach', 'Orange')
df['baseColour'] = df['baseColour'].replace('Coffee Brown', 'Brown')
df['baseColour'] = df['baseColour'].replace('Mushroom Brown', 'Brown')
df['baseColour'] = df['baseColour'].replace('Mustard', 'Yellow')
df['baseColour'] = df['baseColour'].replace('Nude', 'Beige')
df['baseColour'] = df['baseColour'].replace('Tan', 'Beige')
df['baseColour'] = df['baseColour'].replace('Taupe', 'Beige')
df['baseColour'] = df['baseColour'].replace('Skin', 'Beige')
df['baseColour'] = df['baseColour'].replace('Rose', 'Pink')
df['baseColour'] = df['baseColour'].replace('Rust', 'Red')
df['baseColour'] = df['baseColour'].replace('Sea Green', 'Teal')
df['baseColour'] = df['baseColour'].replace('Turquoise Blue', 'Teal')
df = df[['id', 'articleType']] 
#tshirt_indices = df[df['articleType'] == 'Tshirts'].index[:num_drop]
#df = df.drop(tshirt_indices)

df = pd.get_dummies(df, columns=['articleType']).astype(int)

unwanted_labels = [
'baseColour_Lime Green',
'baseColour_Gold',
'baseColour_Silver',
'baseColour_Fluorescent Green',
'articleType_Shrug',
'articleType_Rain Jacket',
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
'articleType_Stockings',
'articleType_Suspenders',
'articleType_Swimwear',
'articleType_Tights',
'usage_Ethnic',
'usage_Party',
'usage_Travel',
'articleType_Dupatta',
'usage_Smart Casual',
'articleType_Kurtas',
'articleType_Kurtis',
'articleType_Clothing Set',
'articleType_Waistcoat',
'articleType_Tracksuits',
'articleType_Night suits',
'articleType_Nightdress',
'baseColour_Grey Melange'
]
df = df.drop(columns=unwanted_labels, errors='ignore')
df = df[(df.drop(columns=['id']).sum(axis=1) > 0)]
label_counts = df.drop(columns=['id']).sum()
print(label_counts)

def preprocess_image(image_id, target_size=(224,224)):
    image_path = f'fashion-dataset/images/{image_id}.jpg'
    img = load_img(image_path, target_size=target_size, keep_aspect_ratio=True)
    img_array = img_to_array(img)
    img_array = img_array/255.0  
    return img_array

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
y = df.drop(columns=['id']).values

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


train_generator = DataGenerator(x_train, y_train, batch_size=32)
test_generator = DataGenerator(x_test, y_test, batch_size=32)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224, 3))

base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(df.shape[1] - 1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

history = model.fit(
    train_generator, 
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stopping],
    verbose=2
)

model.save('type_model.keras')







early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True 
)

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('fashion-dataset/styles.csv')
df = df[df['masterCategory'] == 'Apparel']
df = df[['id', 'articleType', 'baseColour', 'usage']] 
df1 = pd.read_csv('archive/blazers.csv')
df1.dropna(inplace=True)
df = pd.concat([df,df1])
df.dropna(inplace=True)
num_drop = len(df)-(len(df)//64*64)
df['articleType'] = df['articleType'].replace('Shirts', 'Tops')
df['articleType'] = df['articleType'].replace('Jeggings', 'Jeans')
df['articleType'] = df['articleType'].replace('Camisoles', 'Innerwear Vests')
df['articleType'] = df['articleType'].replace('Lounge Shorts', 'Shorts')
df['articleType'] = df['articleType'].replace('Track Pants', 'Lounge Pants')
df['articleType'] = df['articleType'].replace('Innerwear Vests', 'Tank Tops')
df['articleType'] = df['articleType'].replace('Tunics', 'Tops')
df['baseColour'] = df['baseColour'].replace('Off White', 'Beige')
df['baseColour'] = df['baseColour'].replace('Cream', 'Beige')
df['baseColour'] = df['baseColour'].replace('Khaki', 'Beige')
df['baseColour'] = df['baseColour'].replace('Charcoal', 'Grey')
df['baseColour'] = df['baseColour'].replace('Olive', 'Green')
df['baseColour'] = df['baseColour'].replace('Lavender', 'Purple')
df['baseColour'] = df['baseColour'].replace('Mauve', 'Purple')
df['baseColour'] = df['baseColour'].replace('Maroon', 'Red')
df['baseColour'] = df['baseColour'].replace('Burgundy', 'Red')
df['baseColour'] = df['baseColour'].replace('Magenta', 'Pink')
df['baseColour'] = df['baseColour'].replace('Peach', 'Orange')
df['baseColour'] = df['baseColour'].replace('Coffee Brown', 'Brown')
df['baseColour'] = df['baseColour'].replace('Mushroom Brown', 'Brown')
df['baseColour'] = df['baseColour'].replace('Mustard', 'Yellow')
df['baseColour'] = df['baseColour'].replace('Nude', 'Beige')
df['baseColour'] = df['baseColour'].replace('Tan', 'Beige')
df['baseColour'] = df['baseColour'].replace('Taupe', 'Beige')
df['baseColour'] = df['baseColour'].replace('Skin', 'Beige')
df['baseColour'] = df['baseColour'].replace('Rose', 'Pink')
df['baseColour'] = df['baseColour'].replace('Rust', 'Red')
df['baseColour'] = df['baseColour'].replace('Sea Green', 'Teal')
df['baseColour'] = df['baseColour'].replace('Turquoise Blue', 'Teal')
df = df[['id', 'baseColour']] 
#tshirt_indices = df[df['articleType'] == 'Tshirts'].index[:num_drop]
#df = df.drop(tshirt_indices)

df = pd.get_dummies(df, columns=['baseColour']).astype(int)

unwanted_labels = [
'baseColour_Lime Green',
'baseColour_Gold',
'baseColour_Silver',
'baseColour_Fluorescent Green',
'articleType_Shrug',
'articleType_Rain Jacket',
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
'articleType_Stockings',
'articleType_Suspenders',
'articleType_Swimwear',
'articleType_Tights',
'usage_Ethnic',
'usage_Party',
'usage_Travel',
'articleType_Dupatta',
'usage_Smart Casual',
'articleType_Kurtas',
'articleType_Kurtis',
'articleType_Clothing Set',
'articleType_Waistcoat',
'articleType_Tracksuits',
'articleType_Night suits',
'articleType_Nightdress',
'baseColour_Grey Melange'
]
df = df.drop(columns=unwanted_labels, errors='ignore')
df = df[(df.drop(columns=['id']).sum(axis=1) > 0)]
label_counts = df.drop(columns=['id']).sum()
print(label_counts)

def preprocess_image(image_id, target_size=(224,224)):
    image_path = f'fashion-dataset/images/{image_id}.jpg'
    img = load_img(image_path, target_size=target_size, keep_aspect_ratio=True)
    img_array = img_to_array(img)
    img_array = img_array/255.0  
    return img_array

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
y = df.drop(columns=['id']).values

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


train_generator = DataGenerator(x_train, y_train, batch_size=32)
test_generator = DataGenerator(x_test, y_test, batch_size=32)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224, 3))

base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(df.shape[1] - 1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

history = model.fit(
    train_generator, 
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stopping],
    verbose=2
)

model.save('colour_model.keras')










early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True 
)

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('fashion-dataset/styles.csv')
df = df[df['masterCategory'] == 'Apparel']
df = df[['id', 'articleType', 'baseColour', 'usage']] 
df1 = pd.read_csv('archive/blazers.csv')
df1.dropna(inplace=True)
df = pd.concat([df,df1])
df.dropna(inplace=True)
num_drop = len(df)-(len(df)//64*64)
df['articleType'] = df['articleType'].replace('Shirts', 'Tops')
df['articleType'] = df['articleType'].replace('Jeggings', 'Jeans')
df['articleType'] = df['articleType'].replace('Camisoles', 'Innerwear Vests')
df['articleType'] = df['articleType'].replace('Lounge Shorts', 'Shorts')
df['articleType'] = df['articleType'].replace('Track Pants', 'Lounge Pants')
df['articleType'] = df['articleType'].replace('Innerwear Vests', 'Tank Tops')
df['articleType'] = df['articleType'].replace('Tunics', 'Tops')
df['baseColour'] = df['baseColour'].replace('Off White', 'Beige')
df['baseColour'] = df['baseColour'].replace('Cream', 'Beige')
df['baseColour'] = df['baseColour'].replace('Khaki', 'Beige')
df['baseColour'] = df['baseColour'].replace('Charcoal', 'Grey')
df['baseColour'] = df['baseColour'].replace('Olive', 'Green')
df['baseColour'] = df['baseColour'].replace('Lavender', 'Purple')
df['baseColour'] = df['baseColour'].replace('Mauve', 'Purple')
df['baseColour'] = df['baseColour'].replace('Maroon', 'Red')
df['baseColour'] = df['baseColour'].replace('Burgundy', 'Red')
df['baseColour'] = df['baseColour'].replace('Magenta', 'Pink')
df['baseColour'] = df['baseColour'].replace('Peach', 'Orange')
df['baseColour'] = df['baseColour'].replace('Coffee Brown', 'Brown')
df['baseColour'] = df['baseColour'].replace('Mushroom Brown', 'Brown')
df['baseColour'] = df['baseColour'].replace('Mustard', 'Yellow')
df['baseColour'] = df['baseColour'].replace('Nude', 'Beige')
df['baseColour'] = df['baseColour'].replace('Tan', 'Beige')
df['baseColour'] = df['baseColour'].replace('Taupe', 'Beige')
df['baseColour'] = df['baseColour'].replace('Skin', 'Beige')
df['baseColour'] = df['baseColour'].replace('Rose', 'Pink')
df['baseColour'] = df['baseColour'].replace('Rust', 'Red')
df['baseColour'] = df['baseColour'].replace('Sea Green', 'Teal')
df['baseColour'] = df['baseColour'].replace('Turquoise Blue', 'Teal')
df = df[['id', 'usage']] 
#tshirt_indices = df[df['articleType'] == 'Tshirts'].index[:num_drop]
#df = df.drop(tshirt_indices)

df = pd.get_dummies(df, columns=['usage']).astype(int)

unwanted_labels = [
'baseColour_Lime Green',
'baseColour_Gold',
'baseColour_Silver',
'baseColour_Fluorescent Green',
'articleType_Shrug',
'articleType_Rain Jacket',
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
'articleType_Stockings',
'articleType_Suspenders',
'articleType_Swimwear',
'articleType_Tights',
'usage_Ethnic',
'usage_Party',
'usage_Travel',
'articleType_Dupatta',
'usage_Smart Casual',
'articleType_Kurtas',
'articleType_Kurtis',
'articleType_Clothing Set',
'articleType_Waistcoat',
'articleType_Tracksuits',
'articleType_Night suits',
'articleType_Nightdress',
'baseColour_Grey Melange'
]
df = df.drop(columns=unwanted_labels, errors='ignore')
df = df[(df.drop(columns=['id']).sum(axis=1) > 0)]
label_counts = df.drop(columns=['id']).sum()
print(label_counts)

def preprocess_image(image_id, target_size=(224,224)):
    image_path = f'fashion-dataset/images/{image_id}.jpg'
    img = load_img(image_path, target_size=target_size, keep_aspect_ratio=True)
    img_array = img_to_array(img)
    img_array = img_array/255.0  
    return img_array

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
y = df.drop(columns=['id']).values

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


train_generator = DataGenerator(x_train, y_train, batch_size=32)
test_generator = DataGenerator(x_test, y_test, batch_size=32)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224, 3))

base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(df.shape[1] - 1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

history = model.fit(
    train_generator, 
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stopping],
    verbose=2
)

model.save('usage_model.keras')