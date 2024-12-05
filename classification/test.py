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
from keras._tf_keras.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras._tf_keras.keras.models import load_model
from PIL import Image, ImageEnhance

color_model = tf.keras.models.load_model('models/color_model_50.keras')
type_model = tf.keras.models.load_model('models/type_model_50.keras')
usage_model = tf.keras.models.load_model('models/usage_model_50.keras')
correct_answers = [
['Blazers','Black','Formal'],
['Blazers','Black','Formal'],
['Jeans','Blue','Casual'],
['Blazers','Brown','Formal'],
['Blazers','Beige','Formal'],
['Blazers','Grey','Formal'],
['Cardigan','Red','Casual'],
['Cardigan','White','Casual'],
['Cardigan','Blue','Casual'],
['Cardigan','Green','Casual'],
['Cardigan','Grey','Casual'],
['Dresses','Red','Casual'],
['Dresses','Black','Formal'],
['Dresses','Orange','Casual'],
['Dresses','Pink','Formal'],
['Dresses','Yellow','Casual'],
['Hoodie','Black','Casual'],
['Hoodie','White','Casual'],
['Hoodie','Yellow','Casual'],
['Hoodie','Red','Casual'],
['Hoodie','Brown','Casual'],
['Jackets','Grey','Casual'],
['Jackets','Brown','Formal'],
['Jackets','Brown','Formal'],
['Jackets','Brown','Casual'],
['Jackets','Pink','Casual'],
['Jeans','Blue','Casual'],
['Jeans','White','Casual'],
['Jeans','Blue','Casual'],
['Jeans','Blue','Casual'],
['Jumpsuit','Beige','Casual'],
['Jumpsuit','Grey','Casual'],
['Jumpsuit','Grey','Casual'],
['Jumpsuit','Yellow','Casual'],
['Jumpsuit','Blue','Casual'],
['Leggings','Green','Casual'],
['Leggings','Orange','Casual'],
['Leggings','Black','Casual'],
['Leggings','Blue','Casual'],
['Leggings','Pink','Casual'],
['Tshirts','Purple','Casual'],
['Tshirts','Green','Casual'],
['Tshirts','Orange','Casual'],
['Tshirts','Red','Casual'],
['Tshirts','Pink','Casual'],
['Trousers','Grey','Formal'],
['Trousers','Brown','Formal'],
['Trousers','Blue','Formal'],
['Trousers','Blue','Formal'],
['Trousers','Black','Formal'],
['Lounge Pants','Purple','Casual'],
['Lounge Pants','Green','Casual'],
['Lounge Pants','Orange','Casual'],
['Lounge Pants','White','Casual'],
['Lounge Pants','Yellow','Casual'],
['Shorts','Blue','Casual'],
['Shorts','Purple','Casual'],
['Shorts','Green','Casual'],
['Shorts','White','Casual'],
['Shorts','Green','Casual'],
['Skirts','Black','Casual'],
['Skirts','Blue','Casual'],
['Skirts','Orange','Casual'],
['Skirts','Pink','Casual'],
['Skirts','Brown','Formal'],
['Sweaters','Purple','Casual'],
['Sweaters','Yellow','Casual'],
['Sweaters','Red','Casual'],
['Sweaters','White','Casual'],
['Sweaters','Pink','Casual'],
['Tank','Purple','Casual'],
['Tank','White','Formal'],
['Tank','Green','Casual'],
['Tank','Orange','Casual'],
['Tank','Red','Casual'],
['Tops','Yellow','Formal'],
['Tops','Blue','Formal'],
['Tops','Grey','Formal'],
['Tops','Brown','Casual'],
['Tops','Purple','Formal'],
]
type_classes = ['Blazers',
'Cardigan',
'Dresses',
'Hoodie',
'Jackets',
'Jeans',
'Jumpsuit',
'Leggings',
'Lounge Pants',
'Shorts',
'Skirts',
'Sweaters',
'Tank',
'Tops',
'Trousers',
'Tshirts']
color_classes = ['Beige','Black','Blue','Brown','Green','Grey','Orange','Pink','Purple','Red','White','Yellow'
]
usage_classes = ['Casual','Formal']

correct_colors = 0
correct_types = 0
correct_usages = 0

datagen = ImageDataGenerator(
    fill_mode='nearest',
)
translation = 30

for i in range(1,81):
    img = load_img(f'test/{i}.png', target_size=(224,224), keep_aspect_ratio=True)
    img_array = img_to_array(img)
    brightness_factor = 2
    img_array = np.clip(img_array * brightness_factor, 0, 255)
    img_array = img_array/255.0
    input_image = np.expand_dims(img_array, axis=0)

    color_predictions = color_model.predict(input_image, verbose=0)[0]
    type_predictions = type_model.predict(input_image, verbose=0)[0]

    type_prediction = type_classes[np.argmax(type_predictions)]
    color_prediction = color_classes[np.argmax(color_predictions)]
    usage_prediction = 'Casual'

    if type_prediction in {'Dresses', 'Skirts', 'Tops', 'Trousers', 'Jackets'}:
        usage_predictions = usage_model.predict(input_image, verbose=0)[0]
        if usage_predictions[0]>0.5:
            usage_prediction= 'Formal'
    elif type_prediction == 'Blazers':
        usage_prediction= 'Formal'
    if(color_prediction==correct_answers[i-1][1]):
        correct_colors+=1
    if(type_prediction==correct_answers[i-1][0]):
        correct_types+=1
    if(usage_prediction==correct_answers[i-1][2]):
        correct_usages+=1
    
print(f'Type Accuracy: {correct_types/80.0}')
        

'''
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True 
)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.0,
    height_shift_range=0.0,
    zoom_range=0.0,
    horizontal_flip=True
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
df['baseColour'] = df['baseColour'].replace('Sea Green', 'Green')
df['baseColour'] = df['baseColour'].replace('Turquoise Blue', 'Blue')
df = df[['id', 'baseColour']] 
#tshirt_indices = df[df['articleType'] == 'Tshirts'].index[:num_drop]
#df = df.drop(tshirt_indices)

df = pd.get_dummies(df, columns=['baseColour']).astype(int)

unwanted_labels = [
'Teal',
'Multi',
'Lime Green',
'Gold',
'Silver',
'Fluorescent Green',
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
'Grey Melange'
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

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=False)
train_generator = DataGenerator(x_train, y_train, batch_size=64, augment = True, datagen=datagen)
test_generator = DataGenerator(x_test, y_test, batch_size=64)
model = tf.keras.models.load_model('colour_model_1.keras')

y_pred = model.predict(test_generator)


for i, row in enumerate(y_pred):
    total = 0
    arr_max = np.zeros_like(y_pred[i], dtype=int)
    arr_max[np.argmax(y_pred[i])] = 1
    if np.array_equal(arr_max, y_test[i]):
        total+=1
print(total)
print(len(y_pred))
print(y_test.shape)
print(y_pred.shape)
'''