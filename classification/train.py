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
from keras.applications import ResNet101V2
from keras._tf_keras.keras.callbacks import EarlyStopping,ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True 
)

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('fashion-dataset/styles.csv')
df = df[df['masterCategory'] == 'Apparel']
df = df[['id', 'articleType']] 
df1 = pd.read_csv('archive/blazers.csv')
df = pd.concat([df,df1])
df2 = pd.read_csv('extra.csv')
df = pd.concat([df,df2])
num_drop = len(df)-(len(df)//64*64)
df['articleType'] = df['articleType'].replace('Anorak', 'Jackets')
df['articleType'] = df['articleType'].replace('Jacket', 'Jackets')
df['articleType'] = df['articleType'].replace('Bomber', 'Jackets')
df['articleType'] = df['articleType'].replace('Blazer', 'Blazers')
df['articleType'] = df['articleType'].replace('Dress', 'Dresses')
df['articleType'] = df['articleType'].replace('Tee', 'Tshirts')
df['articleType'] = df['articleType'].replace('Blouse', 'Tops')
df['articleType'] = df['articleType'].replace('Trunk', 'Trunks')
df['articleType'] = df['articleType'].replace('Trunks', 'Shorts')
df['articleType'] = df['articleType'].replace('Button-Down', 'Tops')
df['articleType'] = df['articleType'].replace('Caftan', 'Dresses')
df['articleType'] = df['articleType'].replace('Camisoles', 'Tank')
df['articleType'] = df['articleType'].replace('Top', 'Tops')
df['articleType'] = df['articleType'].replace('Chinos', 'Trousers')
df['articleType'] = df['articleType'].replace('Coat', 'Jacket')
df['articleType'] = df['articleType'].replace('Culottes', 'Trousers')
df['articleType'] = df['articleType'].replace('Cutoffs', 'Shorts')
df['articleType'] = df['articleType'].replace('Flannel', 'Tops')
df['articleType'] = df['articleType'].replace('Henley', 'Tops')
df['articleType'] = df['articleType'].replace('Jacket', 'Jackets')
df['articleType'] = df['articleType'].replace('Parka', 'Jackets')
df['articleType'] = df['articleType'].replace('Jeggings', 'Jeans')
df['articleType'] = df['articleType'].replace('Joggers', 'Lounge Pants')
df['articleType'] = df['articleType'].replace('Track Pants', 'Lounge Pants')
df['articleType'] = df['articleType'].replace('Lounge Shorts', 'Shorts')
df['articleType'] = df['articleType'].replace('Sweatshorts', 'Shorts')
df['articleType'] = df['articleType'].replace('Onesie', 'Jumpsuit')
df['articleType'] = df['articleType'].replace('Jersey', 'Tshirts')
df['articleType'] = df['articleType'].replace('Turtleneck', 'Tops')
df['articleType'] = df['articleType'].replace('Sweatshirts', 'Hoodie')
df['articleType'] = df['articleType'].replace('Sweatpants', 'Lounge Pants')
df['articleType'] = df['articleType'].replace('Kaftan', 'Lounge Pants')
df['articleType'] = df['articleType'].replace('Peacoat', 'Jacket')
df['articleType'] = df['articleType'].replace('Shirts', 'Tops')
df['articleType'] = df['articleType'].replace('Skirt', 'Skirts')
df['articleType'] = df['articleType'].replace('Jacket', 'Jackets')
df['articleType'] = df['articleType'].replace('Nightdress', 'Dresses')
df['articleType'] = df['articleType'].replace('Sweater', 'Sweaters')
df['articleType'] = df['articleType'].replace('Rompers', 'Dresses')
df['articleType'] = df['articleType'].replace('Romper', 'Dresses')

df = df[['id', 'articleType']] 
df.dropna(inplace=True)

df = pd.get_dummies(df, columns=['articleType']).astype(int)
indices_to_drop = df[df['articleType_Blazers'] == 1].sample(n=6000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Cardigan'] == 1].sample(n=12000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Dresses'] == 1].sample(n=79000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Jackets'] == 1].sample(n=12500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Jeans'] == 1].sample(n=6500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Jumpsuit'] == 1].sample(n=1000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Lounge Pants'] == 1].sample(n=6500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Shorts'] == 1].sample(n=22000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Skirts'] == 1].sample(n=13600, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Sweaters'] == 1].sample(n=12000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Tank'] == 1].sample(n=14000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Tshirts'] == 1].sample(n=38500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Tops'] == 1].sample(n=35000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Leggings'] == 1].sample(n=3500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Jumpsuit'] == 1].sample(n=3500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Hoodie'] == 1].sample(n=2500, random_state=42).index
df = df.drop(indices_to_drop)

unwanted_labels = [
'articleType_Sarong',
'articleType_Capris',
'articleType_Poncho',
'articleType_Kimono',
'articleType_Kaftan',
'articleType_Tunics',
'articleType_Rain Trousers',
'articleType_Jodhpurs',
'articleType_Gauchos',
'articleType_Halter',
'articleType_Coverup',
'articleType_Innerwear Vests',
'articleType_Capris'
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
'articleType_Kurta Sets',
'articleType_Lehenga Choli',
'articleType_Lounge Tshirts',
'articleType_Nehru Jackets',
'articleType_Patiala',
'articleType_Robe',
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

base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(224,224, 3))

base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(df.shape[1] - 1,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
)

history = model.fit(
    train_generator, 
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

model.save('type_model_1.keras')
