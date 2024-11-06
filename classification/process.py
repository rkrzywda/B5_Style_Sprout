import os
attributes = []
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator, array_to_img
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import Sequence
import sys
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.applications import EfficientNetB3
from keras._tf_keras.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tqdm import tqdm
import os
from PIL import Image, ImageOps

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
indices_to_drop = df[df['articleType_Cardigan'] == 1].sample(n=11700, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Dresses'] == 1].sample(n=78500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Jackets'] == 1].sample(n=12300, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Jeans'] == 1].sample(n=6300, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Jumpsuit'] == 1].sample(n=1500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Lounge Pants'] == 1].sample(n=6100, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Shorts'] == 1].sample(n=21500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Skirts'] == 1].sample(n=13100, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Sweaters'] == 1].sample(n=11500, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Tank'] == 1].sample(n=13700, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Tshirts'] == 1].sample(n=38200, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Tops'] == 1].sample(n=34600, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Leggings'] == 1].sample(n=3200, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Jumpsuit'] == 1].sample(n=3000, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['articleType_Hoodie'] == 1].sample(n=2300, random_state=42).index
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

image_folder = 'fashion-dataset/images'
output_folder = 'subset'
output_csv = 'subset.csv'
output_data = []
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for index, row in tqdm(df.iterrows(), total=len(df)):
    img_id = row['id']
    label = row.drop('id').idxmax()

    img_path = os.path.join(image_folder, f'{img_id}.jpg')
    img = load_img(img_path)
    width, height = img.size

    if width > height:
        padding = (0, (width - height) // 2, 0, width - height - (width - height) // 2)
    else:
        padding = ((height - width) // 2, 0, height - width - (height - width) // 2, 0)
    
    img = ImageOps.expand(img, padding, fill=img.getpixel((0, 0)))
    
    img = img.resize((500, 500), Image.LANCZOS)

    new_img_path = os.path.join(output_folder, f'{img_id}.jpg')
    img.save(new_img_path)

    output_data.append([new_img_path, label])
    
output_df = pd.DataFrame(output_data, columns=['id', 'articleType'])
output_df.to_csv(output_csv, index=False)

'''

image_folder = 'fashion-dataset/images'
output_folder = 'subset'
output_csv = 'subset.csv'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_data = []

for index, row in tqdm(df.iterrows(), total=len(df)):
    img_id = row['id']
    label = row.drop('id').idxmax()

    img_path = os.path.join(image_folder, f'{img_id}.jpg')
    try:
        img = load_img(img_path, target_size=(500, 500),keep_aspect_ratio=True)
        img_array = img_to_array(img)

        new_img_path = os.path.join(output_folder, f'{img_id}.jpg')
        img.save(new_img_path)

        output_data.append([new_img_path, label])
    
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")

output_df = pd.DataFrame(output_data, columns=['id', 'articleType'])
output_df.to_csv(output_csv, index=False)

names = dict()
types = dict()
counter=1
rootdir = 'img'
clothes=[]
with open('list_category_img.txt', 'r') as file:
    for line in file:
        parts = line.split()
        types[parts[0]]=int(parts[1])-1
with open('list_category_cloth.txt', 'r') as file:
    for line in file:
        parts = line.split()
        clothes.append(parts[0])
csv = open('extra.csv', 'a')
for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        prefix = ''
        if(f[:-17] not in names):
            prefix=str(counter)
            names[f[:-17]]=str(counter)
            counter+=1
        else:
            prefix=names[f[:-17]]
        newname = prefix+prefix+f[-12:]
        newname=newname[:-4]
        alternatename = 'img/'+f[:-17]+'/'+f[-16:]
        os.rename("img/"+f, 'fashion-dataset/images/'+newname+'.jpg')


for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        prefix = ''
        if(f[:-17] not in names):
            prefix=str(counter)
            names[f[:-17]]=str(counter)
            counter+=1
        else:
            prefix=names[f[:-17]]
        newname = prefix+prefix+f[-12:]
        newname=newname[:-4]
        alternatename = 'img/'+f[:-17]+'/'+f[-16:]
        try:
            csv.write(newname+','+clothes[types[alternatename]]+'\n')
        except:
            print("error")
'''

