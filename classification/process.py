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
import shutil

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('fashion-dataset/styles.csv')
df = df[df['masterCategory'] == 'Apparel']
df1 = pd.read_csv('archive/blazers.csv')
df = pd.concat([df,df1])
df2 = pd.read_csv('extra.csv')
df = pd.concat([df,df2])
df = df[['id', 'articleType', 'usage']] 
df['usage'] = df['usage'].replace('Sports', 'Casual')
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
df = df[df['articleType'].isin(['Trousers'])]
df = df[df.isna().any(axis=1)]

seen = pd.read_csv('subset_type.csv')
#id_set = set(seen['id'])
#seen.fillna('Formal', inplace=True)
#seen.to_csv('usage_subset.csv', index=False)
#df = df.reset_index(drop=True)
#random_indices = df.sample(n=200).index

#for i in random_indices:
    #row = df.iloc[i]
    #if row['id'] not in id_set:
        #print(row['id'])

if False:
    for index, row in df.iterrows():
        name = row['id']
        if name not in id_set:
            plt.figure()
            img = load_img(f'fashion-dataset/images/{name}.jpg', keep_aspect_ratio=True)
            img_array = img_to_array(img)
            img_array = img_array/255.0 
            plt.imshow(img_array)
            print(name)
            plt.show()


if True:
    image_folder = 'fashion-dataset/images'
    output_folder = 'type_subset'
    output_data = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if True:
        for index, row in tqdm(seen.iterrows(), total=len(seen)):
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

    #output_csv = 'usage_subset.csv'
    #output_df = pd.DataFrame(output_data, columns=['id', 'usage'])
    #output_df.to_csv(output_csv, index=False)

'''
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
df['baseColour'] = df['baseColour'].replace('Off White', 'Beige')
df['baseColour'] = df['baseColour'].replace('Cream', 'Beige')
df['baseColour'] = df['baseColour'].replace('Khaki', 'Beige')
df['baseColour'] = df['baseColour'].replace('Charcoal', 'Black')
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
df['baseColour'] = df['baseColour'].replace('Grey Melange', 'Grey')
df['baseColour'] = df['baseColour'].replace('Lime Green', 'Green')
df['baseColour'] = df['baseColour'].replace('Fluorescent Green', 'Green')
df['baseColour'] = df['baseColour'].replace('Teal', 'Blue')
df['baseColour'] = df['baseColour'].replace('Navy Blue', 'Blue')

unwanted_labels = [
'Belts',
'Booties',
'Boxers',
'Bra',
'Briefs',
'Shapewear',
'Stockings',
'Suspenders',
'Swimwear',
'Tights',
]
df = df[~df['articleType'].isin(unwanted_labels)]

df = df[['id', 'baseColour']] 

df = pd.get_dummies(df, columns=['baseColour']).astype(int)
indices_to_drop = df[df['baseColour_Beige'] == 1].sample(n=276, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Black'] == 1].sample(n=2544, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Blue'] == 1].sample(n=4144, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Brown'] == 1].sample(n=35, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Green'] == 1].sample(n=1121, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Grey'] == 1].sample(n=1093, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Pink'] == 1].sample(n=603, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Purple'] == 1].sample(n=543, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Red'] == 1].sample(n=1219, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_White'] == 1].sample(n=2065, random_state=42).index
df = df.drop(indices_to_drop)
indices_to_drop = df[df['baseColour_Yellow'] == 1].sample(n=116, random_state=42).index
df = df.drop(indices_to_drop)


unwanted_labels = [
'baseColour_Multi',
'baseColour_Gold',
'baseColour_Silver',
]
df = df.drop(columns=unwanted_labels, errors='ignore')
label_counts = df.drop(columns=['id']).sum()
print(label_counts)

df.to_csv('color_subset.csv', index=False)

image_folder = 'fashion-dataset/images'
output_folder = 'color_subset'
output_data = []
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if True:
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

output_csv = 'color_subset.csv'
output_df = pd.DataFrame(output_data, columns=['id', 'baseColour'])
output_df.to_csv(output_csv, index=False)

df = df[(df.drop(columns=['id']).sum(axis=1) > 0)]
df['usage'] = df[['usage_Casual', 'usage_Formal']].idxmax(axis=1)
df = df[['id', 'usage']]
df.to_csv('type_subset.csv', index=False)


count=0
for subdir, dirs, files in os.walk('fashion-dataset/images'):
    for f in files:
        count+=1
print(count)

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

