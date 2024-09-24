import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('fashion-dataset/styles.csv')
print(df.head())
df = df[['id', 'articleType', 'baseColour', 'season', 'usage']]
print(df.isnull().sum())
df.dropna(inplace=True)

df = pd.get_dummies(df, columns=['articleType', 'baseColour', 'season', 'usage'])
def preprocess_image(image_id, target_size=(224, 224)):
    image_path = f'fashion-dataset/images/{image_id}.jpg'
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

df['image'] = df['id'].apply(lambda x: preprocess_image(x))

X = np.stack(df['image'].values)
y = df.drop(columns=['id', 'image']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(20)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=10)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

print('\nTest accuracy:', test_acc)

