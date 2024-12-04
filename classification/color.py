import cv2
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
color_model = tf.keras.models.load_model('color_model_502.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(color_model)
tflite_model = converter.convert()

with open("colornew2.tflite", "wb") as f:
    f.write(tflite_model)
'''
def get_dominant_color(image, k=3):
    # Reshape image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Cluster the pixel intensities
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    
    return dominant_color

# Load the image
image = cv2.imread('test/1.png')

# Crop the center of the image
h, w, _ = image.shape
crop = image[h//4:3*h//4, w//4:3*w//4]

# Get the dominant color
dominant_color = get_dominant_color(crop)
print("Dominant Color (RGB):", dominant_color)
'''