
#starter code from CodingLikeMad's Reading Webcams in Python [Python OpenCV Tutorial] youtube video
import cv2
import sys
#import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException
import mysql.connector
import random
import time
import requests
from config import db_config 
 
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
color_classes = ['Beige',
'Black',
'Blue',
'Brown',
'Green',
'Grey',
'Orange',
'Pink',
'Purple',
'Red',
'White',
'Yellow'
]

# type_model = tf.keras.models.load_model('type_model.keras')
# colour_model = tf.keras.models.load_model('colour_model.keras')
# usage_model = tf.keras.models.load_model('usage_model.keras')

app = FastAPI()

def create_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return None

def scan_clothing():
    pass
    # while camera.isOpened():
    #     # Capture frame-by-frame
          #if cv2.waitKey(1) & 0xFF == ord("Enter"): #when enter is pressed, take an image after a 5s delay
    #         time.sleep(5)
    #         ret, img = camera.read()
    #         
    #     #ret, img = camera.read()
        
    #     if not ret:
    #         print("Error: Failed to capture image.")
    #         break
    #     cv2.imshow('Camera Feed', img)
    #     predicted_classses = feedIntoModel(img);
    #     sendToDatabase(img, predicted_classes)
    #     
    #    
        
    #     # cv2.imwrite('frame_{}.jpg'.format(frame_number), frame) //saves it to a jpg if we want
        
    #     if cv2.waitKey(1) & 0xFF == ord('f'): #when f is pressed, stop capturing
    #         break

    # camera.release()
    # cv2.destroyAllWindows()

# commented out for now to not disrupt riley's work

def sendToDatabase(img, predicted_classes): #TODO send to database with img and predicted_clases, probably a post request
    url = "http://127.0.0.1:8000/outfit/"
    data = {
    "image": img,
    "predicted_classes": predicted_classes
    }

    response = requests.post(url, json=data) #send post request
    print(response.json())
    return

def feedIntoModel(img):
    # image_resized = cv2.resize(img, (224, 224))
    # image_normalized = image_resized / 255.0
    # input_image = np.expand_dims(image_normalized, axis=0)

    # type_predictions = type_model.predict(input_image)
    # predicted_type = type_classes[np.argmax(type_predictions)]
    # string indicating the predicted clothing type

    # colour_predictions = colour_model.predict(input_image)
    # predicted_color = color_classes[np.argmax(color_predictions)]
    pass

'''
def feedIntoModel(img):
    image_resized = cv2.resize(img, (224, 224))
    image_normalized = image_resized / 255.0
    input_image = np.expand_dims(image_normalized, axis=0)

    type_predictions = type_model.predict(input_image)
    predicted_type = type_classes[np.argmax(type_predictions)]

    color_predictions = colour_model.predict(input_image)
    predicted_color = color_classes[np.argmax(color_predictions)]

    predicted_usage = 'Casual'
    if predicted_type in {'Dresses', 'Skirts', 'Tops', 'Trousers', 'Jackets'}:
        usage_predictions = usage_model.predict(input_image, verbose=0)[0]
        predicted_usage = usage_predictions[0]>0.5
    elif predicted_type in {'Blazers'}:
        predicted_usage = 'Formal'
'''

#below is where opencv-python is stored on the virtual environemnt capstone_cv
# /home/style_sprout/Desktop/B5_Style_Sprout/camera/capstone_cv/lib/python3.8/site-packages
#/usr/lib/python3/dist-packages/cv2.cpython-38-aarch64-linux-gnu.so
#export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages/cv2.cpython-38-aarch64-linux-gnu.so


def main():
    scan_clothing()

    camera = cv2.VideoCapture(0)
    print(camera)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    ret, img = camera.read()
    print(ret, img)

    cv2.imshow("video", img)
    cv2.waitKey()

    camera.release()

main()


