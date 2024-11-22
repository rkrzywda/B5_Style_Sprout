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
from pynput import keyboard
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


take_picture = False
exit_loop = False

def on_press(key):
    global take_picture, exit_loop
    try:
        if key.char == 'k':
            take_picture = True
        elif key.char == 'f':
            exit_loop = True
    except AttributeError:
        pass


def scan_clothing():
    global take_picture, exit_loop
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    print(camera)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    while camera.isOpened():
        # Capture frame-by-frame
        # print("should be waiting for key input")
        if take_picture: #when enter is pressed, take an image after a 5s delay
            time.sleep(2)
            print("in if statement")
            ret, img = camera.read()
            img = cv2.resize(img, (224, 224))
            if not ret:
                print("Error: Failed to capture image.")
                break
            print("trying to show the image")
            cv2.imshow('Camera Feed', img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            cv2.waitKey(100)
            take_picture = False
        
        #predicted_classses = feedIntoModel(img);
        #sendToDatabase(img, predicted_classes)

        if exit_loop:
            print("trying to exit loop")
            break
        
       
        
        # cv2.imwrite('frame_{}.jpg'.format(frame_number), frame) //saves it to a jpg if we want
    print("trying to send to database")
    sendToDatabase("bad value", "another one")
    print("finished trying")
    camera.release()
    cv2.destroyAllWindows()


def sendToDatabase(img, predicted_classes): #TODO send to database with img and predicted_clases, probably a post request
    outfitUrl = "http://0.0.0.0:8000/outfit/info" 
    #when we run it with --host 0.0.0.0 might need to change it to the hostees ipaddress
    test_outfit_data = {
        "clothingType": "Sweater",
        "color": "Blue",
        "season": "Winter", 
        "usageType":"Formal",
    }
    print("trying to send post request")
    response = requests.post(outfitUrl, json=test_outfit_data) #send post request
    print("Response from API ", response.json())
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
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    scan_clothing()

main()