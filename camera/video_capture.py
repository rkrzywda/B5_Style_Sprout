#starter code from CodingLikeMad's Reading Webcams in Python [Python OpenCV Tutorial] youtube video
import cv2
import sys
#import tensorflow as tf
import numpy as np
import time
import requests
from pynput import keyboard
import tflite_runtime.interpreter as tflite
 
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
usage_classes=['casual', 'formal']

type_model = tflite.Interpreter(model_path='type.tflite')
type_model.allocate_tensors()
type_input = type_model.get_input_details()
type_output = type_model.get_output_details()

color_model = tflite.Interpreter(model_path='color.tflite')
color_model.allocate_tensors()
color_input = color_model.get_input_details()
color_output = color_model.get_output_details()

usage_model = tflite.Interpreter(model_path='usage.tflite')
usage_model.allocate_tensors()
usage_input = usage_model.get_input_details()
usage_output = usage_model.get_output_details()

print("Que pasa??")

take_picture = False
exit_loop = False
save_to_database = False

def on_press(key):
    global take_picture, exit_loop, save_to_database
    try:
        if key.char == 'c':
            take_picture = True
        elif key.char == '0':
            exit_loop = True
        elif key.char == 's':
            save_to_database = True
    except AttributeError:
        pass


def scan_clothing():
    global take_picture, exit_loop, save_to_database
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    while camera.isOpened():
        if take_picture: #when enter is pressed, take an image after a 2s delay
            print()
            time.sleep(2)
            ret, img = camera.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            print("showing image")
            cv2.imshow('Camera Feed', img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            cv2.waitKey(100)
            predicted_classes = feedIntoModel(img);
            print(predicted_classes)
            take_picture = False

       
        if save_to_database: #button that indicates we should save the data has been pressed
            print()
            print("sending to database")
            sendToDatabase(img, "Test Data")
            save_to_database = False
       
       
        if exit_loop: #Button that we should exit the scanning loop has been sent
            print()
            print("exiting loop")
            break
       
        # cv2.imwrite('frame_{}.jpg'.format(frame_number), frame) //saves it to a jpg if we want

    camera.release()
    cv2.destroyAllWindows()


#Sends a taken image to the database
def sendToDatabase(img, predicted_classes):
    outfitUrl = "http://128.2.13.179:8000/outfit/info"
    #when we run it with --host 0.0.0.0 might need to change it to the hostees ipaddress
    test_outfit_data = {
        "clothingType": "Sweater",
        "color": "Blue",
        "usageType":"Formal",
    }
    print("sending post request")
    response = requests.post(outfitUrl, json=test_outfit_data) #send post request
    print("Response from API ", response.json())
    return


def feedIntoModel(img):
    image_resized = cv2.resize(img, (224, 224))
    image_normalized = image_resized / 255.0
    input_image = np.expand_dims(image_normalized, axis=0)
    input_image = input_image.astype(np.float32)

    type_model.set_tensor(type_input[0]['index'], input_image)
    type_model.invoke()
    type_predictions = type_model.get_tensor(type_output[0]['index'])[0]
    predicted_type = type_classes[np.argmax(type_predictions)]

    color_model.set_tensor(color_input[0]['index'], input_image)
    color_model.invoke()
    color_predictions = color_model.get_tensor(color_output[0]['index'])[0]
    predicted_color = color_classes[np.argmax(color_predictions)]

    predicted_usage = 'Casual'
    if predicted_type in {'Dresses', 'Skirts', 'Tops', 'Trousers', 'Jackets'}:
        usage_model.set_tensor(usage_input[0]['index'], input_image)
        usage_model.invoke()
        usage_predictions = usage_model.get_tensor(usage_output[0]['index'])[0]
        predicted_usage = usage_classes[np.argmax(usage_predictions)]
    elif predicted_type in {'Blazers'}:
        predicted_usage = 'Formal'
    print(predicted_type)
    print(predicted_color)
    print(predicted_usage)
    return [predicted_type, predicted_color, predicted_usage]


def main():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    scan_clothing()

main()
