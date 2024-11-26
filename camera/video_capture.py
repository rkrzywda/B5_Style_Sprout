import cv2
import random, os
import numpy as np
import time
import requests
from pynput import keyboard
import boto3
from config import db_config, access_key, secret_key, api_ip, counter
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

#Create models
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

user_start_scanning = False
take_picture = False
exit_loop = False

# print("Waiting for user to start scan")
# while not user_start_scanning:
#     pass


#for detecting a button press (desire to take a picture or exit)
def on_press(key):
    global take_picture, exit_loop
    try:
        if key.char == 'c':
            take_picture = True
        elif key.char == '0':
            exit_loop = True
    except AttributeError:
        pass

#continously take pictures, classify them, send to a database and s3
def scan_clothing():
    global take_picture, exit_loop
    s3 = boto3.client(
        's3',
        aws_access_key_id = access_key,
        aws_secret_access_key = secret_key,
        region_name = 'us-east-2'
    )
   
    #initialize the camera,
    #starter code for cv from CodingLikeMad's Reading Webcams in Python [Python OpenCV Tutorial]
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
            ret, img = camera.read() #take the picture
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
            
            #write an image to send to s3
            imageName = f"username{counter}"
            cv2.imwrite(f"{imageName}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            s3.upload_file(f"{imageName}.jpg", "style-sprout", imageName)
            try:
                os.remove(f"{imageName}.jpg")
                print("Taken image was deleted")
            except FileNotFoundError:
                print("file not found")
            counter += 1
           
            #send to SQL database
            sendToDatabase(img, predicted_classes, imageName) #update the database
        
            take_picture = False

       
        if save_to_database: #button that indicates we should save the data has been pressed
            print()
            print("sending to database")
            sendToDatabase(img, predicted_classes)
            save_to_database = False
       
       
        if exit_loop: #Button that we should exit the scanning loop has been sent
            print()
            print("exiting loop")
            break
       
    camera.release()
    cv2.destroyAllWindows()


#Sends a taken image to the database
def sendToDatabase(img, predicted_classes, imageName):
   
    outfitUrl = f"http://{api_ip}:8000/outfit/info" #get the current api_ip from config.py

    print("sending post request")
    databaseinfo = {
        "clothingType": predicted_classes["clothingType"],
        "color": predicted_classes["color"],
        "usageType": predicted_classes["usageType"],
        "url": f"s3://style-sprout/{imageName}.jpg",
    }
    response = requests.post(outfitUrl, json=predicted_classes) #send post request
    print("Response from API ", response.json())
    return

#feeds an image into the model and gets the classifications back
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
   
    predicted_classes = {
        "clothingType": predicted_type,
        "color": predicted_color,
        "usageType": predicted_usage,
    }
    return predicted_classes

#start the main processes
def main():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    scan_clothing()

main()