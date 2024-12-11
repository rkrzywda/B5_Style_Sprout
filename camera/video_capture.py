import cv2
import random, os
import numpy as np
import Jetson.GPIO as GPIO
import time
import requests
from pynput import keyboard
import boto3
from config import db_config, access_key, secret_key, api_ip, counter
import tflite_runtime.interpreter as tflite
 
#when wiring the pushbutton, a 3.3V power source with a ~1K ohm resistor is needed to act 
#as a pull up resistor for the pushbutton to ensure that input voltage is not floating. 

#if a 5V power source is used, only one button press will be detected
#This is because the Jetso GPIO only detects a high if the voltage is greater than 2.0V
#and less than or equal 3.3V

BUTTON_PIN = 15

GPIO.setmode(GPIO.BOARD);

GPIO.setup(BUTTON_PIN, GPIO.IN)

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
usage_classes=['Casual', 'Formal']

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

exit_loop = False
take_picture = False

def detect(channel):
    global take_picture
    take_picture = True
    return
    
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=detect, bouncetime=200) #debounce time of 200ms

#for detecting a button press (desire to take a picture or exit)
def on_press(key):
    global take_picture, exit_loop
    try:
        if key.char == '0':
            exit_loop = True
    except AttributeError:
        pass


#callback function when the button is pressed, triggers start of procedure
def processScanRequest(img):
    before = time.time()
    counter = 101
    with open('counter.txt', 'r') as file:
        content = file.read()
        print(content)
        counter = int(content)
        
    print("showing image")
    cv2.imshow('Camera Feed', img)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()
    cv2.waitKey(100)
    beforeClassification = time.time()
    predicted_classes = feedIntoModel(img);
    print("Predicted Classes: ",  predicted_classes)
    timeClassification = time.time() - beforeClassification
    print("Time taken to classify ", timeClassification)
    
    #write an image to send to s3
    beforeS3 = time.time()
    #imageName = f"username{counter}"
    #cv2.imwrite(f"{imageName}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    #s3.upload_file(f"{imageName}.jpg", "style-sprout", f"{imageName}.jpg")
    try:
         with open('counter.txt', 'w') as file:
             file.write(f"{counter+1}")
    #    os.remove(f"{imageName}.jpg")
    except FileNotFoundError:
        print("file not found")
   
        
    timeS3 = time.time() - beforeS3
    print("Time taken to send to s3 ", timeS3)
           
    #send to SQL database
    beforeDB = time.time()
    #sendToDatabase(img, predicted_classes, imageName) #update the database
    timeDB = time.time() - before
    print("Time taken to send to db ", timeDB)

    take_picture = False
    totalTime = time.time() - beforeClassification
    print("Total time taken ", totalTime)
    return



#continously take pictures, classify them, send to a database and s3
def scan_clothing():
    global exit_loop, take_picture
    #s3 = boto3.client(
    #    's3',
    #    aws_access_key_id = access_key,
    #    aws_secret_access_key = secret_key,
    #    region_name = 'us-east-2'
    #)
   
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
        ret, img = camera.read() #take the picture
        if not ret:
            print("Error: Failed to capture image.")
            break
        cv2.imshow('Camera Feed', img)
        cv2.waitKey(1)
        
        if take_picture:
            cv2.destroyAllWindows()
            cv2.waitKey(10)
            time.sleep(2)
            ret, img = camera.read() #take the picture
            
            if not ret:
                print("Error: Failed to capture image.")
                return
        
            processScanRequest(img)
            take_picture = False
            
        if exit_loop: #Button that indicates we should stop scanning has been pressed
            print()
            print("exiting loop")
            break
       
    camera.release()
    cv2.destroyAllWindows()


#Sends a taken image to the database
def sendToDatabase(img, predicted_classes, imageName):
   
    outfitUrl = f"http://{api_ip}:8000/outfit/info" #get the current api_ip from config.py

    print("sending post request")
    databaseInfo = {
        "clothingType": predicted_classes["clothingType"],
        "color": predicted_classes["color"],
        "usageType": predicted_classes["usageType"],
        "imageName": imageName,
    }
    response = requests.post(outfitUrl, json=databaseInfo) #send post request
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
    
    #GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=processScanRequest, bouncetime=200) #debounce time of 200ms
    
    scan_clothing()

main()