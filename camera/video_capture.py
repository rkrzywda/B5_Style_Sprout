#starter code from CodingLikeMad's Reading Webcams in Python [Python OpenCV Tutorial] youtube video
import cv2
import sys
#import tensorflow as tf
import numpy as np

# type_model = tf.keras.models.load_model('type_model.keras')
# colour_model = tf.keras.models.load_model('colour_model.keras')
# usage_model = tf.keras.models.load_model('usage_model.keras')

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


# while camera.isOpened():
#     # Capture frame-by-frame
#     ret, img = camera.read()
    
#     if not ret:
#         print("Error: Failed to capture image.")
#         break
#     cv2.imshow('Camera Feed', img)
    
#     # Save each frame as an image file if needed
#     # cv2.imwrite('frame_{}.jpg'.format(frame_number), frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('f'): #when f is pressed
#         break

# camera.release()
# cv2.destroyAllWindows()

# commented out for now to not disrupt riley's work

def feedIntoModel(img):
    # image_resized = cv2.resize(img, (224, 224))
    # image_normalized = image_resized / 255.0
    # input_image = np.expand_dims(image_normalized, axis=0)

    # type_predictions = type_model.predict(input_image)
    # threshold = 0.5
    # predicted_classes = (type_predictions > threshold).astype(int)
    # print("Predictions:", predicted_classes)
    # # one hot encoded array like [0 0 1] where 1 indicates the class that it is
    # # if classes were [shirt pants jacket] then this would be a jacket


    # colour_predictions = colour_model.predict(input_image)
    # predicted_classes = (colour_predictions > threshold).astype(int)
    # print("Predictions:", predicted_classes)

    # usage_predictions = usage_model.predict(input_image)
    # predicted_classes = (usage_predictions > threshold).astype(int)
    # print("Predictions:", predicted_classes)
    #return predicted_classes
    pass

#below is where opencv-python is stored on the virtual environemnt capstone_cv
# /home/style_sprout/Desktop/B5_Style_Sprout/camera/capstone_cv/lib/python3.8/site-packages
#/usr/lib/python3/dist-packages/cv2.cpython-38-aarch64-linux-gnu.so
#export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages/cv2.cpython-38-aarch64-linux-gnu.so