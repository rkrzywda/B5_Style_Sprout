#starter code from CodingLikeMad's Reading Webcams in Python [Python OpenCV Tutorial] youtube video
import cv2
import sys
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

#below is where opencv-python is stored on the virtual environemnt capstone_cv
# /home/style_sprout/Desktop/B5_Style_Sprout/camera/capstone_cv/lib/python3.8/site-packages
#/usr/lib/python3/dist-packages/cv2.cpython-38-aarch64-linux-gnu.so
#export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages/cv2.cpython-38-aarch64-linux-gnu.so