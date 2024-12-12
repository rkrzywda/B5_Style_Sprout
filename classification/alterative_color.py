import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
'''
model_dir = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model"
model = tf.saved_model.load(model_dir)
infer_fn = model.signatures['serving_default']
'''
model = tf.lite.Interpreter(model_path='objectdetect.tflite')
input_details = model.get_input_details()
model.resize_tensor_input(input_details[0]['index'], (1, 224, 224, 3))
model.allocate_tensors()
output_details = model.get_output_details()

def get_most_prominent_color(image):
    resized_image = cv2.resize(image, (100, 100))
    lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2Lab)

    h, w, _ = lab_image.shape
    center_x, center_y = w // 2, h // 2
    x, y = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))

    sigma = 0.2 * w
    gaussian_weights = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    gaussian_weights /= gaussian_weights.sum()

    pixels = lab_image.reshape((-1, 3))
    weights = gaussian_weights.flatten()
    weighted_sum = np.sum(pixels * weights[:, np.newaxis], axis=0)
    weighted_color = np.round(weighted_sum).astype(np.uint8)

    prominent_color = cv2.cvtColor(np.uint8([[weighted_color]]), cv2.COLOR_Lab2RGB)[0, 0]

    return prominent_color

def categorize_color(rgb):
    r, g, b = rgb
    r = int(r)
    g = int(g)
    b = int(b)

    g_ratio = g/r
    b_ratio = b/r

    if r>=135 and g_ratio < 0.55 and b_ratio<0.55 and abs(b_ratio-g_ratio)<0.1:
        return 'Red'
    if r>135 and g_ratio < 0.85 and b_ratio<0.95 and b_ratio-g_ratio>=0:
        return 'Pink'
    if r<70 and g<70 and b <70 and abs(b_ratio-g_ratio)<0.15 and abs(1-b_ratio)<0.15:
        return 'Black'
    if r>g and b>g and abs(b_ratio-1)<0.3 and g_ratio-1<0 and r>80 and b>80 and b_ratio-g_ratio>0.1:
        return 'Purple'
    if r>g and r>b and r>60 and g_ratio < 0.9 and g_ratio >0.6 and b_ratio<0.8 and b_ratio>0.6:
        return 'Brown'
    if r>g and r>b and g>b and g_ratio>0.85 and g_ratio<1 and b_ratio<0.85:
        return 'Yellow'
    if b-g>=10 and b-r>15 and abs(g_ratio-1)<0.5 and b_ratio-1>0.15:
        return 'Blue'
    if r>200 and g>200 and b>200:
        return 'White'
    if (abs(r-g)<5 or g-r>30) and b<g and g_ratio>=1:
        return 'Green'
    if r>110 and g>110 and b>110 and g_ratio>0.9 and g_ratio<1.1 and b_ratio>0.9 and b_ratio<1.1:
        return 'Grey'
    if r>150 and b<75 and g_ratio<0.6 and b_ratio<0.45 and g_ratio>0.35:
        return 'Orange'
    
    return 'Beige'


def load_image(image_path):
    #img = tf.io.read_file(image_path)
    #img = tf.image.decode_image(img, channels=3)
    #img = tf.image.convert_image_dtype(img, tf.float32)
    img = cv2.imread(image_path)
    return img

def predict_color(image_path):
    image = load_image(image_path)
    image = cv2.resize(image, (224, 224))

    input_tensor = np.expand_dims(image, axis=0).astype(np.uint8)
    model.set_tensor(input_details[0]['index'], input_tensor)
    model.invoke()


    boxes = model.get_tensor(output_details[4]['index'])[0]
    scores = model.get_tensor(output_details[2]['index'])
    max_score_idx = np.argmax(scores)

    highest_score_box = boxes[max_score_idx]

    ymin, xmin, ymax, xmax = highest_score_box
    height, width, _ = image.shape
    top, left, bottom, right = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)
    cropped_image = image[top:bottom, left:right]
    cropped_image = cropped_image
    most_prominent_color = get_most_prominent_color(cropped_image)

    color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    color_patch[:, :] = most_prominent_color

    color_patch = color_patch / 255.0

    #plt.figure()
    #plt.imshow(color_patch)
    #plt.title(f"{image_path[5:]} {categorize_color(most_prominent_color)}")
    #plt.show()

    return categorize_color(most_prominent_color)
