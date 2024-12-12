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

from sklearn.cluster import KMeans

color_classes = {
    'Beige': [(150, 200), (145, 200), (135, 200)],
    'Black': [(0, 60), (0, 60), (0, 60)],
    'Blue': [(0, 130), (0, 140), (150, 255)],
    'Brown': [(100, 180), (50, 120), (30, 90)],
    'Green': [(0, 100), (150, 255), (0, 100)],
    'Orange': [(200, 255), (100, 180), (0, 50)],
    'Pink': [(200, 255), (150, 200), (200, 255)],
    'Purple': [(100, 180), (0, 80), (100, 200)],
    'Red': [(150, 255), (0, 100), (0, 100)],
    'White': [(200, 255), (200, 255), (200, 255)],
    'Yellow': [(200, 255), (200, 255), (0, 100)],
    'Grey': [(100, 200), (100, 200), (100, 200)],
}

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

    g_ratio = g/r
    b_ratio = b/r

    print(f'{r} {g} {b}')
    print(f'{g_ratio} {b_ratio}')

    if r>=135 and g_ratio < 0.55 and b_ratio<0.55 and abs(b_ratio-g_ratio)<0.1:
        return 'Red'
    if r>135 and g_ratio < 0.85 and b_ratio<0.95 and b_ratio-g_ratio>=0:
        return 'Pink'
    if r<65 and g<65 and b <65 and abs(b_ratio-g_ratio)<0.5:
        return 'Black'
    return 'Unknown'


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
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    most_prominent_color = get_most_prominent_color(cropped_image)

    color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    color_patch[:, :] = most_prominent_color

    color_patch = color_patch / 255.0

    plt.figure()
    plt.imshow(color_patch)
    plt.title(f"{categorize_color(most_prominent_color)}")
    plt.show()



