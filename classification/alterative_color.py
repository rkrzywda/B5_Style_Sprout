import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

model_dir = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model"
model = tf.saved_model.load(model_dir)
infer_fn = model.signatures['serving_default']

from sklearn.cluster import KMeans

color_classes = {
    'Beige': [(130, 255), (140, 240), (130, 200)],
    'Black': [(0, 75), (0, 75), (0, 75)],
    'Blue': [(0, 150), (0, 150), (150, 255)],
    'Brown': [(75, 180), (50, 120), (30, 90)],
    'Green': [(0, 110), (100, 255), (0, 100)],
    'Grey': [(100, 200), (100, 200), (100, 200)],
    'Orange': [(200, 255), (100, 180), (0, 50)],
    'Pink': [(200, 255), (150, 220), (200, 255)],
    'Purple': [(100, 180), (0, 80), (100, 200)],
    'Red': [(150, 255), (0, 100), (0, 100)],
    'White': [(200, 255), (200, 255), (200, 255)],
    'Yellow': [(200, 255), (200, 255), (0, 100)],
}

def get_most_prominent_color(image):
    resized_image = cv2.resize(image, (100, 100))
    lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2Lab)

    h, w, _ = lab_image.shape
    center_x, center_y = w // 2, h // 2
    x, y = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))

    sigma = 0.6 * w
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

    print(f'{r} {g} {b}')

    for color, ((r_min, r_max), (g_min, g_max), (b_min, b_max)) in color_classes.items():
        if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
            return color

    return 'Unknown'


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, img.shape

def predict_color(image_path):
    image, image_shape = load_image(image_path)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

    input_tensor = tf.expand_dims(image, axis=0)

    detections = infer_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    max_score_idx = np.argmax(scores)

    highest_score_box = boxes[max_score_idx]

    ymin, xmin, ymax, xmax = highest_score_box
    height, width, _ = image.shape
    top, left, bottom, right = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)
    cropped_image = image[top:bottom, left:right]
    cropped_image = cropped_image.numpy()
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    most_prominent_color = get_most_prominent_color(cropped_image)

    color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    color_patch[:, :] = most_prominent_color

    color_patch = color_patch / 255.0

    plt.figure()
    plt.imshow(color_patch)
    plt.title(f"{categorize_color(most_prominent_color)}")
    plt.show()


