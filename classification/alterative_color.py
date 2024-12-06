import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

model_dir = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model"
detect_fn = tf.saved_model.load(model_dir)
infer_fn = detect_fn.signatures['serving_default']

from sklearn.cluster import KMeans

def get_most_prominent_color(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = image_hsv.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    cluster_sizes = np.bincount(kmeans.labels_)
    dominant_color_idx = np.argmax(cluster_sizes)
    dominant_hsv = dominant_colors[dominant_color_idx]
    dominant_rgb = cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
    
    return dominant_rgb

def categorize_color(rgb):
    return None

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

    return categorize_color(most_prominent_color)


