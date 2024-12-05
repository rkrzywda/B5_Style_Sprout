import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model_dir = "ssd_mobilenet_v2_coco_2018_03_29/saved_model"
detect_fn = tf.saved_model.load(model_dir)
infer_fn = detect_fn.signatures['serving_default']

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, img.shape

image_path = "test/6.png"
image, image_shape = load_image(image_path)
image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

input_tensor = tf.expand_dims(image, axis=0)

detections = infer_fn(input_tensor)

boxes = detections['detection_boxes'][0].numpy()
scores = detections['detection_scores'][0].numpy()
classes = detections['detection_classes'][0].numpy()
max_score_idx = np.argmax(scores)

highest_score_box = boxes[max_score_idx]
highest_score = scores[max_score_idx]

if highest_score > 0:
    ymin, xmin, ymax, xmax = highest_score_box
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.gca().add_patch(
        plt.Rectangle(
            (xmin * image_shape[1], ymin * image_shape[0]),
            (xmax - xmin) * image_shape[1],
            (ymax - ymin) * image_shape[0],
            edgecolor='red',
            facecolor='none',
            linewidth=2,
        )
    )
    plt.title(f"Highest Score: {highest_score:.4f}")
    plt.show()
else:
    print("No objects detected.")
