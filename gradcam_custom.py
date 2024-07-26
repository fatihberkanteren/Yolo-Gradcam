from roboflow import Roboflow
import cv2
import imutils
import numpy as np
from keras import Model
from keras.src.applications import imagenet_utils
from keras.src.applications.vgg16 import VGG16
from keras.src.utils import load_img, img_to_array
import tensorflow as tf

def draw_yolo_predictions(image, predictions):
    for pred in predictions['predictions']:
        # Convert the bounding box center (x, y), width, and height to coordinates
        x_center, y_center = pred['x'], pred['y']
        width, height = pred['width'], pred['height']

        x0 = int(x_center - width / 2)
        y0 = int(y_center - height / 2)
        x1 = int(x_center + width / 2)
        y1 = int(y_center + height / 2)

        class_name = pred['class']
        confidence = pred['confidence']

        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{class_name}: {confidence * 100:.2f}%"
        cv2.putText(image, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(inputs=[self.model.inputs],
                          outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)

img = "resim_dataset/resimler1/1Lira/15538a.jpg"
model = VGG16(weights="imagenet")
orig = cv2.imread(img)
resized = cv2.resize(orig, (224, 224))
image = load_img(img, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)
preds = model.predict(image)
i = np.argmax(preds[0])
decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))

cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("dataset")
yolo_model = project.version("verison").model
yolo_predictions = yolo_model.predict(img, confidence=40, overlap=30).json()

yolo_output = draw_yolo_predictions(orig.copy() ,yolo_predictions)

# Increase the size of the combined output
combined_output = np.vstack([output, heatmap, yolo_output])
combined_output = imutils.resize(combined_output, height=1000)  # Increase the height for better visibility
cv2.imshow("Output", combined_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
