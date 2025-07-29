import tensorflow as tf
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import sys
import pyttsx3
import threading

# Set the console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Define a custom DepthwiseConv2D layer
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

# Register the custom layer
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# Load model function
def load_model(path):
    try:
        model = tf.keras.models.load_model(path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
        print("Model loaded successfully")
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

# Path to the model
model_path = r"C:\projects\main\sign language detection\Sign-Language-detection-main\Model\keras_model.h5"

model = load_model(model_path)

if model:
    class CustomClassifier:
        def __init__(self, model, labels_path):
            self.model = model
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.labels = f.read().splitlines()

        def getPrediction(self, img):
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            predictions = self.model.predict(img)
            index = np.argmax(predictions)
            confidence = predictions[0][index]
            return [confidence], index

    def speak_text(text):
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Error in speak_text:", e)

    # Try camera index 0; if it fails, try 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Camera index 0 not working, trying index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open any video capture device.")
    else:
        detector = HandDetector(maxHands=1)
        classifier = CustomClassifier(model, r"C:\projects\main\sign language detection\Sign-Language-detection-main\Model\labels.txt")
        offset = 20
        imgSize = 300

        labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","O","P","Q","R","V","W","X","Y","Z"]

        while True:
            success, img = cap.read()
            if not success:
                print("Error: Failed to read frame from capture.")
                break

            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

                if imgCrop.size == 0:
                    print("Warning: Empty crop, skipping frame.")
                    continue

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite)
                confidence = prediction[0]

                cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

                threading.Thread(target=speak_text, args=(labels[index],)).start()

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

            cv2.imshow('Image', imgOutput)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
else:
    print("Failed to load the model. Please check the model path and compatibility.")
