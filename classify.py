import cv2
import numpy as np
import tensorflow as tf
import subprocess
import os

# Load the trained model
model_path = 'workout_model_20250126_162435.keras'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}.")
    exit(1)

workout_model = tf.keras.models.load_model(model_path)
class_labels = ['push up', 'squat', 'barbell biceps curl', 'plank']

# Preprocess image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Predict workout
def predict_workout(frame, cap):
    processed_image = preprocess_image(frame)
    predictions = workout_model.predict(processed_image)
    print(f"Raw Predictions: {predictions}")

    confidence = np.max(predictions)
    if confidence > 0.8:  # Confidence threshold
        predicted_class = class_labels[np.argmax(predictions)]
        print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")

        # Script redirection
        script_mapping = {
            'push up': 'pushup.py',
            'squat': 'squat.py',
            'barbell biceps curl': 'bicep_curl.py',
            'plank': 'plank.py',
        }
        script_to_run = script_mapping.get(predicted_class)
        if script_to_run:
            if os.path.exists(script_to_run):
                # Release webcam and destroy all OpenCV windows
                cap.release()
                cv2.destroyAllWindows()
                
                try:
                    subprocess.Popen(['python', script_to_run])
                    print(f"Successfully launched {script_to_run}")
                except Exception as e:
                    print(f"Failed to launch {script_to_run}: {e}")
            else:
                print(f"Script {script_to_run} not found.")
    else:
        print(f"Uncertain prediction: {confidence:.2f}. Try again.")

# Main loop
cap = cv2.VideoCapture(0)
cap.set(3, 256)
cap.set(4, 256)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Workout Classification", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):  # Press 'p' to classify
        predict_workout(frame, cap)
        break  # Exit the main loop after classification and script launch

    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
