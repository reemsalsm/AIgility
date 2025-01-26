import cv2
import numpy as np
import tensorflow as tf
import os
import subprocess
import time

# Load the trained model
model_path = 'workout_model_20250127_123456.keras'  # Replace with your model path
workout_model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['push up', 'squat', 'barbell biceps curl', 'plank']

# Load the webcam
cap = cv2.VideoCapture(0)

# Set video feed size
cap.set(3, 256)  # width
cap.set(4, 256)  # height

# Function to preprocess the image for the model
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (256, 256))  # Resize to match input size
    image = image / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the workout and redirect
def predict_workout(frame):
    # Preprocess the image
    processed_image = preprocess_image(frame)
    
    # Predict using the model
    predictions = workout_model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    print(f"Predicted: {predicted_class}")

    # Redirect to the appropriate Python file based on prediction
    if predicted_class == 'push up':
        subprocess.Popen(['python3', 'pushup.py'])
    elif predicted_class == 'squat':
        subprocess.Popen(['python3', 'squat.py'])
    elif predicted_class == 'barbell biceps curl':
        subprocess.Popen(['python3', 'bicep_curl.py'])
    elif predicted_class == 'plank':
        subprocess.Popen(['python3', 'plank.py'])

# Loop to capture frames from the webcam
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Display the current frame
    cv2.imshow("Workout Classification", frame)


    
    # Classify workout on key press (you can set a specific key to trigger prediction)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Press 'p' to classify the current frame
        predict_workout(frame)

    # Exit condition: Press 'q' to quit
    if key == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

