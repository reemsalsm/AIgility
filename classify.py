import cv2
import numpy as np
import tensorflow as tf
import os
import subprocess

# Load the correct TFLite model
model_path = "workout_model_20250126_162435.tflite"  # Ensure it's a .tflite model!
if not os.path.exists(model_path):
    print(f" Model not found at {model_path}. Ensure you have a valid TFLite model.")
    exit(1)

#  Load the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

#  Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#  Define the workout classes and map them to their scripts
script_mapping = {
    'push up': 'pushup.py',
    'squat': 'squat.py',
    'barbell biceps curl': 'bicep_curl.py',
    'plank': 'plank.py',
}

def preprocess_image(image):
    """Preprocesses the frame before classification."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (256, 256))  # Resize to model input size
    image = image.astype(np.float32) / 255.0  # Normalize to 0-1
    return np.expand_dims(image, axis=0)  # Add batch dimension

def predict_workout(frame):
    """Runs inference on the given frame and determines the workout."""
    processed_image = preprocess_image(frame)

    # Ensure input data type matches model’s expected format
    processed_image = processed_image.astype(input_details[0]["dtype"])

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], processed_image)

    # Run inference
    interpreter.invoke()

    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    predictions = tf.nn.softmax(predictions).numpy()  # Apply softmax

    # Get the predicted class and confidence
    confidence = np.max(predictions)
    predicted_index = np.argmax(predictions)
    predicted_class = list(script_mapping.keys())[predicted_index]

    return predicted_class, confidence

#  Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Failed to open webcam.")
    exit(1)

cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Get prediction
        detected_workout, confidence = predict_workout(frame)

        # Display the detected workout and confidence on the frame
        text = f"{detected_workout} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Workout Classification", frame)

        # Check if confidence is high enough to switch to the workout script
        if confidence > 0.4:  # Confidence threshold
            print(f" Detected: {detected_workout} ({confidence:.2f})")

            # Close the webcam
            cap.release()
            cv2.destroyAllWindows()

            # Get the corresponding script from the mapping
            script_name = script_mapping.get(detected_workout)
            if script_name and os.path.exists(script_name):
                print(f" Running {script_name}...")
                subprocess.run(["python", script_name], check=True)
            else:
                print(f" No script found for {detected_workout}.")
            break  # Exit the classification loop

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
except KeyboardInterrupt:
    print("Script interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released.")
