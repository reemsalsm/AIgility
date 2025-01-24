import cv2
import mediapipe as mp
import numpy as np
from ref_bicep_curl import calculate_angle, reference_angles

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def bicep_curl():
    # Ensure reference_angles is loaded correctly
    if not reference_angles or len(reference_angles) == 0:
        print("Error: reference_angles is empty. Run ref_bicep_curl() to generate reference angles.")
        return

    # Initialize MediaPipe pose and drawing utilities
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    stage = None
    counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Unable to access camera.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles
            arm_angle = calculate_angle(shoulder, elbow, wrist)
            ref_angle = reference_angles[counter % len(reference_angles)]
            angle_diff = abs(arm_angle - ref_angle)

            # Feedback and stage logic
            feedback = "Good Form" if angle_diff < 10 else "Adjust Form"
            if arm_angle > 160:
                stage = "down"
            if arm_angle < 50 and stage == "down":
                stage = "up"
                counter += 1

            # Display text on screen
            cv2.putText(image, f'Arm Angle: {int(arm_angle)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Angle Diff: {int(angle_diff)}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Repetitions: {counter}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Bicep Curl Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
