import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def bicep_curl():
    """Track bicep curl repetitions for both arms in real-time."""
    global reference_angles

    # Load reference angles if available
    try:
        reference_angles = np.load('reference_angles_bicep_curl.npy').tolist()
    except FileNotFoundError:
        print("Error: Reference angles file not found. Run ref_bicep_curl() first.")
        return

    # Initialize MediaPipe pose and drawing utilities
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    # Tracking variables for both arms
    stage_left = None
    stage_right = None
    counter_left = 0
    counter_right = 0

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

            # Right arm landmarks
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Left arm landmarks
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angles
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Additional check: Ensure the elbow is close to the torso
            right_elbow_torso_distance = abs(right_elbow[0] - right_shoulder[0])  # Horizontal distance
            left_elbow_torso_distance = abs(left_elbow[0] - left_shoulder[0])  # Horizontal distance

            # Feedback and stage logic for the right arm
            if right_arm_angle > 150 and right_elbow_torso_distance < 0.1:  # Arm extended and elbow close to torso
                stage_right = "down"
            if right_arm_angle < 60 and stage_right == "down" and right_elbow_torso_distance < 0.1:  # Arm curled and elbow close to torso
                stage_right = "up"
                counter_right += 1

            # Feedback and stage logic for the left arm
            if left_arm_angle > 150 and left_elbow_torso_distance < 0.1:  # Arm extended and elbow close to torso
                stage_left = "down"
            if left_arm_angle < 60 and stage_left == "down" and left_elbow_torso_distance < 0.1:  # Arm curled and elbow close to torso
                stage_left = "up"
                counter_left += 1

            # Display text on screen
            cv2.putText(image, f'Right Arm Angle: {int(right_arm_angle)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Reps: {counter_right}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Left Arm Angle: {int(left_arm_angle)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Left Reps: {counter_left}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Bicep Curl Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
