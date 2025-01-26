import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

reference_angles = []

def ref_bicep_curl(video_path='bicep_curl.mp4'):
    """Extract reference angles from a video of correct bicep curl form."""
    global reference_angles

    # Initialize MediaPipe pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the angle
            arm_angle = calculate_angle(shoulder, elbow, wrist)
            reference_angles.append(arm_angle)

    cap.release()
    # Save reference angles for later use
    np.save('reference_angles_bicep_curl.npy', reference_angles)
    print(f"Reference angles saved to 'reference_angles_bicep_curl.npy'")

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

            # Feedback and stage logic for the right arm
            if right_arm_angle > 160:
                stage_right = "down"
            if right_arm_angle < 50 and stage_right == "down":
                stage_right = "up"
                counter_right += 1

            # Feedback and stage logic for the left arm
            if left_arm_angle > 160:
                stage_left = "down"
            if left_arm_angle < 50 and stage_left == "down":
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

if __name__ == "__main__":
    # Uncomment the following line to generate reference angles from a video
    # ref_bicep_curl("bicep_curl.mp4")

    # Start tracking bicep curls
    bicep_curl()
