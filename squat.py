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

# pushup.py

import time

def give_feedback(reps):
    if reps < 5:
        print("Keep going! Try to complete more reps.")
    elif reps < 10:
        print("You're doing great, keep it up!")
    else:
        print("Excellent work! Keep pushing.")

# Simulate counting pushups
reps = 0
start_time = time.time()

while True:
    reps += 1
    give_feedback(reps)
    time.sleep(2)
    
    if reps >= 20:
        break
print(f"total squats: {reps}")

def ref_squat(video_path='squat.mp4'):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return

    reference_angles_left = []
    reference_angles_right = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Right side
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Left side
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angles
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

            reference_angles_right.append(right_leg_angle)
            reference_angles_left.append(left_leg_angle)

    cap.release()
    np.save('reference_angles_squat_right.npy', reference_angles_right)
    np.save('reference_angles_squat_left.npy', reference_angles_left)
    print(f"Reference angles saved for both sides.")


def squat():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    # Load reference angles
    try:
        reference_angles_left = np.load('reference_angles_squat_left.npy')
        reference_angles_right = np.load('reference_angles_squat_right.npy')
    except FileNotFoundError:
        print("Error: Reference angles not found. Run ref_squat() first.")
        return

    # Initialize variables
    counter = 0
    stage = None

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

            # Right leg
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Left leg
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angles
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Feedback logic
            if right_leg_angle > 169 and left_leg_angle > 169:
                stage = "up"
            if right_leg_angle < 90 and left_leg_angle < 90 and stage == "up":
                stage = "down"
                counter += 1

            # Display feedback
            cv2.putText(image, f'Reps: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Angle: {int(right_leg_angle)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Left Angle: {int(left_leg_angle)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Squat Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Uncomment to generate reference angles
    # ref_squat('squat.mp4')

    # Uncomment to start squat tracking
    squat()
