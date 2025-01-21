
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


cap = cv2.VideoCapture(0)  


reference_angles_left = np.load('reference_angles_left.npy') 
reference_angles_right = np.load('reference_angles_right.npy')  
counter = 0
stage = None


while cap.isOpened():
    Success, frame = cap.read()
 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark


        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]


        angle_knee_right = calculate_angle(right_hip, right_knee, right_ankle)
        angle_knee_left = calculate_angle(left_hip, left_knee, left_ankle)


        angle_difference_right = abs(angle_knee_right - reference_angles_right[counter % len(reference_angles_right)])
        angle_difference_left = abs(angle_knee_left - reference_angles_left[counter % len(reference_angles_left)])

  
        if angle_difference_right > 10 or angle_difference_left > 10:
            feedback = "Adjust Knee Angle"
        else:
            feedback = "Good Form"


        cv2.putText(image, f'Reps: {counter}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'{feedback}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        if angle_knee_right > 169 and angle_knee_left > 169:
            stage = "up"
        if angle_knee_right < 90 and angle_knee_left < 90 and stage == 'up':
            stage = "down"
            counter += 1


    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    cv2.imshow('Squat Form Feedback', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
