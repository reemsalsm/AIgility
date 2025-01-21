mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture('squa.mp4') 

reference_angles_left = []  
reference_angles_right = []  
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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

 
        reference_angles_right.append(angle_knee_right)
        reference_angles_left.append(angle_knee_left)
        frame_count += 1


        cv2.putText(image, f'Frame: {frame_count}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right Knee Angle: {int(angle_knee_right)}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Left Knee Angle: {int(angle_knee_left)}', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the landmarks on the image
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the processed frame
    cv2.imshow('Squat Reference Video - Knee Angles', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


np.save('reference_angles_left.npy', reference_angles_left)
np.save('reference_angles_right.npy', reference_angles_right)

cap.release()
cv2.destroyAllWindows()
