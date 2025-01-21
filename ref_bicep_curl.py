mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


reference_angles = []
ref_cap = cv2.VideoCapture('bicep_curl.mp4')

while ref_cap.isOpened():
    ret, ref_frame = ref_cap.read()
    if not ret:
        break

    ref_image = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
    ref_image.flags.writeable = False
    ref_results = pose.process(ref_image)

    if ref_results.pose_landmarks:
        ref_landmarks = ref_results.pose_landmarks.landmark

        shoulder = [ref_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    ref_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [ref_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 ref_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [ref_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 ref_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        arm_angle = calculate_angle(shoulder, elbow, wrist)
        reference_angles.append(arm_angle)

ref_cap.release()
