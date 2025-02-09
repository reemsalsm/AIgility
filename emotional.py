import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the frame for emotions
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Extract emotion with the highest probability
        dominant_emotion = analysis[0]['dominant_emotion']

        # Display detected emotion
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print("Error:", e)

    # Show the frame
    cv2.imshow('Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
