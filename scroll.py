import cv2
import mediapipe as mp
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)
EYEBROW_THRESHOLD = 7

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            eyebrow = face_landmarks.landmark[65]
            eye = face_landmarks.landmark[159]

            eyebrow_y = int(eyebrow.y * h)
            eye_y = int(eye.y * h)

            distance = eye_y - eyebrow_y

            if distance > EYEBROW_THRESHOLD:
                pyautogui.scroll(40)
                cv2.putText(frame, "SCROLL UP", (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                pyautogui.scroll(-40)
                cv2.putText(frame, "SCROLL DOWN", (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Eyebrow Scroll Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
