import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks (1).dat")

cap = cv2.VideoCapture(0)

lip_points = []
lip_counter = 0

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)

        # Extract lip points (48 to 68)
        lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

        # Get bounding box dimensions for cropping
        x, y, w, h = cv2.boundingRect(np.array(lip_points))
        lip_crop = frame[y:y + h, x:x + w]

        # Create a box of size 150x200 at the top-left corner
        box = np.zeros((150, 200, 3), dtype=np.uint8)
        box[:min(h, 150), :min(w, 200)] = lip_crop[:min(h, 150), :min(w, 200)]
        frame[:150, :200] = box

        if lip_counter < 10:
            # Save the cropped lip image and its corresponding landmarks
            cv2.imwrite(f"dot/lip_{lip_counter}.jpg", lip_crop)

            # Adjust and save the landmark points for the cropped lip region
            adjusted_landmarks = [(point[0] - x, point[1] - y) for point in lip_points]
            with open(f"dot/lip_landmarks_{lip_counter}.txt", 'w') as file:
                for point in adjusted_landmarks:
                    file.write(f"{point[0]},{point[1]}\n")

            lip_counter += 1

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == ord('f'):
        # Show lip points when 'f' is pressed
        for point in lip_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        cv2.imshow('Lip Points', frame)

    elif key == ord('q'):
        # Quit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
