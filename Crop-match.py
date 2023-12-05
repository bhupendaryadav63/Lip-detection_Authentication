import cv2
import dlib
import numpy as np
import os
import time

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks (1).dat")

cap = cv2.VideoCapture(0)

lip_counter = 0
stored_landmarks_dir = "dot"  # Directory containing stored landmarks

start_comparison = False

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
            cv2.imwrite(f"dot2/lip_{lip_counter}.jpg", lip_crop)

            # Adjust and save the landmark points for the cropped lip region
            adjusted_landmarks = [(point[0] - x, point[1] - y) for point in lip_points]
            with open(f"dot2/lip_landmarks_{lip_counter}.txt", 'w') as file:
                for point in adjusted_landmarks:
                    file.write(f"{point[0]},{point[1]}\n")

        lip_counter += 1

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == ord('v'):
        # Start comparison process when 'v' is pressed
        start_comparison = True

    if start_comparison and lip_counter >= 10:
        # Compare the landmarks of the captured frames with stored landmarks
        frame_count = 0
        success_count = 0

        while frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            live_landmarks = []

            for face in faces:
                landmarks = predictor(gray, face)
                lip_points_live = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
                live_landmarks.append(lip_points_live)

            # Calculate relative distances for live stats (from point 1)
            if live_landmarks:
                lip_points = live_landmarks[0]
                reference_point = lip_points[0]
                distances = [np.linalg.norm(np.array(reference_point) - np.array(point)) for point in lip_points]

            similarity_scores = []

            # Iterate over stored files in the directory
            for file_name in os.listdir(stored_landmarks_dir):
                stored_landmarks_path = os.path.join(stored_landmarks_dir, file_name)
                if not os.path.isfile(stored_landmarks_path) or not file_name.endswith('.txt'):
                    continue

                with open(stored_landmarks_path, 'r') as file:
                    stored_landmarks = [
                        tuple(map(float, line.strip().split(',')))
                        for line in file.readlines()
                    ]

                # Compare live relative distances with stored data
                similarity_percentage = 100 - (
                        sum([np.linalg.norm(np.array(a) - np.array(b)) for a, b in
                             zip(distances, stored_landmarks)]) / len(distances))
                similarity_scores.append(similarity_percentage)

            if similarity_scores:
                best_similarity = max(similarity_scores)
                print(f"Frame {frame_count + 1}: Best Similarity: {best_similarity:.2f}%")

                if best_similarity > 90:
                    success_count += 1

            frame_count += 1

        if success_count >= 8:
            print("Success")
            break

        start_comparison = False

    if key == ord('q'):
        # Quit when 'q' is pressed
        break

    if lip_counter >= 10:
        time.sleep(0.05)  # Wait for 0.05 seconds before capturing next frame

cap.release()
cv2.destroyAllWindows()
