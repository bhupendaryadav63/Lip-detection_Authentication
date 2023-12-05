import cv2
import dlib
import os
import numpy as np

def extract_stats_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        mean_distance = float(lines[0].strip().split(':')[1])
        variance_distance = float(lines[1].strip().split(':')[1])
        hausdorff_distance = float(lines[2].strip().split(':')[1])
    return mean_distance, variance_distance, hausdorff_distance

def extract_landmarks_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        landmarks = [
            tuple(map(int, line.strip().split(',')))
            for line in lines
        ]
    return landmarks

def compare_landmarks(live_landmarks, stored_landmarks):
    max_landmark_distance = 100  # Assuming the maximum distance between landmarks is 100

    sum_squared_distances = 0
    for live_point, stored_point in zip(live_landmarks, stored_landmarks):
        squared_distance = calculate_distance(live_point, stored_point) ** 2
        sum_squared_distances += squared_distance

    rmse = np.sqrt(sum_squared_distances / len(live_landmarks))
    normalized_similarity_score = 1 - rmse / (2 * max_landmark_distance)
    percentage_similarity_score = normalized_similarity_score * 100
    percentage_similarity_score = max(1, min(percentage_similarity_score, 100))

    return percentage_similarity_score

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks (1).dat")

cap = cv2.VideoCapture(0)
output_folder = 'dot'
capture_lip_shape = False
image_counter = 1
show_lip_landmarks = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    lip_points = []
    for face in faces:
        landmarks = predictor(gray, face)
        lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
        x, y, w, h = cv2.boundingRect(np.array(lip_points))
        cx, cy = x + w // 2, y + h // 2
        top_x, top_y = cx - 100, cy - 75
        lip_crop = frame[top_y:top_y + 150, top_x:top_x + 200]
        frame[:150, :200] = lip_crop

    key = cv2.waitKey(1)
    if key & 0xFF == ord('f'):
        show_lip_landmarks = not show_lip_landmarks

    if show_lip_landmarks:
        for point in lip_points:
            cv2.circle(frame, point, 1, (0, 0, 255), -1)
    
    if capture_lip_shape:
        for face in faces:
            landmarks = predictor(gray, face)
            live_lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

            stored_landmarks_paths = [
                f"dot/image_{i}/landmarks_{i}.txt"
                for i in range(1, 6)  # Assuming there are five sets of stored data
            ]

            for stored_landmarks_path in stored_landmarks_paths:
                if not os.path.exists(stored_landmarks_path):
                    print(f"Data file not found: {stored_landmarks_path}")
                    continue

                stored_lip_points = extract_landmarks_data(stored_landmarks_path)

                similarity = compare_landmarks(live_lip_points, stored_lip_points)
                print(f"Similarity with {stored_landmarks_path}: {similarity:.2f}%")

                # Example: If similarity > threshold, perform an action
                if similarity > 80:
                    print("Match found! Take action...")

            # Save the live lip data or perform further actions based on matches
            # Example: Save the live lip data for future comparison
            folder_path = os.path.join(output_folder, f'image_{image_counter}')
            os.makedirs(folder_path, exist_ok=True)
            live_lip_file_path = os.path.join(folder_path, f'live_lip_{image_counter}.txt')
            with open(live_lip_file_path, 'w') as file:
                for point in live_lip_points:
                    file.write(f"{point[0]}, {point[1]}\n")

            capture_lip_shape = False
            image_counter += 1

    cv2.imshow('Lip Landmarks', frame)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        capture_lip_shape = True

cap.release()
cv2.destroyAllWindows()