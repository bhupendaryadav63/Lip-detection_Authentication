import cv2
import dlib
import numpy as np
import os
import time

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

successful_matches_count = 0

print("Press 'v' to start capturing live frames.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Press V to Start', frame)
    
    key = cv2.waitKey(1)
    if key == ord('v'):
        print("Comparison started...")
        break
    elif key == 27:
        print("Exiting...")
        exit()

while True:
    frame_count = 0
    success_count = 0
    best_similarity_scores = []

    while frame_count < 10:  # Capture live data for 10 frames
        time.sleep(0.5)

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        live_landmarks = []

        for face in faces:
            landmarks = predictor(gray, face)
            lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            x, y, w, h = cv2.boundingRect(np.array(lip_points))
            cx, cy = x + w // 2, y + h // 2
            top_x, top_y = cx - 100, cy - 75
            lip_crop = frame[top_y:top_y + 150, top_x:top_x + 200]
            frame[:150, :200] = lip_crop
            
            stored_landmarks_paths = [
                f"dot/image_{i}/landmarks_{i}.txt"
                for i in range(1, 6)  # Assuming there are five sets of stored data
            ]

            for stored_landmarks_path in stored_landmarks_paths:
                if not os.path.exists(stored_landmarks_path):
                    print(f"Data file not found: {stored_landmarks_path}")
                    continue

                stored_lip_points = extract_landmarks_data(stored_landmarks_path)

                similarity = compare_landmarks(lip_points, stored_lip_points)
                print(f"Similarity with {stored_landmarks_path}: {similarity:.2f}%")

                if similarity > 80:
                    print("Match found! Take action...")
                    successful_matches_count += 1

            frame_count += 1

    if successful_matches_count >= 8:
        print("Success")
    else:
        print("Failure")

    cv2.destroyAllWindows()
    break
