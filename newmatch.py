import cv2
import dlib
import numpy as np
import os
import time

# Function to extract statistical data from the stored file
def extract_stats_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        mean_distance = float(lines[0].strip().split(':')[1])
        variance_distance = float(lines[1].strip().split(':')[1])
        hausdorff_distance = float(lines[2].strip().split(':')[1])
    return mean_distance, variance_distance, hausdorff_distance

# Function to extract landmarks data from the stored file
def extract_landmarks_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        landmarks = [
            tuple(map(int, line.strip().split(',')))
            for line in lines
        ]
    return landmarks

# Function to calculate the similarity between live and stored landmarks
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

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Initialize dlib's face detector and shape predictor
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
            lip_points_live = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            live_landmarks.append(lip_points_live)

        stored_landmarks_paths = [
            f"dot/image_{i}/landmarks_{i}.txt"
            for i in range(1, 6)  # Assuming there are five sets of stored data
        ]

        similarity_scores = []

        for stored_landmarks_path in stored_landmarks_paths:
            if not os.path.exists(stored_landmarks_path):
                print(f"Data file not found: {stored_landmarks_path}")
                continue

            stored_landmarks_distances = extract_landmarks_data(stored_landmarks_path)
            similarity_percentage = compare_landmarks(live_landmarks[0], stored_landmarks_distances)
            similarity_scores.append(similarity_percentage)

            print(f"Frame {frame_count + 1}: Similarity Score: {similarity_percentage:.2f}%")

        if similarity_scores:
            best_similarity = max(similarity_scores)
            best_similarity_scores.append(best_similarity)

            print(f"Frame {frame_count + 1}: Best Similarity Score: {best_similarity:.2f}%")

            if best_similarity > 80:
                success_count += 1

        frame_count += 1

    if best_similarity_scores:
        average_similarity = sum(best_similarity_scores) / len(best_similarity_scores)
        print(f"Average Best Similarity across frames: {average_similarity:.2f}%")

    if success_count >= 8:
        print("Success")
    else:
        print("Failure")

    cv2.destroyAllWindows()
    break
