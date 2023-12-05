import os
import dlib
import cv2
import numpy as np
import json

# Load the pre-trained facial landmark predictor
predictor_path = "Model/shape_predictor_68_face_landmarks (1).dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Function to calculate lip features
import numpy as np

import numpy as np

def calculate_lip_features(landmarks, precision=1):
    # Define points for upper and lower lip
    upper_lip_points = list(range(48, 55))  # Points for upper lip
    lower_lip_points = list(range(54, 61))  # Points for lower lip

    # Calculate lip thickness
    upper_lip_thickness = round(np.mean([landmarks.part(i).y for i in upper_lip_points]) - landmarks.part(51).y, precision)
    lower_lip_thickness = round(landmarks.part(57).y - np.mean([landmarks.part(i).y for i in lower_lip_points]), precision)

    # Calculate lip width and height
    lip_width = round(landmarks.part(54).x - landmarks.part(48).x, precision)
    lip_height = round(landmarks.part(66).y - landmarks.part(62).y, precision)

    # Calculate Cupid's Bow
    cupid_bow = round(landmarks.part(53).y - landmarks.part(50).y, precision)

    # Calculate lip curvature
    lip_curvature = round(landmarks.part(54).y - landmarks.part(51).y, precision)

    # Calculate symmetry between upper and lower lip
    upper_lower_symmetry = round(np.mean([landmarks.part(i).y for i in upper_lip_points]) - np.mean([landmarks.part(i).y for i in lower_lip_points]), precision)

    # Calculate proportions and ratios between upper and lower lips
    upper_lower_ratio = round(upper_lip_thickness / lower_lip_thickness, precision)

    # Calculate distances between lip points for variance and mean
    lip_points = [landmarks.part(i) for i in range(48, 61)]  # All lip points
    distances = []
    for p1 in lip_points:
        for p2 in lip_points:
            if lip_points.index(p1) < lip_points.index(p2):
                distance = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
                distances.append(distance)
    
    mean_distance = round(np.mean(distances), precision)
    variance_distance = round(np.var(distances), precision)

    return upper_lip_thickness, lower_lip_thickness, lip_width, lip_height, cupid_bow, lip_curvature, upper_lower_symmetry, upper_lower_ratio, mean_distance, variance_distance

# Function to draw an ellipse on the frame
def draw_ellipse(frame, center, major_axis, minor_axis, angle):
    ellipses_axes = (major_axis, minor_axis)  # Combine major and minor axis into a tuple
    cv2.ellipse(frame, center, ellipses_axes, angle, 0, 360, (255, 0, 0), 2)
    # Define a slightly shifted center
    shifted_center = (center[0], center[1] + 40)
    
    # Draw a dot at the shifted center of the ellipse
    cv2.circle(frame, shifted_center, 3, (255, 0, 0), -1)

def create_lip_features_dict(landmarks):
    features = calculate_lip_features(landmarks)
    features_dict = {
        'Upper Lip Thickness': features[0],
        'Lower Lip Thickness': features[1],
        'Lip Width': features[2],
        'Lip Height': features[3],
        'Cupid Bow': features[4],
        'Lip Curvature': features[5],
        'Upper-Lower Symmetry': features[6],
        'Upper-Lower Ratio': features[7],
        'Mean Distance': features[8],
        'Variance Distance': features[9]
    }
    return features_dict

# Create a folder to save images
save_folder = "image_save"
os.makedirs(save_folder, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Define ellipse parameters
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    major_axis = 150  # Length of the major axis
    minor_axis = 200  # Length of the minor axis
    angle = 0  # Angle of rotation in degrees
    # Draw the ellipse on the frame
    draw_ellipse(frame, center, major_axis, minor_axis, angle)


    for face in faces:
        landmarks = predictor(gray, face)
        lip_features_dict = create_lip_features_dict(landmarks)

        for i in range(48, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Extract lip points (48 to 68)
        lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

        # Get bounding box dimensions for cropping
        x, y, w, h = cv2.boundingRect(np.array(lip_points))
        lip_crop = frame[y:y + h, x:x + w]

        # Create a box of size 150x200 at the top-left corner
        box = np.zeros((150, 200, 3), dtype=np.uint8)
        box[:min(h, 150), :min(w, 200)] = lip_crop[:min(h, 150), :min(w, 200)]
        frame[:150, :200] = box

        if cv2.waitKey(1) & 0xFF == ord('s'):
            file_name = os.path.join(save_folder, f"lip_features_{len(os.listdir(save_folder)) + 1}.json")
            print(f"lip_features_{len(os.listdir(save_folder)) + 1}.json saved in directory")
            with open(file_name, "w") as file:
                json.dump(lip_features_dict, file)
        if cv2.waitKey(1) & 0xFF == ord('f'):
            for i in range(48, 68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    cv2.imshow("Face Landmarks", frame)
   

