
import json

# Function to extract lip features from a text file
def extract_lip_features(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract lip features from the text file
    upper_lip_thickness = float(lines[0].split(':')[1].strip())
    lower_lip_thickness = float(lines[1].split(':')[1].strip())
    lip_width = int(lines[2].split(':')[1].strip())
    lip_height = int(lines[3].split(':')[1].strip())

    # Extract relative distances (ignoring the header)
    relative_distances = [tuple(map(int, distance.strip('()\n').split(','))) for distance in lines[5:]]

    return upper_lip_thickness, lower_lip_thickness, lip_width, lip_height, relative_distances

# Load data from JSON files
with open('image_save/lip_features_1.json', 'r') as file:
    data1 = json.load(file)

with open('image_livesave/lip_features_10.json', 'r') as file:
    data2 = json.load(file)

# Define weights for each parameter
weights = {
    "Upper Lip Thickness": 0.20,
    "Lower Lip Thickness": 0.20,
    "Lip Width": 0.5,
    "Lip Height": 0.5,
    "Cupid Bow": 0.15,
    "Lip Curvature": 0.15,
    "Upper-Lower Symmetry": 0.05,
    "Upper-Lower Ratio": 0.05,
    "Mean Distance": 0.05,
    "Variance Distance": 0.05
}

# Function to calculate similarity score
def calculate_similarity(data1, data2, weights):
    similarity = 0.0
    total_weight = sum(weights.values())

    for key in weights:
        value1 = data1[key]
        value2 = data2[key]

        # Calculate similarity for each feature separately
        if value1 != 0 or value2 != 0:
            similarity += abs(value1 - value2) / max(value1, value2) * (weights[key] / total_weight)

    return (1 - (similarity / len(weights))) * 100

# Calculate similarity score
similarity_score = calculate_similarity(data1, data2, weights)
print(f"Similarity score: {similarity_score:.2f}%")

'''
import os
import json
from scipy.spatial.distance import euclidean

# Path to the folders containing saved and live lip features
save_features_folder = "image_save"
live_features_folder = "image_livesave"

# Load saved features
saved_features = []
for filename in os.listdir(save_features_folder):
    file_path = os.path.join(save_features_folder, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
        saved_features.append(data)

# Load live features
live_features = []
for i in range(1, 11):  # Assuming 10 live frames were captured
    file_path = os.path.join(live_features_folder, f"lip_features_{i}.json")
    with open(file_path, 'r') as file:
        data = json.load(file)
        live_features.append(data)

# Function to calculate similarity between two feature sets
# Function to calculate similarity between two feature sets
# Function to calculate similarity between two feature sets
def calculate_similarity(feature1, feature2):
    squared_distances = []

    # Iterate through the keys in the dictionaries and calculate squared distances
    for key in feature1.keys():
        squared_distance = (feature1[key] - feature2[key]) ** 2
        squared_distances.append(squared_distance)

    # Calculate the Euclidean distance
    euclidean_distance = sum(squared_distances) ** 0.5

    # Normalize similarity
    max_distance = max(max(feature1.values()), max(feature2.values()))
    similarity = 1 - (euclidean_distance / max_distance)
    
    return similarity



# Calculate similarity for each live frame with saved features
similarities = []
for live_feature in live_features:
    frame_similarities = [calculate_similarity(live_feature, saved_feature) for saved_feature in saved_features]
    frame_similarity = sum(frame_similarities) / len(frame_similarities)
    similarities.append(frame_similarity)

# Print individual frame similarities and average similarity
for i, similarity in enumerate(similarities, start=1):
    print(f"Frame {i} Similarity: {similarity}")

average_similarity = sum(similarities) / len(similarities)
print(f"Average Similarity across all frames: {average_similarity}")
'''