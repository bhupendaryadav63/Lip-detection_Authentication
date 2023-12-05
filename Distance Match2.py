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

# Function to calculate similarity between two feature sets
def calculate_similarity(feature1, feature2, weights):
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

    # Check if the denominator is zero
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0

    # Calculate the weighted similarity
    weighted_similarity = 0
    for key in weights:
        if max(feature1[key], feature2[key]) != 0:
            similarity_term = (1 - (abs(feature1[key] - feature2[key]) / max(feature1[key], feature2[key]))) * (weights[key] / total_weight)
            weighted_similarity += similarity_term
    
    return weighted_similarity


# Match live frames with saved features
success_count = 0
for live_frame in live_features:
    best_similarity = 0
    best_match_index = -1
    
    for idx, saved_feature in enumerate(saved_features):
        similarity_score = calculate_similarity(live_frame, saved_feature, weights)
        if similarity_score > best_similarity:
            best_similarity = similarity_score
            best_match_index = idx
    
    if best_similarity >= 0.9:
        success_count += 1
    
if success_count >= 8:
    print("\nSuccess! \nMatch Found above 90%.")
else:
    print("\n\nUnsucessful, \nSimilarity very low.")
