# Define a slightly shifted center
    shifted_center = (center[0], center[1] + 40)
    
    # Draw a dot at the shifted center of the ellipse
    cv2.circle(frame, shifted_center, 3, (255, 0, 0), -1