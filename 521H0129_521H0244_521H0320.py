import cv2
import numpy as np
import os

# Dictionary to store template images
SIGN_TEMPLATES = {}

# Dictionary to map template sign names to new names
SIGN_NAME_MAPPING = {
    "camnguocchieu": "Wrong way",
    "wrongway": "Wrong way",
    "huongdi1": "Keep right",
    "keepright": "Keep right",
    "no_left": "No left turn",
    "noleftturn":"No left turn",
    "camdungxe": "No parking",
    "noparking": "No parking",
    "children": "Children",
    "slow": "Slow",
    "nostopandparking": "No stopping and parking",
    "camdungxedonha":"No parking",
    "camdungxe":"No parking"
}

def load_templates(template_dir='sign_templates'):
    """Load template images from directory"""
    for filename in os.listdir(template_dir):
        if filename.endswith(('.png', '.jpg')):
            sign_name = os.path.splitext(filename)[0]
            template_path = os.path.join(template_dir, filename)
            template = cv2.imread(template_path)
            SIGN_TEMPLATES[sign_name] = template

def match_template(roi, template, target_size=(100, 100)):
    """Compare ROI with template using template matching"""
    # Resize both images to same size
    roi_resized = cv2.resize(roi, target_size)
    template_resized = cv2.resize(template, target_size)
    
    # Convert both to grayscale
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
    
    # Template matching
    result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    return np.max(result)

def detect_traffic_signs(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    # Single red range
    lower_red = np.array([150, 50, 50])  # Lower bound of red
    upper_red = np.array([180, 255, 255])  # Upper bound of red
    # Single blue range
    lower_blue = np.array([90, 120, 100])  # Lower bound of blue
    upper_blue = np.array([130, 255, 255])  # Upper bound of blue
    # Yellow range
    lower_yellow = np.array([0, 120, 0])  # Lower bound of yellow
    upper_yellow = np.array([30, 255, 255])  # Upper bound of yellow
    
    # Create red, blue, and yellow masks
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine the red, blue, and yellow masks
    combined_mask = cv2.bitwise_or(red_mask, blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
    
    # Noise removal (Morphological operations)
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_signs = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            if 0.8 <= aspect_ratio <= 1.5:
                # Extract ROI
                roi = frame[y:y + h, x:x + w]
                
                # Get color type (Red, Blue, or Yellow)
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                red_pixels = cv2.countNonZero(cv2.inRange(roi_hsv, lower_red, upper_red))
                blue_pixels = cv2.countNonZero(cv2.inRange(roi_hsv, lower_blue, upper_blue))
                yellow_pixels = cv2.countNonZero(cv2.inRange(roi_hsv, lower_yellow, upper_yellow))
                
                # Determine the color type
                if red_pixels > max(blue_pixels, yellow_pixels):
                    color_type = "Red"
                elif blue_pixels > max(red_pixels, yellow_pixels):
                    color_type = "Blue"
                else:
                    color_type = "Yellow"
                
                # Template matching
                best_match = None
                best_score = 0
                
                for sign_name, template in SIGN_TEMPLATES.items():
                    score = match_template(roi, template)
                    if score > best_score and score > 0.55:
                        best_score = score
                        best_match = sign_name
                
                if best_match:
                    # Map the sign name to a custom name
                    new_sign_name = SIGN_NAME_MAPPING.get(best_match, best_match)  # If no mapping, use the original name
                    sign_info = f"{new_sign_name} ({color_type})"
                    detected_signs.append((x, y, w, h, sign_info))
    
    return detected_signs

def process_video(video_path):
    # Load template images
    load_templates()
    
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter('output_video2.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        signs = detect_traffic_signs(frame)
        
        for (x, y, w, h, sign_name) in signs:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, sign_name, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        
        cv2.putText(frame, "Student ID: 521H0129_521H0244_521H0320", (10, 30),  
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)   
        
        cv2.imshow("Processed Video", frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Create directory structure
if not os.path.exists('sign_templates'):
    os.makedirs('sign_templates')

# Process video
process_video('video2.mp4')
