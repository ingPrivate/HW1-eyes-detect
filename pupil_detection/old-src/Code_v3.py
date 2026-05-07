#!/usr/bin/env python
import cv2
import numpy as np
import glob
import pandas as pd
import os
import math

# Updated Paths to cascades
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_DIR = os.path.join(BASE_DIR, 'test-results')
FACE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

# Updated Output Directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-png', 'output_v3')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

data_list = []

# Search for both .jpg and .png
image_extensions = ['*.jpg', '*.png']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(BASE_DIR, 'Database', ext)))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None: continue
    
    display_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    all_pupils = []
    
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2) # Face box
        face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
        face_roi_color = display_img[fy:fy+fh, fx:fx+fw]
        
        # 2. Detect Eyes within Face ROI
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10, minSize=(30, 30))
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2) # Eye ROI box
            
            # 3. Pupil Detection within Eye ROI
            eye_gray = face_roi_gray[ey:ey+eh, ex:ex+ew]
            eye_blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
            
            # Use adaptive threshold to find the darkest part (pupil)
            thresh = cv2.adaptiveThreshold(eye_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 21, 5)
            
            # Hough Circles specifically for this eye
            diag = math.sqrt(ew**2 + eh**2)
            circles = cv2.HoughCircles(eye_blurred, cv2.HOUGH_GRADIENT, 1, diag,
                                      param1=50, param2=15, minRadius=int(ew*0.05), maxRadius=int(ew*0.3))
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Take the best candidate
                best_c = circles[0, 0]
                cx, cy, cr = best_c
                
                # Global coordinates
                global_x = fx + ex + cx
                global_y = fy + ey + cy
                
                all_pupils.append({'x': int(global_x), 'y': int(global_y), 'r': int(cr)})
                cv2.circle(display_img, (int(global_x), int(global_y)), int(cr), (0, 255, 0), 2)

    # 4. Calculate Distance
    dist = None
    if len(all_pupils) >= 2:
        # Sort by x to get Left and Right
        all_pupils = sorted(all_pupils, key=lambda p: p['x'])
        p1, p2 = all_pupils[0], all_pupils[1]
        dist = math.sqrt((p1['x']-p2['x'])**2 + (p1['y']-p2['y'])**2)
        cv2.line(display_img, (p1['x'], p1['y']), (p2['x'], p2['y']), (255, 0, 0), 2)
        cv2.putText(display_img, f"Dist: {dist:.1f}px", (p1['x'], p1['y']-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    data_list.append({
        'filename': os.path.basename(img_path),
        'face_detected': len(faces) > 0,
        'pupils_found': len(all_pupils),
        'distance_px': dist
    })
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), display_img)

df = pd.DataFrame(data_list)
# Save CSV to test-results
df.to_csv(os.path.join(CASCADE_DIR, 'roi_pupil_analysis.csv'), index=False)
print("ROI-based analysis complete.")
print(df.to_string())
