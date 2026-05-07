#!/usr/bin/env python
import cv2
import numpy as np
import glob
import pandas as pd
import os
import math

# Paths to cascades
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_DIR = os.path.join(BASE_DIR, 'test-results')
FACE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

OUTPUT_DIR = os.path.join(BASE_DIR, 'output-png', 'output_v4')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

data_list = []
image_paths = []
for ext in ['*.jpg', '*.png']:
    image_paths.extend(glob.glob(os.path.join(BASE_DIR, 'Database', ext)))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None: continue
    
    display_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # A. Image Enhancement: CLAHE
    gray_enhanced = clahe.apply(gray)
    
    # 1. Detect Face
    faces = face_cascade.detectMultiScale(gray_enhanced, 1.1, 4)
    
    final_pupils = []
    
    for (fx, fy, fw, fh) in faces:
        face_center_x = fx + fw // 2
        cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
        
        face_roi_gray = gray_enhanced[fy:fy+fh, fx:fx+fw]
        face_roi_color = display_img[fy:fy+fh, fx:fx+fw]
        
        # 2. Detect Eyes
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10, minSize=(30, 30))
        
        current_face_pupils = []
        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi_gray[ey:ey+eh, ex:ex+ew]
            
            # C. Ellipse Fitting Strategy
            # Use thresholding to find the dark pupil
            _, eye_thresh = cv2.threshold(eye_roi, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_ellipse = None
            max_circularity = 0
            
            for cnt in contours:
                if len(cnt) >= 5: # Need 5 points for ellipse fitting
                    area = cv2.contourArea(cnt)
                    if 50 < area < (ew * eh * 0.3):
                        ellipse = cv2.fitEllipse(cnt)
                        (center, axes, angle) = ellipse
                        # Calculate circularity/aspect ratio
                        ratio = min(axes) / max(axes) if max(axes) > 0 else 0
                        if ratio > 0.6 and ratio > max_circularity:
                            max_circularity = ratio
                            best_ellipse = ellipse
            
            if best_ellipse:
                (ecx, ecy), (ew_e, eh_e), e_angle = best_ellipse
                gx = fx + ex + int(ecx)
                gy = fy + ey + int(ecy)
                current_face_pupils.append({'x': gx, 'y': gy, 'type': 'detected'})
                cv2.ellipse(display_img, (gx, gy), (int(ew_e/2), int(eh_e/2)), e_angle, 0, 360, (0, 255, 0), 2)
        
        # Pupil Center Correlation (Facial Symmetry)
        if len(current_face_pupils) == 1:
            p1 = current_face_pupils[0]
            dx = p1['x'] - face_center_x
            corr_x = face_center_x - dx
            corr_y = p1['y'] # Assume horizontal alignment
            current_face_pupils.append({'x': int(corr_x), 'y': int(corr_y), 'type': 'correlated'})
            cv2.drawMarker(display_img, (int(corr_x), int(corr_y)), (255, 100, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(display_img, "Correlated", (int(corr_x)-30, int(corr_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        final_pupils.extend(current_face_pupils)

    # Calculate Distance
    dist = None
    if len(final_pupils) >= 2:
        final_pupils = sorted(final_pupils, key=lambda p: p['x'])
        p1, p2 = final_pupils[0], final_pupils[1]
        dist = math.sqrt((p1['x']-p2['x'])**2 + (p1['y']-p2['y'])**2)
        cv2.line(display_img, (p1['x'], p1['y']), (p2['x'], p2['y']), (255, 0, 0), 2)

    data_list.append({
        'filename': os.path.basename(img_path),
        'pupils_found': len(final_pupils),
        'distance_px': dist
    })
    cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), display_img)

df = pd.DataFrame(data_list)
df.to_csv(os.path.join(CASCADE_DIR, 'enhanced_pupil_analysis.csv'), index=False)
print("Enhanced analysis complete.")
print(df.to_string())
