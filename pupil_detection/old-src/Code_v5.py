#!/usr/bin/env python
import cv2
import numpy as np
import glob
import pandas as pd
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_DIR = os.path.join(BASE_DIR, 'test-results')
FACE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

OUTPUT_DIR = os.path.join(BASE_DIR, 'output-png', 'output_v5')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
    
    # 1. Image Enhancement: CLAHE
    gray_enhanced = clahe.apply(gray)
    
    # 2. Detect Face
    faces = face_cascade.detectMultiScale(gray_enhanced, 1.1, 4)
    
    final_pupils = []
    
    for (fx, fy, fw, fh) in faces:
        face_center_x = fx + fw // 2
        cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
        face_roi_gray = gray_enhanced[fy:fy+fh, fx:fx+fw]
        
        # 3. Detect Eyes
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10, minSize=(30, 30))
        
        current_face_pupils = []
        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi_gray[ey:ey+eh, ex:ex+ew]
            eye_blurred = cv2.GaussianBlur(eye_roi, (5, 5), 0)
            
            # --- HYBRID DETECTION: Ellipse + Hough Fallback ---
            # Try Ellipse Fitting first
            _, eye_thresh = cv2.threshold(eye_blurred, 40, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_p = None
            for cnt in contours:
                if len(cnt) >= 5:
                    area = cv2.contourArea(cnt)
                    if 30 < area < (ew * eh * 0.3):
                        ellipse = cv2.fitEllipse(cnt)
                        (center, axes, angle) = ellipse
                        ratio = min(axes) / max(axes) if max(axes) > 0 else 0
                        if ratio > 0.6:
                            best_p = (int(center[0]), int(center[1]), int(max(axes)/2))
                            break
            
            # If Ellipse fails, try Hough
            if best_p is None:
                diag = math.sqrt(ew**2 + eh**2)
                circles = cv2.HoughCircles(eye_blurred, cv2.HOUGH_GRADIENT, 1, diag,
                                          param1=50, param2=12, minRadius=int(ew*0.05), maxRadius=int(ew*0.3))
                if circles is not None:
                    c = circles[0, 0]
                    best_p = (int(c[0]), int(c[1]), int(c[2]))

            if best_p:
                gx = fx + ex + best_p[0]
                gy = fy + ey + best_p[1]
                current_face_pupils.append({'x': gx, 'y': gy, 'r': best_p[2]})
                cv2.circle(display_img, (gx, gy), best_p[2], (0, 255, 0), 2)

        # 4. Pupil Center Correlation (Fallback for symmetry)
        if len(current_face_pupils) == 1:
            p1 = current_face_pupils[0]
            dx = p1['x'] - face_center_x
            corr_x = face_center_x - dx
            corr_y = p1['y']
            current_face_pupils.append({'x': int(corr_x), 'y': int(corr_y), 'r': p1['r']})
            cv2.drawMarker(display_img, (int(corr_x), int(corr_y)), (255, 100, 0), cv2.MARKER_CROSS, 20, 2)

        final_pupils.extend(current_face_pupils)

    # 5. Distance Calculation
    dist = None
    if len(final_pupils) >= 2:
        final_pupils = sorted(final_pupils, key=lambda p: p['x'])
        p1, p2 = final_pupils[0], final_pupils[1]
        dist = math.sqrt((p1['x']-p2['x'])**2 + (p1['y']-p2['y'])**2)
        cv2.line(display_img, (p1['x'], p1['y']), (p2['x'], p2['y']), (255, 0, 0), 2)
        cv2.putText(display_img, f"Dist: {dist:.1f}px", (p1['x'], p1['y']-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    data_list.append({'filename': os.path.basename(img_path), 'pupils': len(final_pupils), 'dist': dist})
    cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), display_img)

df = pd.DataFrame(data_list)
df.to_csv(os.path.join(CASCADE_DIR, 'hybrid_pupil_analysis.csv'), index=False)
print("Hybrid analysis complete.")
print(df.to_string())
