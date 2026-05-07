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

OUTPUT_DIR = os.path.join(BASE_DIR, 'output-png', 'output_v10')
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def get_pupil_v10(roi_gray, fw, profile="A"):
    """
    Profile A: Avoid eyebrows (Strict ROI shifting)
    Profile B: Normal sensitivity for smaller features
    """
    ideal_r = int(fw * 0.018)
    k = 11 if profile == "A" else 7
    roi_blurred = cv2.GaussianBlur(roi_gray, (k, k), 0)
    
    # In Profile A, skip top 20% to avoid eyebrow overlap
    h, w = roi_gray.shape
    search_roi = roi_blurred[int(h*0.2):, :] if profile == "A" else roi_blurred
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(search_roi)
    actual_y = min_loc[1] + (int(h*0.2) if profile == "A" else 0)
    
    return (min_loc[0], actual_y, ideal_r)

data_list = []
image_paths = []
for ext in ['*.jpg', '*.png']:
    image_paths.extend(glob.glob(os.path.join(BASE_DIR, 'Database', ext)))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_en = clahe.apply(gray)
    display_img = img.copy()
    
    # Standard Cascade Detection
    faces = face_cascade.detectMultiScale(gray_en, 1.1, 5, minSize=(120, 120))
    
    pupils = []
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (fx, fy, fw, fh) = faces[0]
        cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
        
        # Adaptive Case: Shift ROI down to 25% of face height to avoid eyebrows
        eye_y_start, eye_h = int(fh * 0.25), int(fh * 0.30)
        eye_zones = [
            {'name': 'Left', 'box': (int(fw*0.08), eye_y_start, int(fw*0.42), eye_h)},
            {'name': 'Right', 'box': (int(fw*0.5), eye_y_start, int(fw*0.42), eye_h)}
        ]
        
        face_pupils = []
        for zone in eye_zones:
            zx, zy, zw, zh = zone['box']
            zone_gray = gray_en[fy+zy:fy+zy+zh, fx+zx:fx+zx+zw]
            res = get_pupil_v10(zone_gray, fw, profile="A")
            if res:
                gx, gy = fx+zx+res[0], fy+zy+res[1]
                face_pupils.append({'x': gx, 'y': gy, 'r': res[2]})
                cv2.circle(display_img, (gx, gy), res[2], (0, 255, 0), 2)
                cv2.circle(display_img, (gx, gy), 2, (0, 0, 255), -1)
        
        # Symmetry Fallback
        if len(face_pupils) == 1:
            p1 = face_pupils[0]
            cx = fx + fw//2
            nx, ny = cx - (p1['x'] - cx), p1['y']
            face_pupils.append({'x': int(nx), 'y': int(ny), 'r': p1['r']})
            cv2.circle(display_img, (int(nx), int(ny)), p1['r'], (0, 255, 128), 2)
        pupils = face_pupils
    else:
        # NIR Fallback (Parameters from Code_v9.py)
        _, thresh = cv2.threshold(gray_en, 35, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in cnts:
            if 400 < cv2.contourArea(cnt) < 15000:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    candidates.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
        candidates.sort(key=lambda c: c[1])
        for i in range(len(candidates)-1):
            c1, c2 = candidates[i], candidates[i+1]
            if abs(c1[1] - c2[1]) < 80 and 400 < abs(c1[0] - c2[0]) < 1800:
                pupils = [{'x': c1[0], 'y': c1[1], 'r': 40}, {'x': c2[0], 'y': c2[1], 'r': 40}]
                for p in pupils: cv2.circle(display_img, (p['x'], p['y']), p['r'], (0, 0, 255), 3)
                break

    dist = None
    if len(pupils) >= 2:
        pupils.sort(key=lambda p: p['x'])
        dist = math.sqrt((pupils[0]['x']-pupils[1]['x'])**2 + (pupils[0]['y']-pupils[1]['y'])**2)
        cv2.line(display_img, (pupils[0]['x'], pupils[0]['y']), (pupils[1]['x'], pupils[1]['y']), (255, 0, 0), 2)
        cv2.putText(display_img, f"{dist:.1f}px", (pupils[0]['x'], pupils[0]['y']-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    data_list.append({'filename': os.path.basename(img_path), 'pupils': len(pupils), 'dist': dist})
    cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), display_img)

df = pd.DataFrame(data_list)
df.to_csv(os.path.join(CASCADE_DIR, 'final_v10_analysis.csv'), index=False)
print("V10 with v9-NIR Fallback Complete.")
