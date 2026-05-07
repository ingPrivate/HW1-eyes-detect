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

OUTPUT_DIR = os.path.join(BASE_DIR, 'output-png', 'output_v8')
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def get_pupil_v8(roi_gray, ew, eh):
    """Refined pupil detection with darkest point fallback."""
    roi_blurred = cv2.GaussianBlur(roi_gray, (7, 7), 0)
    
    # 1. Try Ellipse Fitting
    _, thresh = cv2.threshold(roi_blurred, 40, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if 30 < area < (ew * eh * 0.3):
                ellipse = cv2.fitEllipse(cnt)
                ratio = min(ellipse[1]) / max(ellipse[1]) if max(ellipse[1]) > 0 else 0
                if ratio > 0.65:
                    return (int(ellipse[0][0]), int(ellipse[0][1]), int(max(ellipse[1])/2), "ellipse")
    
    # 2. Try Hough Circles
    circles = cv2.HoughCircles(roi_blurred, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=10, 
                              minRadius=int(ew*0.05), maxRadius=int(ew*0.25))
    if circles is not None:
        c = circles[0, 0]
        return (int(c[0]), int(c[1]), int(c[2]), "hough")
        
    # 3. Fallback: Darkest Point Search (Min-Max)
    # Pupils are usually the darkest part of the eye ROI
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi_blurred)
    if min_val < 80: # Ensure it's reasonably dark
        # Estimated radius based on eye width
        est_r = int(ew * 0.08)
        return (min_loc[0], min_loc[1], est_r, "darkspot")
        
    return None

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
    
    # 1. Detect Face
    faces = face_cascade.detectMultiScale(gray_en, 1.1, 5, minSize=(120, 120))
    
    pupils = []
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (fx, fy, fw, fh) = faces[0]
        cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
        
        # Heuristic Eye Regions (Left Zone / Right Zone)
        # These zones will be used if cascade fails
        eye_zones = [
            {'name': 'Left', 'box': (int(fw*0.1), int(fh*0.2), int(fw*0.4), int(fh*0.35))},
            {'name': 'Right', 'box': (int(fw*0.5), int(fh*0.2), int(fw*0.4), int(fh*0.35))}
        ]
        
        face_pupils = []
        for zone in eye_zones:
            zx, zy, zw, zh = zone['box']
            zone_gray = gray_en[fy+zy:fy+zy+zh, fx+zx:fx+zx+zw]
            
            # Try Eye Cascade in this zone
            eyes = eye_cascade.detectMultiScale(zone_gray, 1.1, 5, minSize=(zw//4, zh//4))
            
            best_res = None
            if len(eyes) > 0:
                # Use detected eye ROI
                ex, ey, ew, eh = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[0]
                eye_roi = zone_gray[ey:ey+eh, ex:ex+ew]
                res = get_pupil_v8(eye_roi, ew, eh)
                if res:
                    best_res = (fx+zx+ex+res[0], fy+zy+ey+res[1], res[2], res[3])
            
            if best_res is None:
                # Fallback: Search the whole zone directly
                res = get_pupil_v8(zone_gray, zw, zh)
                if res:
                    best_res = (fx+zx+res[0], fy+zy+res[1], res[2], res[3])
            
            if best_res:
                face_pupils.append({'x': best_res[0], 'y': best_res[1], 'r': best_res[2]})
                color = (0, 255, 0) if best_res[3] != "darkspot" else (0, 255, 255)
                cv2.circle(display_img, (best_res[0], best_res[1]), best_res[2], color, 2)
                cv2.circle(display_img, (best_res[0], best_res[1]), 2, (0, 0, 255), -1)

        # Symmetry Correlation if only 1 pupil found
        if len(face_pupils) == 1:
            p1 = face_pupils[0]
            cx = fx + fw//2
            nx, ny = cx - (p1['x'] - cx), p1['y']
            face_pupils.append({'x': int(nx), 'y': int(ny), 'r': p1['r']})
            cv2.circle(display_img, (int(nx), int(ny)), p1['r'], (0, 255, 128), 2) # Mirror color
            cv2.putText(display_img, "Sym", (int(nx)-15, int(ny)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)
            
        pupils = face_pupils
    else:
        # NIR Fallback (Global)
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
        cv2.putText(display_img, f"{dist:.1f}px", (pupils[0]['x'], pupils[0]['y']-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    data_list.append({'filename': os.path.basename(img_path), 'pupils': len(pupils), 'dist': dist})
    cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), display_img)

df = pd.DataFrame(data_list)
df.to_csv(os.path.join(CASCADE_DIR, 'final_v8_analysis.csv'), index=False)
print(df.to_string())
