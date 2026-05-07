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

OUTPUT_DIR = os.path.join(BASE_DIR, 'output-png', 'output_v6')
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

def detect_pupil_in_roi(roi_gray, ew, eh):
    roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    # Ellipse Fitting
    _, thresh = cv2.threshold(roi_blurred, 45, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if 30 < area < (ew * eh * 0.4):
                ellipse = cv2.fitEllipse(cnt)
                ratio = min(ellipse[1]) / max(ellipse[1]) if max(ellipse[1]) > 0 else 0
                if ratio > 0.6:
                    return (int(ellipse[0][0]), int(ellipse[0][1]), int(max(ellipse[1])/2))
    # Hough Fallback
    circles = cv2.HoughCircles(roi_blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=15, minRadius=int(ew*0.05), maxRadius=int(ew*0.3))
    if circles is not None:
        c = circles[0, 0]
        return (int(c[0]), int(c[1]), int(c[2]))
    return None

data_list = []
for ext in ['*.jpg', '*.png']:
    for img_path in glob.glob(os.path.join(BASE_DIR, 'Database', ext)):
        img = cv2.imread(img_path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_en = clahe.apply(gray)
        display_img = img.copy()
        
        # 1. Standard Cascade Detection
        faces = face_cascade.detectMultiScale(gray_en, 1.1, 4)
        pupils = []
        
        if len(faces) > 0:
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
                eyes = eye_cascade.detectMultiScale(gray_en[fy:fy+fh, fx:fx+fw], 1.1, 10)
                face_pupils = []
                for (ex, ey, ew, eh) in eyes:
                    res = detect_pupil_in_roi(gray_en[fy+ey:fy+ey+eh, fx+ex:fx+ex+ew], ew, eh)
                    if res:
                        gx, gy = fx+ex+res[0], fy+ey+res[1]
                        face_pupils.append({'x': gx, 'y': gy, 'r': res[2]})
                        cv2.circle(display_img, (gx, gy), res[2], (0, 255, 0), 2)
                
                # Correlation fallback
                if len(face_pupils) == 1:
                    p1 = face_pupils[0]
                    cx = fx + fw//2
                    nx, ny = cx - (p1['x'] - cx), p1['y']
                    face_pupils.append({'x': int(nx), 'y': int(ny), 'r': p1['r']})
                    cv2.drawMarker(display_img, (int(nx), int(ny)), (255, 100, 0), cv2.MARKER_CROSS, 20, 2)
                pupils.extend(face_pupils)
        else:
            # 2. NIR Fallback: Global search for symmetric dark regions
            _, thresh = cv2.threshold(gray_en, 40, 255, cv2.THRESH_BINARY_INV)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for cnt in cnts:
                if 500 < cv2.contourArea(cnt) < 10000:
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                        candidates.append((cx, cy, cv2.contourArea(cnt)))
            
            # Sort by y, find horizontal pairs
            candidates.sort(key=lambda c: c[1])
            for i in range(len(candidates)-1):
                c1, c2 = candidates[i], candidates[i+1]
                if abs(c1[1] - c2[1]) < 50 and 500 < abs(c1[0] - c2[0]) < 1500:
                    pupils = [{'x': c1[0], 'y': c1[1], 'r': 40}, {'x': c2[0], 'y': c2[1], 'r': 40}]
                    for p in pupils: cv2.circle(display_img, (p['x'], p['y']), p['r'], (0, 0, 255), 3)
                    break

        dist = None
        if len(pupils) >= 2:
            pupils.sort(key=lambda p: p['x'])
            dist = math.sqrt((pupils[0]['x']-pupils[1]['x'])**2 + (pupils[0]['y']-pupils[1]['y'])**2)
            cv2.line(display_img, (pupils[0]['x'], pupils[0]['y']), (pupils[1]['x'], pupils[1]['y']), (255, 0, 0), 2)

        data_list.append({'filename': os.path.basename(img_path), 'pupils': len(pupils), 'dist': dist})
        cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), display_img)

df = pd.DataFrame(data_list)
df.to_csv(os.path.join(CASCADE_DIR, 'final_v6_analysis.csv'), index=False)
print(df.to_string())
