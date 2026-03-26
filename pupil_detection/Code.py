#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import glob
import pandas as pd
import os
import math

# 確保目錄存在
if not os.path.exists('output'):
    os.makedirs('output')

data_list = []

# 讀取 Database 下的所有 .jpg 檔案
for bb, img_path in enumerate(glob.glob("Database/*.jpg")):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        continue

    # 1. 前處理 (Gaussian Blur + Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. 二值化 (Thresholding)
    ret, thresh = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 形態學處理 (Opening/Closing)
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    
    # 4. 填充孔洞 (Floodfill)
    im_floodfill = opening.copy()
    h, w = opening.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_out = opening | cv2.bitwise_not(im_floodfill)

    # 5. 邊緣檢測 (Canny)
    edges = cv2.Canny(im_out, 100, 200)

    # 6. 霍夫變換 (Hough Circle Transform)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=30, param2=25, minRadius=20, maxRadius=100)

    pupils = []
    if circles is not None:
        # 將座標轉為整數避免溢位
        circles = np.int32(np.around(circles))
        for (x, y, r) in circles[0, :]:
            # 過濾掉太小或太大的圓
            if r < 20 or r > 80: continue
            
            # 將 numpy 數值轉為原生 python int 避免後續計算溢位
            curr_x, curr_y, curr_r = int(x), int(y), int(r)
            pupils.append({'x': curr_x, 'y': curr_y, 'r': curr_r})
            
            # 畫出瞳孔 (綠色) 與 中心 (黃色)
            cv2.circle(img, (curr_x, curr_y), curr_r, (0, 255, 0), 3)
            cv2.circle(img, (curr_x, curr_y), 3, (0, 255, 255), -1)
            cv2.putText(img, f"({curr_x},{curr_y})", (curr_x - 40, curr_y + curr_r + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 7. 計算兩瞳孔中心距離 (重點需求)
    dist = None
    if len(pupils) >= 2:
        # 取前兩個偵測到的圓
        p1, p2 = pupils[0], pupils[1]
        
        # 計算歐幾里得距離
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        dist = math.sqrt(dx**2 + dy**2)
        
        # 畫出連接線 (藍色)
        cv2.line(img, (p1['x'], p1['y']), (p2['x'], p2['y']), (255, 0, 0), 3)
        
        # 在連線中央標註距離文字
        mid_x = (p1['x'] + p2['x']) // 2
        mid_y = (p1['y'] + p2['y']) // 2
        label = f"Distance: {dist:.2f}px"
        
        # 加上背景框讓文字更清楚
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (mid_x - 5, mid_y - th - 15), (mid_x + tw + 5, mid_y + 5), (255, 0, 0), -1)
        cv2.putText(img, label, (mid_x, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 儲存 CSV 數據
    data_list.append({
        'filename': os.path.basename(img_path),
        'pupil_L_center': f"({pupils[0]['x']},{pupils[0]['y']})" if len(pupils) > 0 else None,
        'pupil_R_center': f"({pupils[1]['x']},{pupils[1]['y']})" if len(pupils) > 1 else None,
        'distance_px': dist
    })
    
    # 儲存圖片
    output_name = os.path.basename(img_path).replace(".jpg", "_result.jpg")
    cv2.imwrite(f'output/{output_name}', img)

# 儲存報表
df = pd.DataFrame(data_list)
df.to_csv('final_pupil_analysis.csv', index=False)
print("Updated analysis complete. Results with text labels saved.")
