import cv2
import numpy as np
import os

def generate_algorithm_steps(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None: return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Gaussian Blur (平滑化)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0) # 加強模糊效果以供視覺展示
    cv2.imwrite(os.path.join(output_dir, "step0_blurred.jpg"), blurred)
    
    # 2. Binarization (基於直方圖的二值化)
    # 模擬 Code.py 中的門檻值分割
    ret, thresh = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(output_dir, "step1_binarization.jpg"), thresh)
    
    # 3. Canny Edges
    edges = cv2.Canny(thresh, 100, 200)
    cv2.imwrite(os.path.join(output_dir, "step2_edges.jpg"), edges)
    
    # 4. Hough Circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=30, param2=25, minRadius=20, maxRadius=100)
    result_img = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result_img, (i[0], i[1]), i[2], (0, 0, 255), 3)
            cv2.circle(result_img, (i[0], i[1]), 2, (255, 0, 0), 3)
    cv2.imwrite(os.path.join(output_dir, "step3_hough_result.jpg"), result_img)

if __name__ == "__main__":
    test_img = "pupil_detection/Database/frame_NIR_000022.jpg"
    output_dir = "face_analysis/results"
    generate_algorithm_steps(test_img, output_dir)
    print("All algorithm step images (including Blur) generated.")
