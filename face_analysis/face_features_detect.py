import cv2
import mediapipe as mp
import os
import numpy as np

def detect_face_features(image_path, output_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5 # 提高信心值
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    image = cv2.imread(image_path)
    if image is None:
        return False

    # 由於 Test_Database 是彩色影像，直接轉換為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 繪製完整的面部網格與輪廓
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        cv2.imwrite(output_path, image)
        return True
    return False

if __name__ == "__main__":
    # 測試新的資料庫
    test_dir = "Test_Database"
    output_dir = "results"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(test_dir, filename)
            output_path = os.path.join(output_dir, f"landmarks_{filename}")
            print(f"Testing on {img_path}...")
            if detect_face_features(img_path, output_path):
                print(f"Success! Result saved to {output_path}")
            else:
                print(f"Failed to detect face in {img_path}")
                
    # 將其中一張成功的結果設為 README 的展示圖
    if os.path.exists("results/landmarks_face1.jpg"):
        os.system("cp results/landmarks_face1.jpg results/face_landmarks_result.jpg")
