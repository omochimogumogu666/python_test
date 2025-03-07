import cv2
import mediapipe as mp
import numpy as np

# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 描画スタイルをカスタマイズ
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))
connection_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(255, 255, 0))

def calculate_angle(a, b, c):
    """3点間の角度を計算"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def calculate_shoulder_tilt(left_shoulder, right_shoulder):
    """shoulderの傾きを計算"""
    angle = np.arctan2(right_shoulder[1] - left_shoulder[1],
                      right_shoulder[0] - left_shoulder[0]) * 180.0 / np.pi
    return angle

def main():
    # Webカメラの初期化
    cap = cv2.VideoCapture(0)
    
    # フォント設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # モーションキャプチャの精度を高めるための設定
    with mp_pose.Pose(
        model_complexity=2,  # 最高精度モデルを使用
        enable_segmentation=True,  # セグメンテーションを有効化
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("カメラからの読み取りに失敗しました。")
                continue

            # 画像の処理を効率化するために画像を反転
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            # 描画用に画像を準備
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 各ランドマークの取得
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # 角度の計算
                # 腕の角度
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # 脚の角度
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                # 背骨/首の角度
                neck_angle = calculate_angle(nose, 
                                           [(left_shoulder[0] + right_shoulder[0])/2,
                                            (left_shoulder[1] + right_shoulder[1])/2],
                                           [(left_hip[0] + right_hip[0])/2,
                                            (left_hip[1] + right_hip[1])/2])
                
                # 肩の傾き
                shoulder_tilt = calculate_shoulder_tilt(left_shoulder, right_shoulder)
                
                # 胸の開き
                chest_angle = calculate_angle(left_shoulder, 
                                            [(left_shoulder[0] + right_shoulder[0])/2,
                                             (left_shoulder[1] + right_shoulder[1])/2],
                                            right_shoulder)
                
                # ポーズの検出結果を描画 - より詳細なスタイルで
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=connection_drawing_spec)
                
                # セグメンテーションマスクを表示する場合
                if results.segmentation_mask is not None:
                    segm_mask = results.segmentation_mask
                    # 背景を暗くして人物を強調
                    condition = np.stack((segm_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(image.shape, dtype=np.uint8)
                    bg_image[:] = (0, 0, 0)
                    image = np.where(condition, image, bg_image)
                
                # 角度と座標を画像に表示
                h, w, c = image.shape
                
                # 腕の角度
                cv2.putText(image, f'R Elbow: {int(right_elbow_angle)}deg',
                          (int(right_elbow[0]*w), int(right_elbow[1]*h)),
                          font, 0.5, (255,255,255), 2)
                cv2.putText(image, f'L Elbow: {int(left_elbow_angle)}deg',
                          (int(left_elbow[0]*w), int(left_elbow[1]*h)),
                          font, 0.5, (255,255,255), 2)
                
                # 腕の回旋角度を計算
                right_arm_rotation = calculate_angle(
                    [right_shoulder[0], right_shoulder[1] - 0.1], 
                    right_shoulder, 
                    right_elbow)
                left_arm_rotation = calculate_angle(
                    [left_shoulder[0], left_shoulder[1] - 0.1], 
                    left_shoulder, 
                    left_elbow)
                
                # 腕の回旋角度を表示
                cv2.putText(image, f'R Arm Rot: {int(right_arm_rotation)}deg',
                          (int(right_shoulder[0]*w), int((right_shoulder[1]-0.05)*h)),
                          font, 0.5, (255,255,255), 2)
                cv2.putText(image, f'L Arm Rot: {int(left_arm_rotation)}deg',
                          (int(left_shoulder[0]*w), int((left_shoulder[1]-0.05)*h)),
                          font, 0.5, (255,255,255), 2)
                
                # 脚の角度
                cv2.putText(image, f'R Knee: {int(right_knee_angle)}deg',
                          (int(right_knee[0]*w), int(right_knee[1]*h)),
                          font, 0.5, (255,255,255), 2)
                cv2.putText(image, f'L Knee: {int(left_knee_angle)}deg',
                          (int(left_knee[0]*w), int(left_knee[1]*h)),
                          font, 0.5, (255,255,255), 2)
                
                # 脚の回旋角度を計算
                right_leg_rotation = calculate_angle(
                    [right_hip[0], right_hip[1] - 0.1], 
                    right_hip, 
                    right_knee)
                left_leg_rotation = calculate_angle(
                    [left_hip[0], left_hip[1] - 0.1], 
                    left_hip, 
                    left_knee)
                
                # 脚の回旋角度を表示
                cv2.putText(image, f'R Leg Rot: {int(right_leg_rotation)}deg',
                          (int(right_hip[0]*w), int((right_hip[1]-0.05)*h)),
                          font, 0.5, (255,255,255), 2)
                cv2.putText(image, f'L Leg Rot: {int(left_leg_rotation)}deg',
                          (int(left_hip[0]*w), int((left_hip[1]-0.05)*h)),
                          font, 0.5, (255,255,255), 2)
                
                # 首と背骨の角度
                cv2.putText(image, f'Neck: {int(neck_angle)}deg',
                          (10, 30),
                          font, 0.5, (255,255,255), 2)
                
                # 肩の傾き
                cv2.putText(image, f'Shoulder Tilt: {int(shoulder_tilt)}deg',
                          (10, 60),
                          font, 0.5, (255,255,255), 2)
                
                # 胸の開き
                cv2.putText(image, f'Chest: {int(chest_angle)}deg',
                          (10, 90),
                          font, 0.5, (255,255,255), 2)
                
                # 腰の角度
                hip_angle = calculate_angle(
                    [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
                    [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
                    [(left_knee[0] + right_knee[0])/2, (left_knee[1] + right_knee[1])/2])
                cv2.putText(image, f'Hip: {int(hip_angle)}deg',
                          (10, 120),
                          font, 0.5, (255,255,255), 2)
                
                # バランススコアの計算
                balance_score = 100 - min(abs(shoulder_tilt), 45) * (100/45)
                cv2.putText(image, f'Balance: {int(balance_score)}%',
                          (10, 150),
                          font, 0.5, (255,255,255), 2)
                
                # 足首の角度
                right_foot_angle = calculate_angle(
                    right_knee, 
                    right_ankle, 
                    [right_ankle[0] + 0.1, right_ankle[1]])
                left_foot_angle = calculate_angle(
                    left_knee, 
                    left_ankle, 
                    [left_ankle[0] + 0.1, left_ankle[1]])
                
                cv2.putText(image, f'R Foot: {int(right_foot_angle)}deg',
                          (int(right_ankle[0]*w), int((right_ankle[1]+0.05)*h)),
                          font, 0.5, (255,255,255), 2)
                cv2.putText(image, f'L Foot: {int(left_foot_angle)}deg',
                          (int(left_ankle[0]*w), int((left_ankle[1]+0.05)*h)),
                          font, 0.5, (255,255,255), 2)
            
            # 結果を表示
            # フレームレートを表示
            fps_text = f'FPS: {int(cap.get(cv2.CAP_PROP_FPS))}'
            cv2.putText(image, fps_text, (w-150, 30), font, 0.5, (255,255,255), 2)
            
            # 画像の解像度を表示
            resolution_text = f'Resolution: {w}x{h}'
            cv2.putText(image, resolution_text, (w-250, 60), font, 0.5, (255,255,255), 2)
            
            cv2.imshow('High Precision Motion Capture', image)
            
            # 'q'キーで終了
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()