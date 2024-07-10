from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

# Mediapipe 손 인식 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 추적할 궤적을 저장할 리스트 초기화
trajectory = []
tracking = True
start_time = None

# 화면 중앙에 반지름이 2cm인 원을 그리기 위한 설정
circle_radius_cm = 2  # 반지름 2cm
dpi = 96  # 화면의 DPI (디스플레이 해상도에 따라 다름)
cm_to_px = dpi / 2.54  # 1cm를 픽셀로 변환
circle_radius_px = int(circle_radius_cm * cm_to_px)

def is_pointing(landmarks):
    tips_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, 
                mp_hands.HandLandmark.PINKY_TIP]
    pips_ids = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP, 
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, 
                mp_hands.HandLandmark.PINKY_PIP]
    
    if landmarks[tips_ids[1]].y >= landmarks[pips_ids[1]].y:
        return False
    
    for tip_id, pip_id in zip(tips_ids[2:], pips_ids[2:]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            return False
    return True

def fit_circle(pts):
    if len(pts) < 3:
        return (0, 0), 0, []

    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    A = np.column_stack([x, y, np.ones(len(pts))])
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    radius = np.sqrt(c[2] + cx**2 + cy**2)

    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (int(cx), int(cy)), int(radius), distances

def calculate_accuracy(distances, radius, target_radius, pts):
    radius_accuracy = 100 * (1 - abs(radius - target_radius) / target_radius)
    circularity = 100 * (1 - np.std(distances) / np.mean(distances))
    return (radius_accuracy + circularity) / 2

def is_circle_completed(start_point, current_point, threshold=20):
    return len(trajectory) > 100 and np.linalg.norm(np.array(start_point) - np.array(current_point)) < threshold

def generate_frames():
    global trajectory, tracking, start_time

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인해주세요.")
        exit()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
                break
            
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            circle_center = (frame_width // 2, frame_height // 2)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            result = hands.process(rgb_frame)
            
            if tracking:
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        if is_pointing(hand_landmarks.landmark):
                            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            current_point = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))
                            
                            if not trajectory:
                                start_time = time.time()
                            
                            trajectory.append(current_point)
                            
                            if time.time() - start_time > 2 and is_circle_completed(trajectory[0], current_point):
                                tracking = False
                        else:
                            trajectory = []
                            start_time = None
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    trajectory = []
                    start_time = None
            
            # 중앙에 원과 녹색 선 그리기
            cv2.circle(frame, circle_center, circle_radius_px, (0, 255, 255), 2)
            end_point = (circle_center[0] + circle_radius_px, circle_center[1])
            cv2.arrowedLine(frame, circle_center, end_point, (0, 255, 0), 3, tipLength=0.1)

            # 궤적 그리기
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 3)

            if not tracking and len(trajectory) > 2:
                center, radius, distances = fit_circle(trajectory)
                accuracy = calculate_accuracy(distances, radius, circle_radius_px, trajectory)
                cv2.circle(frame, center, 2, (0, 0, 255), 3)
                cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, 'Circle completed!', (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

    finally:
        cap.release()
        print("프로그램이 종료되었습니다.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
