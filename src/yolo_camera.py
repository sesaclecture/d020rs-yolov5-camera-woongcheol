import torch
import cv2

# Model
# 모델 로드 경로는 사용자 환경에 맞게 확인해주세요.
model = torch.hub.load('/home/intel/Desktop/project/0922/yolov5/', 'yolov5x6', source='local')

# 웹캠 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# 실시간 처리를 위한 반복문
while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("오류: 프레임을 읽을 수 없습니다.")
        break
    
    # 1. 모델 추론 (YOLOv5는 BGR 입력을 자동으로 처리하므로 색상 변환이 불필요합니다)
    results = model(frame)

    # 2. 판다스 데이터프레임으로 결과 추출 (가독성 및 사용성 향상)
    detections = results.pandas().xyxy[0]

    # 3. 탐지된 각 객체에 대해 바운딩 박스와 라벨 그리기
    for index, row in detections.iterrows():
        # 필요한 데이터 추출
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        confidence = row['confidence']
        name = row['name'] # 클래스 이름

        # 라벨 텍스트 생성 (예: person: 0.88)
        label = f"{name} {confidence:.2f}"
        
        # 원본 frame(BGR)에 사각형과 텍스트 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print(f"Object {index}: {name} (Confidence: {confidence:.2f})")

    # 4. 화면 표시 (BGR 프레임 사용)
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()