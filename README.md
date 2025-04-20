# SmartICT 프로젝트

## 프로젝트 소개
이 프로젝트는 실시간 이미지 처리와 객체 인식을 위한 AI 기반 시스템입니다. YOLOv8과 토치비전(TorchVision)을 활용하여 이미지 분석 및 객체 탐지를 수행합니다.

## 주요 기능
- 실시간 이미지 처리 및 객체 탐지
- 소켓 통신을 통한 클라이언트-서버 구조
- YOLOv8 모델을 활용한 객체 인식
- 이미지 전처리 및 엣지 검출
- 카테고리 분류 및 라벨링

## 시스템 구성
### 서버 (`server.py`)
- 소켓 통신을 통한 클라이언트 연결 관리
- 실시간 이미지 데이터 수신 및 처리
- AI 모델을 통한 이미지 분석
- 분석 결과를 클라이언트에게 전송

### 예측 모듈
- `predict.py`: TorchVision 모델을 사용한 이미지 예측
- `predict_yolo.py`: YOLOv8 모델을 사용한 객체 탐지
- `functions.py`: 유틸리티 함수 모음

### 학습 모듈
- `training.py`: 기본 모델 학습
- `training_version2.py`: 개선된 버전의 모델 학습
- `yolo_train.py`: YOLOv8 모델 학습

## 설치 및 실행 방법
1. 필요한 패키지 설치:
```bash
pip install opencv-python
pip install torch torchvision
pip install ultralytics
```

2. 서버 실행:
```bash
python server.py
```

3. 클라이언트 연결 후 이미지 전송

## 디렉토리 구조
```
smartICT/
├── model/              # 학습된 모델 저장
├── test_img/          # 테스트 이미지
├── runs/              # 학습 결과
├── server.py          # 메인 서버
├── predict.py         # TorchVision 예측
├── predict_yolo.py    # YOLO 예측
├── functions.py       # 유틸리티 함수
├── edge.py           # 엣지 검출
└── training*.py      # 모델 학습
```

## 주요 파일 설명
- `server.py`: 소켓 통신을 통한 이미지 수신 및 처리
- `predict_yolo.py`: YOLOv8 모델을 사용한 객체 탐지
- `functions.py`: 이미지 처리 및 유틸리티 함수
- `edge.py`: 이미지 엣지 검출
- `training.py`: 모델 학습 스크립트

## 시스템 요구사항
- Python 3.8 이상
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- CUDA 지원 GPU (권장)

## 라이센스
이 프로젝트는 MIT 라이센스를 따릅니다.