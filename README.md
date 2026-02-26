# Qwen2.5-VL-7B × RealSense D455 실시간 이미지 분석

RealSense D455 카메라로 RGB 영상을 실시간 스트리밍하며, `c` 키 하나로 프레임을 캡처해 **Qwen2.5-VL-7B-Instruct** Vision-Language 모델로 분석하는 프로그램입니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 실시간 RGB 스트리밍 | RealSense D455 · 1280×720 · 30fps |
| 프레임 캡처 & 분석 | `c` 키로 즉시 캡처 후 VLM 추론 |
| 비차단 추론 | 분석 중에도 실시간 영상 유지 (별도 스레드) |
| 결과 시각화 | 캡처 이미지 + 한글 분석 결과를 별도 창으로 표시 |
| 메모리 효율 | 4-bit NF4 양자화 → 12GB VRAM GPU에서 실행 가능 |

---

## 요구 사항

### 하드웨어
- NVIDIA GPU (VRAM 12GB 이상 권장, RTX 4070 / 4080 / 5070 등)
- Intel RealSense D455 카메라

### 소프트웨어
- Python 3.10 이상
- CUDA 11.8 이상
- Anaconda / Miniconda

---

## 설치

### 1. conda 가상환경 생성

```bash
conda create -n vlm python=3.12 -y
conda activate vlm
```

### 2. PyTorch 설치 (CUDA 12.1 기준)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 실행

```bash
conda activate vlm
python run.py
```

---

## 조작 방법

| 키 | 동작 |
|----|------|
| `c` | 현재 프레임 캡처 및 VLM 분석 시작 |
| `q` | 프로그램 종료 |

### 동작 흐름

```
카메라 스트림 시작
       │
       ▼
  실시간 RGB 영상 표시 ──────────────────────────┐
       │                                          │ (분석 중에도 계속)
   [c 키 입력]                                    │
       │                                          │
       ▼                                          │
  프레임 캡처 (c 키 비활성화)                      │
       │                                          │
       ▼                                          │
  백그라운드 스레드에서 VLM 추론 ←─────────────────┘
       │
       ▼
  결과창 표시 + c 키 재활성화
```

---

## 모델 정보

| 항목 | 값 |
|------|-----|
| 모델 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 양자화 | 4-bit NF4 (BitsAndBytes) |
| 연산 dtype | float16 |
| 이미지 토큰 범위 | 256 ~ 1024 토큰 |
| VRAM 사용량 (RTX 5070) | 할당 ~5.5GB / 예약 ~5.9GB |

---

## 프로젝트 구조

```
VLM/
├── run.py           # 메인 프로그램
├── requirements.txt # Python 의존성
└── README.md
```

---

## 의존성

```
transformers>=4.45.0
accelerate>=0.34.0
bitsandbytes>=0.44.0
qwen-vl-utils>=0.0.8
Pillow>=10.0.0
requests>=2.31.0
opencv-python>=4.8.0
pyrealsense2>=2.55.0
```
