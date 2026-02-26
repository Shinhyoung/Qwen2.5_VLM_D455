# Qwen2.5-VL-7B × RealSense D455 실시간 이미지 분석

RealSense D455 카메라로 RGB 영상을 실시간 스트리밍하며, `c` 키 하나로 프레임을 캡처해 **Qwen2.5-VL-7B-Instruct** Vision-Language 모델로 분석하는 프로그램입니다.
감지 대상 객체의 위치를 **빨간색 바운딩 박스**로 결과 이미지에 표시합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 실시간 RGB 스트리밍 | RealSense D455 · 1280×720 · 30fps |
| 프레임 캡처 & 분석 | `c` 키로 즉시 캡처 후 VLM 추론 |
| 비차단 추론 | 분석 중에도 실시간 영상 유지 (별도 스레드) |
| 객체 감지 & 시각화 | 감지 대상을 빨간 bbox로 표시 + 한글 분석 결과를 별도 창으로 표시 |
| 감지 대상 변경 | 두 줄 수정만으로 감지 대상 교체 가능 |
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
  백그라운드 스레드                                │
  ├─ 1. 이미지 설명 생성 (VLM)  ←────────────────┘
  └─ 2. 객체 그라운딩 감지 (VLM)
       │
       ▼
  결과창 표시 (캡처 이미지 + 빨간 bbox + 분석 텍스트)
  + c 키 재활성화
```

---

## 감지 대상 변경 방법

`run.py` 상단의 두 줄만 수정하면 됩니다.

```python
# ── 감지 대상 (변경 시 이 두 줄만 수정) ──────────────────────────────────────
DETECTION_TARGET = "keyboard"   # 영문 단수형 — 모델 프롬프트에 사용
DETECTION_LABEL  = "키보드"      # 결과창 한글 표시
```

**변경 예시:**

| 감지 대상 | DETECTION_TARGET | DETECTION_LABEL |
|-----------|-----------------|-----------------|
| 키보드 (기본) | `"keyboard"` | `"키보드"` |
| 사람 | `"person"` | `"사람"` |
| 자동차 | `"car"` | `"자동차"` |
| 마우스 | `"mouse"` | `"마우스"` |

---

## 모델 정보

| 항목 | 값 |
|------|-----|
| 모델 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 양자화 | 4-bit NF4 (BitsAndBytes) |
| 연산 dtype | float16 |
| 이미지 토큰 범위 | 256 ~ 1024 토큰 |
| VRAM 사용량 (RTX 5070) | 할당 ~5.5GB / 예약 ~6.0GB |

---

## 코드 구조

```
run.py
├── [상수 블록]             MODEL_ID, 카메라, 감지 대상, UI 파라미터
├── load_model()            모델·프로세서 로드 (4-bit NF4 양자화)
├── _infer()                공통 추론 헬퍼
├── run_inference()         이미지 설명 생성 래퍼
├── detect_objects()        객체 그라운딩 감지 (bbox 반환)
├── draw_object_boxes()     감지 bbox 오버레이 (빨간색)
├── _draw_outlined_text()   외곽선 텍스트 헬퍼
├── _wrap_text_pil()        PIL 한글 텍스트 줄바꿈
├── build_result_image()    결과 이미지 합성
├── init_camera()           RealSense D455 초기화
├── make_analysis_worker()  백그라운드 워커 클로저 생성
├── run_loop()              메인 루프 (영상 표시 + 키 입력)
└── main()                  진입점
```

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
