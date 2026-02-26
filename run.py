"""
Qwen2.5-VL-7B-Instruct RealSense D455 실시간 이미지 분석 프로그램
- RealSense D455 카메라로부터 RGB 영상을 실시간으로 표시
- 'c' 키로 현재 프레임 캡처 후 VLM 모델 분석
- 분석 중에도 실시간 영상 유지
- 분석 완료 후 결과창 표시 및 'c' 키 재활성화
- 사람 얼굴 감지 시 결과 이미지에 빨간색 바운딩 박스 표시
"""

import re
import sys
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

MIN_PIXELS = 256 * 28 * 28     # 최소 256 토큰
MAX_PIXELS = 1024 * 28 * 28    # 최대 1024 토큰 (~896×896)

DEFAULT_PROMPT     = "이 이미지를 자세히 설명해주세요."
FACE_DETECT_PROMPT = "Detect all human faces."

FONT_PATH      = "C:/Windows/Fonts/malgun.ttf"    # 맑은 고딕
FONT_BOLD_PATH = "C:/Windows/Fonts/malgunbd.ttf"  # 맑은 고딕 Bold

LIVE_WIN   = "RealSense D455 - Live"
RESULT_WIN = "Analysis Result"
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    """4-bit NF4 양자화로 모델과 프로세서를 로드한다."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"모델 로딩 중: {MODEL_ID}")
    print("양자화: 4-bit NF4, 연산 dtype: float16")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("모델 로딩 완료!")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv  = torch.cuda.memory_reserved()  / 1024**3
        print(f"VRAM - 할당: {alloc:.2f}GB, 예약: {resv:.2f}GB")

    return model, processor


def _infer(model, processor, pil_image: Image.Image,
           prompt: str, max_new_tokens: int,
           skip_special_tokens: bool) -> str:
    """공통 추론 헬퍼 (설명 생성 / 그라운딩 공용)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text":  prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]


def run_inference(model, processor, pil_image: Image.Image,
                  prompt: str = DEFAULT_PROMPT) -> str:
    """이미지 설명 텍스트를 생성한다."""
    return _infer(model, processor, pil_image,
                  prompt=prompt, max_new_tokens=1024, skip_special_tokens=True)


def detect_face_boxes(model, processor, pil_image: Image.Image) -> list:
    """
    얼굴 그라운딩을 실행하고 바운딩 박스를 반환한다.

    Qwen2.5-VL 그라운딩 출력 형식:
      <|object_ref_start|>face<|object_ref_end|>
      <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    좌표: [0, 1000] 정규화 범위

    반환값: [(x1, y1, x2, y2), ...] — 빈 리스트이면 얼굴 없음
    """
    raw = _infer(model, processor, pil_image,
                 prompt=FACE_DETECT_PROMPT, max_new_tokens=256,
                 skip_special_tokens=False)
    print(f"[얼굴 감지 원문] {raw!r}")

    boxes = []
    # <|box_start|>(x1,y1),(x2,y2)<|box_end|> 형식에서 좌표 추출
    for section in re.findall(r'<\|box_start\|>(.*?)<\|box_end\|>', raw):
        for m in re.finditer(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)', section):
            boxes.append(tuple(int(m.group(i)) for i in range(1, 5)))

    return boxes


def draw_face_boxes(frame_bgr: np.ndarray, boxes_1000: list) -> np.ndarray:
    """
    [0, 1000] 정규화 좌표의 얼굴 박스를 원본 프레임 위에 빨간색으로 그린다.
    """
    if not boxes_1000:
        return frame_bgr

    annotated = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]

    for idx, (x1, y1, x2, y2) in enumerate(boxes_1000):
        px1 = int(x1 / 1000 * w)
        py1 = int(y1 / 1000 * h)
        px2 = int(x2 / 1000 * w)
        py2 = int(y2 / 1000 * h)

        # 빨간색 박스 (두께 2)
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)

        # 박스 위에 "Face N" 레이블
        label = f"Face {idx + 1}"
        label_y = max(py1 - 8, 16)
        cv2.putText(annotated, label, (px1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),   3)
        cv2.putText(annotated, label, (px1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return annotated


def _wrap_text_pil(text: str, font, max_width: int) -> list:
    """PIL 폰트 실제 렌더링 너비 기준으로 텍스트를 줄바꿈한다."""
    lines = []
    for para in text.split('\n'):
        if not para:
            lines.append('')
            continue
        current = ''
        for char in para:
            test = current + char
            if font.getlength(test) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def build_result_image(frame_bgr: np.ndarray, result_text: str,
                       face_count: int = 0,
                       window_width: int = 960) -> np.ndarray:
    """캡처 이미지(얼굴 박스 포함)와 분석 결과 텍스트를 합성한다."""
    padding         = 20
    line_height     = 24
    text_font_size  = 16
    title_font_size = 18

    try:
        font       = ImageFont.truetype(FONT_PATH,      text_font_size)
        title_font = ImageFont.truetype(FONT_BOLD_PATH, title_font_size)
    except OSError:
        font       = ImageFont.load_default()
        title_font = font

    img_w = window_width - 2 * padding
    h, w  = frame_bgr.shape[:2]
    img_h = int(h * img_w / w)

    resized_rgb = cv2.cvtColor(
        cv2.resize(frame_bgr, (img_w, img_h)), cv2.COLOR_BGR2RGB
    )

    lines        = _wrap_text_pil(result_text, font, img_w)
    text_area_h  = len(lines) * line_height + 2 * padding + 56
    total_height = padding + img_h + padding + text_area_h

    canvas = Image.new('RGB', (window_width, total_height), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    # 캡처 이미지 (얼굴 박스가 그려진 상태)
    canvas.paste(Image.fromarray(resized_rgb), (padding, padding))

    # 구분선
    sep_y = padding + img_h + padding // 2
    draw.line([(padding, sep_y), (window_width - padding, sep_y)],
              fill=(80, 80, 80), width=1)

    # 제목 + 얼굴 감지 수
    title_y = padding + img_h + padding
    draw.text((padding, title_y), "분석 결과", font=title_font, fill=(100, 200, 100))

    if face_count > 0:
        face_label = f"  |  얼굴 감지: {face_count}명"
        face_color = (100, 160, 255)
    else:
        face_label = "  |  얼굴 감지: 없음"
        face_color = (130, 130, 130)
    draw.text((padding + 110, title_y + 1), face_label, font=font, fill=face_color)

    # 분석 결과 텍스트
    for i, line in enumerate(lines):
        y = title_y + 32 + i * line_height
        draw.text((padding, y), line, font=font, fill=(220, 220, 220))

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def main():
    # ── GPU 확인 ──────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("오류: CUDA GPU를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"모델 로딩 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # ── RealSense 파이프라인 초기화 ───────────────────────────────────────────
    pipeline  = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    try:
        pipeline.start(rs_config)
        print("RealSense D455 시작됨 (1280×720 @ 30fps)")
    except Exception as e:
        print(f"카메라 시작 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # ── 공유 상태 ─────────────────────────────────────────────────────────────
    lock            = threading.Lock()
    capture_enabled = True
    is_analyzing    = False
    pending_result  = None  # worker → main: (annotated_frame, text, face_count)
    result_image    = None  # 현재 결과창에 표시 중인 이미지

    # ── 분석 워커 (별도 스레드) ───────────────────────────────────────────────
    def analysis_worker(frame_bgr: np.ndarray):
        nonlocal capture_enabled, is_analyzing, pending_result

        pil_image  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        annotated  = frame_bgr
        face_count = 0

        try:
            # 1단계: 이미지 설명 생성
            description = run_inference(model, processor, pil_image)

            # 2단계: 얼굴 그라운딩 감지
            face_boxes = detect_face_boxes(model, processor, pil_image)
            face_count = len(face_boxes)

            if face_boxes:
                # 원본 프레임에 빨간 박스 그리기
                annotated = draw_face_boxes(frame_bgr, face_boxes)
                print(f"[얼굴 감지] {face_count}명 감지 — 바운딩 박스 적용")
            else:
                print("[얼굴 감지] 얼굴 없음")

        except torch.cuda.OutOfMemoryError:
            description = "오류: GPU 메모리가 부족합니다."
        except Exception as e:
            description = f"분석 오류: {e}"

        print(f"\n[분석 결과]\n{description}\n")

        with lock:
            pending_result = (annotated, description, face_count)

        is_analyzing    = False
        capture_enabled = True

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    print("\n조작 방법")
    print("  c  - 현재 프레임 캡처 및 분석 (얼굴 감지 포함)")
    print("  q  - 종료\n")

    cv2.namedWindow(LIVE_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_WIN, 960, 540)

    try:
        while True:
            # 프레임 수신
            frames      = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame   = np.asanyarray(color_frame.get_data())
            display = frame.copy()

            # 상태 오버레이
            if is_analyzing:
                label = "Analyzing...  please wait"
                fg    = (0, 165, 255)   # 주황
            else:
                label = "Press 'c' to capture  |  'q' to quit"
                fg    = (0, 220, 0)     # 초록

            cv2.putText(display, label, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(display, label, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, fg, 2)

            cv2.imshow(LIVE_WIN, display)

            # 분석 결과 수신 → 결과창 갱신 (OpenCV 창은 메인 스레드에서만 조작)
            with lock:
                if pending_result is not None:
                    annotated_frame, desc, n_faces = pending_result
                    result_image   = build_result_image(
                        annotated_frame, desc, face_count=n_faces
                    )
                    pending_result = None

            if result_image is not None:
                cv2.namedWindow(RESULT_WIN, cv2.WINDOW_NORMAL)
                cv2.imshow(RESULT_WIN, result_image)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c') and capture_enabled and not is_analyzing:
                capture_enabled = False
                is_analyzing    = True
                result_image    = None

                try:
                    cv2.destroyWindow(RESULT_WIN)
                except cv2.error:
                    pass

                print("캡처됨 - 분석 시작...")
                threading.Thread(
                    target=analysis_worker,
                    args=(frame.copy(),),
                    daemon=True,
                ).start()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("종료.")


if __name__ == "__main__":
    main()
