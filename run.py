"""
Qwen2.5-VL-7B-Instruct RealSense D455 실시간 이미지 분석 프로그램
- RealSense D455 카메라로부터 RGB 영상을 실시간으로 표시
- 'c' 키로 현재 프레임 캡처 후 VLM 모델 분석
- 분석 중에도 실시간 영상 유지
- 분석 완료 후 결과창 표시 및 'c' 키 재활성화
- 감지 대상(DETECTION_TARGET) 객체에 빨간색 바운딩 박스 표시
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

# ── 모델 ──────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"
MIN_PIXELS = 256  * 28 * 28    # 최소 256 토큰
MAX_PIXELS = 1024 * 28 * 28    # 최대 1024 토큰 (~896×896)

# ── 카메라 ────────────────────────────────────────────────────────────────────
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# ── 감지 대상 (변경 시 이 두 줄만 수정) ──────────────────────────────────────
DETECTION_TARGET = "keyboard"   # 영문 단수형 — 프롬프트에 사용
DETECTION_LABEL  = "키보드"      # 결과창 한글 표시

DETECTION_PROMPT = f"Detect all {DETECTION_TARGET}s."
DEFAULT_PROMPT   = "이 이미지를 자세히 설명해주세요."

# ── 폰트 (Windows 맑은 고딕) ─────────────────────────────────────────────────
FONT_PATH      = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/malgunbd.ttf"

# ── 결과 이미지 UI ────────────────────────────────────────────────────────────
RESULT_WIN_WIDTH   = 960
RESULT_PADDING     = 20
RESULT_LINE_HEIGHT = 24
RESULT_FONT_SIZE   = 16
RESULT_TITLE_SIZE  = 18

# ── 오버레이 / 박스 ───────────────────────────────────────────────────────────
BOX_COLOR         = (0, 0, 255)   # BGR 빨간색
BOX_THICKNESS     = 2
BOX_FONT_SCALE    = 0.65
STATUS_FONT_SCALE = 0.8

# ── 창 이름 ───────────────────────────────────────────────────────────────────
LIVE_WIN   = "RealSense D455 - Live"
RESULT_WIN = "Analysis Result"
# ──────────────────────────────────────────────────────────────────────────────


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

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


# ── 추론 ──────────────────────────────────────────────────────────────────────

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


def detect_objects(model, processor, pil_image: Image.Image) -> list:
    """
    DETECTION_TARGET 객체를 그라운딩하고 바운딩 박스를 반환한다.

    Qwen2.5-VL 출력 형식 두 가지를 모두 파싱한다.
      형식 A: <|box_start|>(x1,y1),(x2,y2)<|box_end|>  (특수 토큰)
      형식 B: {"bbox_2d": [x1, y1, x2, y2]}             (JSON)
    좌표 범위: [0, 1000] 정규화

    반환값: [(x1, y1, x2, y2), ...] — 빈 리스트이면 감지 없음
    """
    raw = _infer(model, processor, pil_image,
                 prompt=DETECTION_PROMPT, max_new_tokens=512,
                 skip_special_tokens=False)
    print(f"[{DETECTION_LABEL} 감지 원문] {raw!r}")

    boxes = []

    # 형식 A: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    for section in re.findall(r'<\|box_start\|>(.*?)<\|box_end\|>', raw):
        for m in re.finditer(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)', section):
            boxes.append(tuple(int(m.group(i)) for i in range(1, 5)))

    # 형식 B: JSON {"bbox_2d": [x1, y1, x2, y2]}
    if not boxes:
        for m in re.finditer(
            r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
            raw
        ):
            boxes.append(tuple(int(m.group(i)) for i in range(1, 5)))

    return boxes


# ── 그리기 헬퍼 ───────────────────────────────────────────────────────────────

def _draw_outlined_text(img: np.ndarray, text: str, pos: tuple,
                        font_scale: float, color: tuple) -> None:
    """검은 외곽선 + 컬러 본문 이중 렌더링으로 가독성 있는 텍스트를 그린다."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,    2)


def draw_object_boxes(frame_bgr: np.ndarray, boxes_1000: list) -> np.ndarray:
    """
    [0, 1000] 정규화 좌표의 객체 박스를 원본 프레임 위에 빨간색으로 그린다.
    레이블 형식: "{DETECTION_LABEL} N"
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

        cv2.rectangle(annotated, (px1, py1), (px2, py2), BOX_COLOR, BOX_THICKNESS)

        label   = f"{DETECTION_LABEL} {idx + 1}"
        label_y = max(py1 - 8, 16)
        _draw_outlined_text(annotated, label, (px1, label_y), BOX_FONT_SCALE, BOX_COLOR)

    return annotated


# ── 결과 이미지 합성 ─────────────────────────────────────────────────────────

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
                       obj_count: int = 0) -> np.ndarray:
    """객체 박스 오버레이가 적용된 캡처 이미지와 분석 결과 텍스트를 합성한다."""
    try:
        font       = ImageFont.truetype(FONT_PATH,      RESULT_FONT_SIZE)
        title_font = ImageFont.truetype(FONT_BOLD_PATH, RESULT_TITLE_SIZE)
    except OSError:
        font       = ImageFont.load_default()
        title_font = font

    img_w = RESULT_WIN_WIDTH - 2 * RESULT_PADDING
    h, w  = frame_bgr.shape[:2]
    img_h = int(h * img_w / w)

    resized_rgb = cv2.cvtColor(
        cv2.resize(frame_bgr, (img_w, img_h)), cv2.COLOR_BGR2RGB
    )

    lines        = _wrap_text_pil(result_text, font, img_w)
    text_area_h  = len(lines) * RESULT_LINE_HEIGHT + 2 * RESULT_PADDING + 56
    total_height = RESULT_PADDING + img_h + RESULT_PADDING + text_area_h

    canvas = Image.new('RGB', (RESULT_WIN_WIDTH, total_height), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    # 객체 박스 오버레이가 적용된 캡처 이미지
    canvas.paste(Image.fromarray(resized_rgb), (RESULT_PADDING, RESULT_PADDING))

    # 구분선
    sep_y = RESULT_PADDING + img_h + RESULT_PADDING // 2
    draw.line([(RESULT_PADDING, sep_y), (RESULT_WIN_WIDTH - RESULT_PADDING, sep_y)],
              fill=(80, 80, 80), width=1)

    # 제목 + 감지 수
    title_y = RESULT_PADDING + img_h + RESULT_PADDING
    draw.text((RESULT_PADDING, title_y), "분석 결과", font=title_font, fill=(100, 200, 100))

    if obj_count > 0:
        det_label = f"  |  {DETECTION_LABEL} 감지: {obj_count}개"
        det_color = (100, 160, 255)
    else:
        det_label = f"  |  {DETECTION_LABEL} 감지: 없음"
        det_color = (130, 130, 130)
    draw.text((RESULT_PADDING + 110, title_y + 1), det_label, font=font, fill=det_color)

    # 분석 결과 텍스트
    for i, line in enumerate(lines):
        y = title_y + 32 + i * RESULT_LINE_HEIGHT
        draw.text((RESULT_PADDING, y), line, font=font, fill=(220, 220, 220))

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


# ── 카메라 초기화 ─────────────────────────────────────────────────────────────

def init_camera() -> rs.pipeline:
    """RealSense D455 파이프라인을 생성하고 시작한다."""
    pipeline  = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.color,
        CAMERA_WIDTH, CAMERA_HEIGHT,
        rs.format.bgr8, CAMERA_FPS,
    )
    try:
        pipeline.start(rs_config)
        print(f"RealSense D455 시작됨 ({CAMERA_WIDTH}×{CAMERA_HEIGHT} @ {CAMERA_FPS}fps)")
        return pipeline
    except Exception as e:
        print(f"카메라 시작 실패: {e}", file=sys.stderr)
        sys.exit(1)


# ── 분석 워커 클로저 생성 ─────────────────────────────────────────────────────

def make_analysis_worker(model, processor, lock: threading.Lock,
                         state: dict) -> callable:
    """
    백그라운드 스레드에서 실행할 analysis_worker 클로저를 반환한다.
    state 딕셔너리 키: capture_enabled, is_analyzing, pending_result
    """
    def analysis_worker(frame_bgr: np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        annotated = frame_bgr
        obj_count = 0

        try:
            # 1단계: 이미지 설명 생성
            description = run_inference(model, processor, pil_image)

            # 2단계: 객체 그라운딩 감지
            obj_boxes = detect_objects(model, processor, pil_image)
            obj_count = len(obj_boxes)

            if obj_boxes:
                annotated = draw_object_boxes(frame_bgr, obj_boxes)
                print(f"[{DETECTION_LABEL} 감지] {obj_count}개 감지 — 바운딩 박스 적용")
            else:
                print(f"[{DETECTION_LABEL} 감지] {DETECTION_LABEL} 없음")

        except torch.cuda.OutOfMemoryError:
            description = "오류: GPU 메모리가 부족합니다."
        except Exception as e:
            description = f"분석 오류: {type(e).__name__}: {e}"

        print(f"\n[분석 결과]\n{description}\n")

        # 공유 상태를 모두 lock 내에서 원자적으로 갱신
        with lock:
            state['pending_result']  = (annotated, description, obj_count)
            state['is_analyzing']    = False
            state['capture_enabled'] = True

    return analysis_worker


# ── 메인 루프 ─────────────────────────────────────────────────────────────────

def run_loop(pipeline: rs.pipeline, model, processor) -> None:
    """실시간 영상 표시 및 사용자 입력 처리 루프."""
    lock  = threading.Lock()
    state = {
        'capture_enabled': True,
        'is_analyzing':    False,
        'pending_result':  None,   # (annotated_frame, text, obj_count)
    }
    result_image = None

    worker = make_analysis_worker(model, processor, lock, state)

    live_h = RESULT_WIN_WIDTH * CAMERA_HEIGHT // CAMERA_WIDTH
    cv2.namedWindow(LIVE_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_WIN, RESULT_WIN_WIDTH, live_h)

    print("\n조작 방법")
    print(f"  c  - 현재 프레임 캡처 및 분석 ({DETECTION_LABEL} 감지 포함)")
    print("  q  - 종료\n")

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
            with lock:
                analyzing = state['is_analyzing']
                enabled   = state['capture_enabled']

            if analyzing:
                label = "Analyzing...  please wait"
                fg    = (0, 165, 255)   # 주황
            else:
                label = "Press 'c' to capture  |  'q' to quit"
                fg    = (0, 220, 0)     # 초록

            _draw_outlined_text(display, label, (10, 34), STATUS_FONT_SCALE, fg)
            cv2.imshow(LIVE_WIN, display)

            # 분석 결과 수신 → 결과창 갱신 (OpenCV 창은 메인 스레드에서만 조작)
            with lock:
                if state['pending_result'] is not None:
                    f_out, desc, n = state['pending_result']
                    result_image           = build_result_image(f_out, desc, obj_count=n)
                    state['pending_result'] = None

            if result_image is not None:
                cv2.namedWindow(RESULT_WIN, cv2.WINDOW_NORMAL)
                cv2.imshow(RESULT_WIN, result_image)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c') and enabled and not analyzing:
                with lock:
                    state['capture_enabled'] = False
                    state['is_analyzing']    = True

                result_image = None
                try:
                    cv2.destroyWindow(RESULT_WIN)
                except cv2.error:
                    pass

                print("캡처됨 - 분석 시작...")
                threading.Thread(
                    target=worker,
                    args=(frame.copy(),),
                    daemon=True,
                ).start()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("종료.")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    # GPU 확인
    if not torch.cuda.is_available():
        print("오류: CUDA GPU를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # 모델 로드
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"모델 로딩 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # 카메라 초기화 및 메인 루프 시작
    pipeline = init_camera()
    run_loop(pipeline, model, processor)


if __name__ == "__main__":
    main()
