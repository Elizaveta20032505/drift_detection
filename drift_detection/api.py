"""
FastAPI приложение для загрузки данных и отслеживания дрейфа объектов.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import time
import pickle
import zipfile
import uuid
import threading
from collections import deque
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from drift_detector import ObjectDriftDetector
from cvat_loader import extract_images_from_archive

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_PATH_FILE = os.path.join(DATA_DIR, "model_path.txt")
MODEL_WEIGHTS_PATH = os.path.join(DATA_DIR, "trained_model.pt")
DATASET_PATH_FILE = os.path.join(DATA_DIR, "dataset_path.txt")
DATASET_CONFIG_PATH = os.path.join(DATA_DIR, "dataset_config.yaml")
BASELINE_IMAGES_FILE = os.path.join(DATA_DIR, "baseline_images.pkl")
TRAINING_STATUS_FILE = os.path.join(DATA_DIR, "training_status.txt")
TRAINING_ERROR_FILE = os.path.join(DATA_DIR, "training_error.txt")

print(f"Директория состояния: {DATA_DIR}")
print(f"Файл модели: {MODEL_PATH_FILE}")
print(f"Файл весов модели: {MODEL_WEIGHTS_PATH}")
print(f"Файл статуса обучения: {TRAINING_STATUS_FILE}")
print(f"Файл ошибки обучения: {TRAINING_ERROR_FILE}")

app = FastAPI(
    title="Drift Detection API",
    description="API для отслеживания дрейфа данных в системе детекции объектов",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

drift_detector: Optional[ObjectDriftDetector] = None
trained_model_path: Optional[str] = None
baseline_dataset_path: Optional[str] = None
baseline_images: List[np.ndarray] = []
baseline_ready: bool = False
training_status: str = "not_started"
training_error: Optional[str] = None

_pretrained_detector: Optional[ObjectDriftDetector] = None
_pretrained_object_classes: Optional[List[str]] = None


def get_pretrained_detector(object_classes: Optional[List[str]] = None) -> ObjectDriftDetector:
    """Детектор на YOLO11l (качается при первом вызове), без SAM. object_classes — фильтр по классам COCO (person, car, ...)."""
    global _pretrained_detector, _pretrained_object_classes
    if object_classes is None:
        object_classes = []
    same = _pretrained_object_classes == object_classes if _pretrained_object_classes is not None else not object_classes
    if _pretrained_detector is not None and same:
        return _pretrained_detector
    dummy = [np.zeros((64, 64, 3), dtype=np.uint8)]
    _pretrained_detector = ObjectDriftDetector(
        baseline_images=dummy,
        yolo_model_path=None,
        allowed_class_ids=None,
        allowed_name_tokens=object_classes if object_classes else None,
        use_sam=False,
    )
    _pretrained_object_classes = object_classes[:] if object_classes else []
    return _pretrained_detector

# Метрики Prometheus
drift_detections = Counter('object_drift_detections_total', 'Общее количество детекций дрейфа')
drift_psi = Histogram('object_drift_psi', 'PSI метрика дрейфа')
drift_kl_divergence = Histogram('object_drift_kl_divergence', 'KL divergence метрика дрейфа')
drift_ks_statistic = Histogram('object_drift_ks_statistic', 'KS статистика дрейфа')
drift_ks_pvalue = Histogram('object_drift_ks_pvalue', 'KS p-value (дрейф при < 0.05)')
drift_wasserstein = Histogram('object_drift_wasserstein', 'Расстояние Вассерштейна по яркости')
drift_js_divergence = Histogram('object_drift_js_divergence', 'Дивергенция Дженсена-Шеннона')
drift_aggregate_score = Histogram('object_drift_aggregate_score', 'Агрегированная метрика дрейфа (взвешенная)')
video_processing_seconds = Histogram('object_video_processing_seconds', 'Время обработки видео в секундах')
detections_count = Gauge('object_detections_count', 'Количество детекций объектов на текущем видео')
video_seconds_gauge = Gauge('object_drift_video_seconds', 'Текущая секунда видео (одна серия: при запуске нового видео график по сути начинается заново)')
drift_alert_gauge = Gauge('object_drift_alert', 'Детекция дрейфа (1=да, 0=нет)')
ph_alert_gauge = Gauge('object_drift_ph_alert', 'Page-Hinkley алерт (1=да, 0=нет)')

VIDEO_JOBS: Dict[str, Dict[str, Any]] = {}
VIDEO_JOBS_LOCK = threading.Lock()
PROCESSED_FRAMES_DIR = os.path.join(DATA_DIR, "processed_frames")
os.makedirs(PROCESSED_FRAMES_DIR, exist_ok=True)
DRIFT_FRAME_MAX_EDGE = 320

class TrainingResponse(BaseModel):
    message: str
    status: str
    epochs: Optional[int] = None
    model_path: Optional[str] = None


class VideoJobResponse(BaseModel):
    job_id: str
    message: str

def validate_and_save_archive(archive: UploadFile) -> str:
    """Валидирует и сохраняет ZIP архив"""
    if not archive.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате ZIP")

    temp_path = tempfile.mktemp(suffix='.zip')
    try:
        with open(temp_path, 'wb') as f:
            content = archive.file.read()
            f.write(content)

        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.testzip()

        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail=f"Невалидный ZIP архив: {str(e)}")

def validate_and_save_video(video: UploadFile) -> str:
    """Валидирует и сохраняет видеофайл"""
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Видео должно быть в формате MP4/AVI/MOV/MKV")

    temp_path = tempfile.mktemp(suffix=Path(video.filename).suffix)
    try:
        with open(temp_path, 'wb') as f:
            content = video.file.read()
            f.write(content)
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail=f"Не удалось сохранить видео: {str(e)}")


def apply_distortion(
    frame: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    noise_std: float = 0.0,
    hue_shift: int = 0,
    saturation_scale: float = 1.0
) -> np.ndarray:
    """Применяет искажения к кадру (яркость, контраст, шум, цвет)."""
    distorted = frame.astype(np.float32)

    if contrast != 1.0 or brightness != 0.0:
        distorted = distorted * contrast + brightness

    if noise_std > 0.0:
        noise = np.random.normal(0, noise_std, distorted.shape).astype(np.float32)
        distorted = distorted + noise

    distorted = np.clip(distorted, 0, 255).astype(np.uint8)

    if hue_shift != 0 or saturation_scale != 1.0:
        hsv = cv2.cvtColor(distorted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
        distorted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return distorted


BRIGHT_COLORS_BGR = [
    (0, 255, 0),    # зелёный
    (0, 0, 255),    # красный
    (255, 0, 0),    # синий
    (0, 255, 255),  # жёлтый
    (255, 0, 255),  # magenta
    (255, 165, 0),  # оранжевый (BGR)
    (0, 191, 255),  # deep sky blue
    (203, 192, 255), # lavender
]


def draw_detections(
    frame: np.ndarray,
    detections: List[dict],
    class_colors: Optional[Dict[str, tuple]] = None,
) -> np.ndarray:
    """
    Отрисовывает боксы детекций на кадре.
    class_colors: словарь class_name -> (B, G, R); если None — все зелёные.
    """
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det.get('confidence', 0.0)
        class_name = det.get('class_name', 'object')
        label = f"{class_name} {conf:.2f}"
        color = (0, 255, 0)
        if class_colors and class_name in class_colors:
            color = class_colors[class_name]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _get_or_assign_class_colors(detections: List[dict], class_colors: Dict[str, tuple]) -> None:
    """Дополняет class_colors новыми классами из detections (цвета из BRIGHT_COLORS_BGR)."""
    for det in detections:
        name = det.get('class_name', 'object')
        if name not in class_colors:
            class_colors[name] = BRIGHT_COLORS_BGR[len(class_colors) % len(BRIGHT_COLORS_BGR)]

def record_drift_metrics(metrics_dict: dict, processing_time: float, job_id: Optional[str] = None, video_second: Optional[float] = None):
    """Записывает метрики в Prometheus. job_id и video_second — для второй временной шкалы в Grafana."""
    try:
        if video_second is not None:
            video_seconds_gauge.set(video_second)
        # drift_detected ожидается как булев флаг
        if metrics_dict.get('drift_detected'):
            drift_detections.inc()
            drift_alert_gauge.set(1)
        else:
            drift_alert_gauge.set(0)
        ph_alert_gauge.set(1 if metrics_dict.get('page_hinkley_alert') else 0)

        psi_value = metrics_dict.get('psi')
        if psi_value is None:
            psi_value = metrics_dict.get('psi_mean')
        if psi_value is not None:
            drift_psi.observe(float(psi_value))

        kl_value = metrics_dict.get('kl_divergence')
        if kl_value is None:
            kl_value = metrics_dict.get('kl_mean')
        if kl_value is not None:
            drift_kl_divergence.observe(float(kl_value))

        ks_value = metrics_dict.get('ks_statistic')
        if ks_value is not None:
            drift_ks_statistic.observe(float(ks_value))
        ks_pval = metrics_dict.get('ks_pvalue')
        if ks_pval is not None:
            drift_ks_pvalue.observe(float(ks_pval))

        w_value = metrics_dict.get('wasserstein_distance')
        if w_value is not None:
            drift_wasserstein.observe(float(w_value))

        js_value = metrics_dict.get('js_divergence')
        if js_value is not None:
            drift_js_divergence.observe(float(js_value))
        agg_value = metrics_dict.get('aggregate_score')
        if agg_value is not None:
            drift_aggregate_score.observe(float(agg_value))

        if processing_time is not None:
            video_processing_seconds.observe(float(processing_time))

        detections_count.set(int(metrics_dict.get('total_detections', 0)))
    except Exception as e:
        print(f"Ошибка записи метрик: {e}")


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def init_video_job(job_id: str, output_dir: str):
    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {
            "status": "running",
            "message": "Обработка видео запущена",
            "output_dir": output_dir,
            "processed_frames": 0,
            "total_frames": None,
            "metrics_history": [],
            "last_metrics": None,
            "last_detection_second": None,
            "started_at": time.time(),
            "finished_at": None,
            "error": None,
        }


def update_video_job(job_id: str, **kwargs):
    with VIDEO_JOBS_LOCK:
        if job_id in VIDEO_JOBS:
            VIDEO_JOBS[job_id].update(kwargs)


def get_video_job(job_id: str) -> Optional[Dict[str, Any]]:
    with VIDEO_JOBS_LOCK:
        return VIDEO_JOBS.get(job_id)


def process_video_job(
    job_id: str,
    video_path: str,
    loop_video: bool,
    loop_count: int,
    frame_stride: int,
    drift_window_frames: int,
    drift_window_sec: Optional[float],
    only_frames_with_detections: bool,
    distortion_mode: str,
    brightness: float,
    contrast: float,
    noise_std: float,
    hue_shift: int,
    saturation_scale: float,
    segment_duration_sec: float,
    max_duration_sec: Optional[float],
):
    output_dir = os.path.join(PROCESSED_FRAMES_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    init_video_job(job_id, output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        update_video_job(job_id, status="error", error="Не удалось открыть видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    update_video_job(job_id, total_frames=total_frames)
    if drift_window_sec is not None and drift_window_sec > 0:
        drift_window_frames = max(2, int(drift_window_sec * fps / max(1, frame_stride)))

    stages = [
        {"name": "original", "brightness": 0.0, "contrast": 1.0, "noise_std": 0.0, "hue_shift": 0, "saturation_scale": 1.0},
        {"name": "brightness", "brightness": brightness, "contrast": 1.0, "noise_std": 0.0, "hue_shift": 0, "saturation_scale": 1.0},
        {"name": "contrast", "brightness": 0.0, "contrast": contrast, "noise_std": 0.0, "hue_shift": 0, "saturation_scale": 1.0},
        {"name": "noise", "brightness": 0.0, "contrast": 1.0, "noise_std": noise_std, "hue_shift": 0, "saturation_scale": 1.0},
        {"name": "color", "brightness": 0.0, "contrast": 1.0, "noise_std": 0.0, "hue_shift": hue_shift, "saturation_scale": saturation_scale},
    ]

    processed_frames = 0
    global_frame_index = 0
    loops_done = 0
    start_time = time.time()
    last_detection_second = None
    frame_window = deque(maxlen=drift_window_frames)
    class_colors = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                loops_done += 1
                if not loop_video or (loop_count > 0 and loops_done >= loop_count):
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            global_frame_index += 1
            current_second = global_frame_index / fps
            if max_duration_sec is not None and current_second >= max_duration_sec:
                break

            if frame_stride > 1 and (global_frame_index % frame_stride != 0):
                continue

            if distortion_mode == "uniform":
                processed_frame = apply_distortion(
                    frame,
                    brightness=brightness,
                    contrast=contrast,
                    noise_std=noise_std,
                    hue_shift=hue_shift,
                    saturation_scale=saturation_scale,
                )
                stage_name = "uniform"
            elif distortion_mode == "staged":
                stage_index = int(current_second / max(segment_duration_sec, 0.1)) % len(stages)
                stage = stages[stage_index]
                processed_frame = apply_distortion(
                    frame,
                    brightness=stage["brightness"],
                    contrast=stage["contrast"],
                    noise_std=stage["noise_std"],
                    hue_shift=stage["hue_shift"],
                    saturation_scale=stage["saturation_scale"],
                )
                stage_name = stage["name"]
            else:
                processed_frame = frame
                stage_name = "original"

            result = drift_detector.process_frame(processed_frame)
            detections = result['detections']
            object_images = result['object_images']

            # В окно кладём уменьшенную копию кадра для дрейфа (экономия памяти, без OOM)
            h, w = processed_frame.shape[:2]
            if max(h, w) > DRIFT_FRAME_MAX_EDGE:
                scale = DRIFT_FRAME_MAX_EDGE / max(h, w)
                small_frame = cv2.resize(
                    processed_frame,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small_frame = processed_frame.copy()
            if not only_frames_with_detections or len(detections) > 0:
                frame_window.append(small_frame)
            drift_metrics_raw = None
            if len(frame_window) > 0:
                try:
                    drift_metrics_raw = drift_detector.analyzer.analyze_drift(list(frame_window))
                except Exception as e:
                    print(f"Ошибка расчёта метрик дрейфа: {e}")
                    drift_metrics_raw = None

            drift_metrics = convert_numpy_types(drift_metrics_raw or {})

            if detections:
                last_detection_second = current_second
            _get_or_assign_class_colors(detections, class_colors)
            overlay = draw_detections(processed_frame, detections, class_colors)
            frame_filename = os.path.join(output_dir, f"frame_{processed_frames:06d}.jpg")
            cv2.imwrite(frame_filename, overlay)

            processed_frames += 1
            metrics_entry = {
                "frame_index": global_frame_index,
                "second": current_second,
                "processed_frames": processed_frames,
                "detections_count": len(detections),
                "drift_metrics": drift_metrics,
                "distortion_stage": stage_name,
            }

            with VIDEO_JOBS_LOCK:
                job = VIDEO_JOBS.get(job_id)
                if job is not None:
                    job["metrics_history"].append(metrics_entry)
                    job["last_metrics"] = metrics_entry
                    job["processed_frames"] = processed_frames
                    job["last_detection_second"] = last_detection_second

            if drift_metrics:
                metrics_payload = drift_metrics.copy()
                metrics_payload["total_detections"] = len(detections)
                record_drift_metrics(
                    metrics_payload,
                    time.time() - start_time,
                    job_id=job_id,
                    video_second=current_second,
                )

    except Exception as e:
        update_video_job(job_id, status="error", error=str(e))
        return
    finally:
        cap.release()
        if os.path.exists(video_path):
            os.unlink(video_path)

    update_video_job(job_id, status="completed", message="Обработка видео завершена", finished_at=time.time())


def process_video_job_pretrained(
    job_id: str,
    video_path: str,
    object_classes: List[str],
    frame_stride: int,
    drift_window_sec: float,
    only_frames_with_detections: bool,
    loop_video: bool,
    loop_count: int,
    max_duration_sec: Optional[float],
):
    """
    Обработка видео предобученной YOLO11l без baseline.
    Дрейф: скользящее окно W сек, на каждом шаге сравниваем старшую половину окна с младшей.
    Кадры с разметкой сохраняются в output_dir, скачать: GET /video_jobs/{job_id}/download.
    """
    output_dir = os.path.join(PROCESSED_FRAMES_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    init_video_job(job_id, output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        update_video_job(job_id, status="error", error="Не удалось открыть видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    update_video_job(job_id, total_frames=total_frames)

    # Окно в кадрах: из секунд и frame_stride
    frames_per_window = max(2, int(drift_window_sec * fps / max(1, frame_stride)))
    frame_window = deque(maxlen=frames_per_window)

    det = get_pretrained_detector(object_classes)
    detector, analyzer = det.detector, det.analyzer
    class_colors = {}

    processed_frames = 0
    global_frame_index = 0
    loops_done = 0
    start_time = time.time()
    last_detection_second = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                loops_done += 1
                if not loop_video or (loop_count > 0 and loops_done >= loop_count):
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            global_frame_index += 1
            current_second = global_frame_index / fps
            if max_duration_sec is not None and current_second >= max_duration_sec:
                break

            if frame_stride > 1 and (global_frame_index % frame_stride != 0):
                continue

            processed_frame = frame
            h, w = processed_frame.shape[:2]
            if max(h, w) > DRIFT_FRAME_MAX_EDGE:
                scale = DRIFT_FRAME_MAX_EDGE / max(h, w)
                small_frame = cv2.resize(processed_frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            else:
                small_frame = processed_frame.copy()

            detections = detector.detect_objects(processed_frame)
            if not only_frames_with_detections or len(detections) > 0:
                frame_window.append(small_frame)

            drift_metrics_raw = None
            if len(frame_window) >= 2:
                try:
                    drift_metrics_raw = analyzer.analyze_drift_stream(list(frame_window))
                except Exception as e:
                    print(f"Ошибка расчёта дрейфа (stream): {e}")
            drift_metrics = convert_numpy_types(drift_metrics_raw or {})

            if detections:
                last_detection_second = current_second
            _get_or_assign_class_colors(detections, class_colors)
            overlay = draw_detections(processed_frame, detections, class_colors)
            cv2.imwrite(os.path.join(output_dir, f"frame_{processed_frames:06d}.jpg"), overlay)
            processed_frames += 1

            metrics_entry = {
                "frame_index": global_frame_index,
                "second": current_second,
                "processed_frames": processed_frames,
                "detections_count": len(detections),
                "drift_metrics": drift_metrics,
                "distortion_stage": "original",
            }
            with VIDEO_JOBS_LOCK:
                job = VIDEO_JOBS.get(job_id)
                if job:
                    job["metrics_history"].append(metrics_entry)
                    job["last_metrics"] = metrics_entry
                    job["processed_frames"] = processed_frames
                    job["last_detection_second"] = last_detection_second

            if drift_metrics:
                metrics_payload = drift_metrics.copy()
                metrics_payload["total_detections"] = len(detections)
                record_drift_metrics(metrics_payload, time.time() - start_time, job_id=job_id, video_second=current_second)
    except Exception as e:
        update_video_job(job_id, status="error", error=str(e))
        return
    finally:
        cap.release()
        if os.path.exists(video_path):
            os.unlink(video_path)
    update_video_job(job_id, status="completed", message="Обработка видео (pretrained) завершена", finished_at=time.time())


# API endpoints
@app.get("/status")
async def get_status():
    """Получить статус системы"""
    return {
        "model_trained": trained_model_path is not None and os.path.exists(trained_model_path),
        "trained_model_path": trained_model_path,
        "detector_ready": drift_detector is not None,
        "data_directory": DATA_DIR,
        "training_status": training_status,
        "training_error": training_error,
        "ready_for_drift_detection": drift_detector is not None and trained_model_path is not None
    }

@app.get("/metrics")
def get_metrics():
    """Получить метрики Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/upload_baseline")
async def upload_baseline(
    cvat_archive: UploadFile = File(..., description="CVAT ZIP архив с изображениями для baseline")
):
    """
    Загружает baseline-изображения из CVAT ZIP архива без переобучения модели.

    Из архива извлекаются все изображения (игнорируя разметку), они сохраняются
    в файл BASELINE_IMAGES_FILE и используются для инициализации DriftAnalyzer.
    """
    global baseline_images, baseline_ready, drift_detector, trained_model_path

    temp_archive = validate_and_save_archive(cvat_archive)

    try:
        images = extract_images_from_archive(temp_archive)
        if not images:
            raise HTTPException(
                status_code=400,
                detail="В архиве не найдено ни одного изображения (jpg/png)"
            )

        baseline_images = images
        baseline_ready = True

        try:
            with open(BASELINE_IMAGES_FILE, 'wb') as f:
                pickle.dump(baseline_images, f)
        except Exception as e:
            print(f"✗ Ошибка сохранения baseline изображений: {e}")

        if trained_model_path is not None and os.path.exists(trained_model_path):
            try:
                sam_path = "sam_b.pt" if os.path.exists("sam_b.pt") else None
                drift_detector = ObjectDriftDetector(
                    baseline_images=baseline_images,
                    yolo_model_path=trained_model_path,
                    allowed_class_ids=None,
                    sam_checkpoint_path=sam_path if sam_path else "sam_b.pt",
                    use_sam=sam_path is not None
                )
            except Exception as e:
                print(f"✗ Ошибка переинициализации детектора с новым baseline: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Baseline сохранён, но не удалось инициализировать детектор: {e}"
                )

        return {
            "message": f"Baseline успешно загружен: {len(baseline_images)} изображений",
            "images_count": len(baseline_images),
            "model_ready": trained_model_path is not None and os.path.exists(trained_model_path),
        }
    finally:
        if os.path.exists(temp_archive):
            os.unlink(temp_archive)

@app.post("/train_model")
async def train_model(
    cvat_archive: UploadFile = File(..., description="CVAT архив с изображениями и аннотациями YOLO"),
    epochs: int = Form(default=50, description="Количество эпох обучения (минимум 50 для хорошего результата)"),
    batch_size: int = Form(default=2, description="Размер батча"),
    imgsz: int = Form(default=320, description="Размер изображений")
):
    """
    Обучает YOLO модель на данных из CVAT архива.

    Args:
        cvat_archive: CVAT архив с изображениями и аннотациями YOLO
        epochs: Количество эпох обучения
        batch_size: Размер батча
        imgsz: Размер изображений

    ВНИМАНИЕ: Обучение выполняется синхронно и может занять много времени.
    """
    global drift_detector, trained_model_path, baseline_dataset_path, training_status, training_error

    temp_archive = validate_and_save_archive(cvat_archive)

    try:
        print("Создаем датасет из CVAT архива...")
        from model_trainer import prepare_dataset_from_cvat_archive
        import tempfile

        temp_dataset_dir = tempfile.mkdtemp(prefix="cvat_dataset_")
        print(f"Временная директория датасета: {temp_dataset_dir}")

        try:
            dataset_yaml = prepare_dataset_from_cvat_archive(
                archive_path=temp_archive,
                output_dir=temp_dataset_dir,
                object_class_id=0
            )
            baseline_dataset_path = dataset_yaml
            print(f"✓ Датасет из CVAT создан: {dataset_yaml}")
        except Exception as e:
            import shutil
            if os.path.exists(temp_dataset_dir):
                shutil.rmtree(temp_dataset_dir)
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка обработки CVAT архива: {str(e)}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка сохранения архива: {str(e)}"
        )

    import threading

    training_status = "training"
    training_error = None
    with open(TRAINING_STATUS_FILE, 'w') as f:
        f.write("training")

    def train_sync():
        import shutil
        global trained_model_path, drift_detector, training_status, training_error

        data_dir = os.path.join(os.getcwd(), "data")

        try:
            print("🚀 Начинаем обучение модели в фоне...")
            print(f"Количество baseline изображений: {len(baseline_images)}")
            print(f"Путь к датасету: {baseline_dataset_path}")

            if epochs < 10:
                print(f"⚠️  КРИТИЧНО: {epochs} эпох - ЭТОГО НЕДОСТАТОЧНО!")
                print("   YOLO модели нуждаются минимум в 50-100 эпохах для обучения")
                print("   С 1 эпохой модель работает как случайный классификатор!")
            elif epochs < 50:
                print(f"⚠️  ВНИМАНИЕ: {epochs} эпох маловато, результат будет плохим")

            # Вызываем обучение
            from model_trainer import train_yolo_model
            model_path = train_yolo_model(
                dataset_yaml=baseline_dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device="cpu"
            )

            print("Обучение завершено, сохраняем модель...")

            print(f"Проверяем файл модели: {model_path}")
            print(f"Файл модели существует: {os.path.exists(model_path)}")
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                print(f"Размер файла модели: {file_size} байт")

                model_in_data = os.path.join(data_dir, "trained_model.pt")
                shutil.copy2(model_path, model_in_data)
                print(f"✓ Модель сохранена в папку data: {model_in_data}")
                print(f"Файл в data существует: {os.path.exists(model_in_data)}")

                shutil.copy2(model_path, MODEL_WEIGHTS_PATH)
                print(f"✓ Модель скопирована в постоянную директорию: {MODEL_WEIGHTS_PATH}")
                print(f"Файл весов существует: {os.path.exists(MODEL_WEIGHTS_PATH)}")

                trained_model_path = MODEL_WEIGHTS_PATH
                with open(MODEL_PATH_FILE, 'w') as f:
                    f.write(MODEL_WEIGHTS_PATH)
                print(f"✓ Путь к модели сохранен в {MODEL_PATH_FILE}: {MODEL_WEIGHTS_PATH}")
                print("✅ Модель успешно сохранена!")

                if len(baseline_images) > 0:
                    try:
                        with open(BASELINE_IMAGES_FILE, 'wb') as f:
                            pickle.dump(baseline_images, f)
                        print(f"✓ Baseline сохранен: {len(baseline_images)} изображений в {BASELINE_IMAGES_FILE}")
                    except Exception as e:
                        print(f"✗ Предупреждение: не удалось сохранить baseline: {e}")

                try:
                    sam_path = "sam_b.pt" if os.path.exists("sam_b.pt") else None
                    drift_detector = ObjectDriftDetector(
                        baseline_images=baseline_images,
                        yolo_model_path=trained_model_path,
                        allowed_class_ids=None,
                        sam_checkpoint_path=sam_path if sam_path else "sam_b.pt",
                        use_sam=sam_path is not None
                    )
                    training_status = "completed"
                    with open(TRAINING_STATUS_FILE, 'w') as f:
                        f.write("completed")
                    print("✅ Детектор инициализирован с новой моделью")
                except Exception as e:
                    training_status = "error"
                    training_error = f"Ошибка инициализации детектора: {str(e)}"
                    with open(TRAINING_STATUS_FILE, 'w') as f:
                        f.write("error")
                    with open(TRAINING_ERROR_FILE, 'w') as f:
                        f.write(str(e))
                    print(f"❌ Ошибка инициализации детектора: {e}")
            else:
                print(f"✗ Файл модели не найден: {model_path}")
                training_status = "error"
                training_error = f"Обученная модель не найдена по пути: {model_path}"
                with open(TRAINING_STATUS_FILE, 'w') as f:
                    f.write("error")
                with open(TRAINING_ERROR_FILE, 'w') as f:
                    f.write(training_error)

        except Exception as e:
            training_status = "error"
            training_error = str(e)
            with open(TRAINING_STATUS_FILE, 'w') as f:
                f.write("error")
            with open(TRAINING_ERROR_FILE, 'w') as f:
                f.write(str(e))
            print(f"❌ Ошибка обучения модели: {e}")
            import traceback
            print("Traceback:")
            traceback.print_exc()

    print("Запускаем обучение...")
    train_sync()

    if training_status == "completed":
        return {
            "message": "Обучение модели завершено успешно",
            "status": "completed",
            "epochs": epochs,
            "model_path": trained_model_path
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обучения модели: {training_error}"
        )

@app.post("/process_video", response_model=VideoJobResponse)
async def process_video(
    video: UploadFile = File(..., description="Видео файл для анализа дрейфа"),
    loop_video: bool = Form(False, description="Зацикливать видео"),
    loop_count: int = Form(1, description="Количество циклов (0 = бесконечно, если задан max_duration_sec)"),
    frame_stride: int = Form(5, description="Обрабатывать каждый N-й кадр"),
    drift_window_frames: int = Form(30, description="Размер скользящего окна по кадрам для расчёта дрейфа"),
    drift_window_sec: Optional[float] = Form(None, description="Окно в секундах (если задано, переопределяет drift_window_frames по fps)"),
    distortion_mode: str = Form("none", description="none | uniform | staged"),
    brightness: float = Form(0.0, description="Смещение яркости (0-255)"),
    contrast: float = Form(1.0, description="Коэффициент контраста (1.0 = без изменений)"),
    noise_std: float = Form(0.0, description="Стандартное отклонение шума"),
    hue_shift: int = Form(0, description="Сдвиг оттенка (0-180)"),
    saturation_scale: float = Form(1.0, description="Множитель насыщенности"),
    segment_duration_sec: float = Form(10.0, description="Длительность одного сегмента при staged-режиме"),
    max_duration_sec: Optional[float] = Form(None, description="Ограничение по длительности обработки (секунды)"),
    only_frames_with_detections: bool = Form(False, description="Считать дрейф только по кадрам, где есть хотя бы одна детекция"),
):
    """
    Обработка видео для определения дрейфа.

    Поддерживаются режимы:
    - none: без искажений;
    - uniform: одно искажение на весь поток;
    - staged: последовательные искажения по сегментам.
    """
    global drift_detector, trained_model_path, baseline_ready

    if trained_model_path is None or not os.path.exists(trained_model_path):
        raise HTTPException(
            status_code=400,
            detail="Модель не обучена. Сначала обучите модель через /train_model с CVAT архивом"
        )

    if drift_detector is None:
        raise HTTPException(status_code=400, detail="Детектор не инициализирован")

    if not baseline_ready:
        raise HTTPException(
            status_code=400,
            detail="Baseline изображения не загружены. Сначала загрузите baseline через /upload_baseline"
        )

    if frame_stride < 1:
        raise HTTPException(status_code=400, detail="frame_stride должен быть >= 1")

    if drift_window_frames < 1 and (drift_window_sec is None or drift_window_sec <= 0):
        raise HTTPException(status_code=400, detail="Задайте drift_window_frames >= 1 или drift_window_sec > 0")

    if distortion_mode not in {"none", "uniform", "staged"}:
        raise HTTPException(status_code=400, detail="distortion_mode должен быть none|uniform|staged")

    if loop_video and loop_count == 0 and max_duration_sec is None:
        raise HTTPException(
            status_code=400,
            detail="При loop_count=0 укажите max_duration_sec, чтобы остановить бесконечный цикл"
        )

    temp_video = validate_and_save_video(video)
    job_id = str(uuid.uuid4())

    worker = threading.Thread(
        target=process_video_job,
        args=(
            job_id,
            temp_video,
            loop_video,
            loop_count,
            frame_stride,
            drift_window_frames,
            drift_window_sec,
            only_frames_with_detections,
            distortion_mode,
            brightness,
            contrast,
            noise_std,
            hue_shift,
            saturation_scale,
            segment_duration_sec,
            max_duration_sec,
        ),
        daemon=True,
    )
    worker.start()

    return VideoJobResponse(job_id=job_id, message="Задача обработки видео запущена")


@app.post("/process_video_pretrained", response_model=VideoJobResponse)
async def process_video_pretrained(
    video: UploadFile = File(..., description="Видео для анализа (предобученная YOLO11l, без baseline)"),
    object_classes: str = Form("person,car", description="Классы COCO через запятую: person, car, truck, ..."),
    frame_stride: int = Form(5, description="Обрабатывать каждый N-й кадр"),
    drift_window_sec: float = Form(10.0, description="Скользящее окно для дрейфа (секунды)"),
    only_frames_with_detections: bool = Form(False, description="Считать дрейф только по кадрам с детекциями"),
    loop_video: bool = Form(False),
    loop_count: int = Form(1),
    max_duration_sec: Optional[float] = Form(None),
):
    """
    Обработка видео предобученной YOLO11l.
    Без baseline: дрейф по скользящему окну.
    Кадры с разметкой сохраняются; скачать: GET /video_jobs/{job_id}/download.
    """
    classes = [c.strip().lower() for c in object_classes.split(",") if c.strip()]
    if not classes:
        classes = ["person", "car"]

    temp_video = validate_and_save_video(video)
    job_id = str(uuid.uuid4())
    worker = threading.Thread(
        target=process_video_job_pretrained,
        args=(
            job_id,
            temp_video,
            classes,
            frame_stride,
            drift_window_sec,
            only_frames_with_detections,
            loop_video,
            loop_count,
            max_duration_sec,
        ),
        daemon=True,
    )
    worker.start()
    return VideoJobResponse(job_id=job_id, message="Задача обработки видео (pretrained) запущена")


@app.get("/video_jobs/{job_id}")
async def get_video_job_status(job_id: str):
    job = get_video_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job


@app.get("/video_jobs/{job_id}/metrics")
async def get_video_job_metrics(job_id: str, limit: int = 100):
    job = get_video_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    metrics_history = job.get("metrics_history", [])
    return metrics_history[-limit:]


@app.get("/video_jobs/{job_id}/download")
async def download_video_job_frames(job_id: str):
    job = get_video_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    output_dir = job.get("output_dir")
    if not output_dir or not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="Архив с кадрами пока не готов")

    archive_base = os.path.join(DATA_DIR, f"processed_frames_{job_id}")
    archive_path = shutil.make_archive(archive_base, 'zip', output_dir)
    with open(archive_path, 'rb') as f:
        content = f.read()

    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=processed_frames_{job_id}.zip"}
    )

def load_saved_state():
    """Загружает сохраненное состояние при старте приложения"""
    global trained_model_path, baseline_dataset_path, baseline_images, drift_detector, baseline_ready

    print(f"Загрузка состояния из: {DATA_DIR}")

    # Загружаем статус обучения
    global training_status, training_error
    if os.path.exists(TRAINING_STATUS_FILE):
        try:
            with open(TRAINING_STATUS_FILE, 'r') as f:
                saved_status = f.read().strip()
                if saved_status in ["not_started", "training", "completed", "error"]:
                    training_status = saved_status
                    print(f"✓ Загружен статус обучения: {training_status}")
        except Exception as e:
            print(f"✗ Ошибка загрузки статуса обучения: {e}")

    if os.path.exists(TRAINING_ERROR_FILE):
        try:
            with open(TRAINING_ERROR_FILE, 'r') as f:
                training_error = f.read().strip()
                if training_error:
                    print(f"✓ Загружена ошибка обучения: {training_error}")
        except Exception as e:
            print(f"✗ Ошибка загрузки ошибки обучения: {e}")
    # Загружаем baseline изображения, если есть
    baseline_images = []
    baseline_ready = False
    if os.path.exists(BASELINE_IMAGES_FILE):
        try:
            with open(BASELINE_IMAGES_FILE, 'rb') as f:
                baseline_images = pickle.load(f)
            if isinstance(baseline_images, list) and len(baseline_images) > 0:
                baseline_ready = True
                print(f"✓ Загружен baseline: {len(baseline_images)} изображений из {BASELINE_IMAGES_FILE}")
            else:
                baseline_images = []
                baseline_ready = False
                print(f"✗ Файл baseline пустой, baseline будет считаться неинициализированным")
        except Exception as e:
            baseline_images = []
            baseline_ready = False
            print(f"✗ Ошибка загрузки baseline изображений: {e}")

    # Проверяем модель
    print(f"Проверяем модель в: {MODEL_WEIGHTS_PATH}")
    print(f"Файл существует: {os.path.exists(MODEL_WEIGHTS_PATH)}")
    if os.path.exists(MODEL_WEIGHTS_PATH):
        trained_model_path = MODEL_WEIGHTS_PATH
        print(f"✓ Найдена обученная модель: {MODEL_WEIGHTS_PATH}")
        print(f"Размер файла: {os.path.getsize(MODEL_WEIGHTS_PATH) if os.path.exists(MODEL_WEIGHTS_PATH) else 'N/A'} байт")

        # Проверяем содержимое директории
        print(f"Содержимое директории {DATA_DIR}:")
        try:
            files = os.listdir(DATA_DIR)
            for f in files:
                print(f"  - {f}")
        except Exception as e:
            print(f"  Ошибка чтения директории: {e}")

        # Инициализируем детектор, только если есть baseline
        try:
            if baseline_ready:
                sam_path = "sam_b.pt" if os.path.exists("sam_b.pt") else None
                drift_detector = ObjectDriftDetector(
                    baseline_images=baseline_images,
                    yolo_model_path=trained_model_path,
                    allowed_class_ids=None,
                    sam_checkpoint_path=sam_path if sam_path else "sam_b.pt",
                    use_sam=sam_path is not None
                )
                print(f"✓ Детектор инициализирован с обученной моделью и baseline ({len(baseline_images)} изображений)")
            else:
                drift_detector = None
                print("✗ Baseline не найден, детектор дрейфа не инициализирован (ожидается загрузка baseline)")
        except Exception as e:
            print(f"✗ Ошибка инициализации детектора: {e}")
    else:
        print("✗ Модель не найдена")

# Загружаем состояние при импорте модуля
load_saved_state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
