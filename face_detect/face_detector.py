#  face_detector.py

import cv2
import numpy as np
import os
import face_recognition
import asyncio
import aiohttp
import logging
import threading
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from config import Config
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.detection_times = []
        self.max_samples = 20
        self.process_frames = Config.base_process_frames

    def add_frame_time(self, time_ms: float):
        self.frame_times.append(time_ms)
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)

    def add_detection_time(self, time_ms: float):
        self.detection_times.append(time_ms)
        if len(self.detection_times) > self.max_samples:
            self.detection_times.pop(0)

    def get_avg_fps(self) -> float:
        return 1000 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0

    def get_avg_detection_time(self) -> float:
        return sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0

    def update_process_frames(self):
        fps = self.get_avg_fps()
        if fps < Config.min_fps_threshold:
            self.process_frames = min(self.process_frames + 1, 6)
        elif fps > Config.min_fps_threshold * 1.5:
            self.process_frames = max(self.process_frames - 1, 2)

class FaceCache:
    def __init__(self, cache_file: str, cache_ttl: int = 86400):
        self.cache_file = cache_file
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.last_save = 0
        self.save_interval = 300
        self.load_cache()

    def load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    now = datetime.now()
                    for staff_id, cache_data in data.items():
                        cached_at = datetime.fromisoformat(cache_data.get('cached_at', '1970-01-01'))
                        if (now - cached_at).total_seconds() < self.cache_ttl:
                            if 'encoding' in cache_data:
                                cache_data['encoding'] = np.array(cache_data['encoding'], dtype=np.float32)
                            self.cache[staff_id] = cache_data
                    logger.info(f"Loaded {len(self.cache)} valid cached encodings")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    def save_cache(self):
        current_time = time.time()
        if current_time - self.last_save < self.save_interval:
            return
        try:
            cache_to_save = {
                staff_id: {
                    **cache_data,
                    'encoding': cache_data['encoding'].tolist() if 'encoding' in cache_data else None
                } for staff_id, cache_data in self.cache.items()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_to_save, f)
            self.last_save = current_time
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def get_encoding(self, staff_id: str, image_url: str) -> Optional[np.ndarray]:
        entry = self.cache.get(staff_id)
        if entry and entry.get('image_url') == image_url:
            return entry.get('encoding')
        return None

    def set_encoding(self, staff_id: str, image_url: str, encoding: np.ndarray):
        self.cache[staff_id] = {
            'image_url': image_url,
            'encoding': encoding.astype(np.float32),
            'cached_at': datetime.now().isoformat()
        }

class FaceDetector:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.known_ids: List[str] = []
        self.staff_list: List[Dict] = []
        self.lock = threading.Lock()
        self.last_staff_ids: set = set()
        self.running = False
        self.recently_recorded: Dict[str, float] = {}
        self.cache = FaceCache(config.cache_file)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.recognition_buffer: Dict[str, int] = {}
        self.performance = PerformanceMonitor()
        self.encoding_tree: Optional[cKDTree] = None

    def _build_encoding_tree(self):
        if self.known_encodings:
            self.encoding_tree = cKDTree(np.array(self.known_encodings, dtype=np.float32))
        else:
            self.encoding_tree = None

    async def initialize(self):
        await self.refresh_staff_data()
        self.running = True

    async def fetch_staff(self) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.max_image_download_retries):
                try:
                    async with session.get(self.config.staff_api_url, timeout=10) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        staff = data.get('data', {}).get('staff', []) if isinstance(data, dict) else []
                        logger.info(f"Fetched {len(staff)} staff members")
                        return staff
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_image_download_retries - 1:
                        await asyncio.sleep(0.5 * (2 ** attempt))
        logger.error(f"Failed to fetch staff data after {self.config.max_image_download_retries} attempts")
        return []

    async def load_image_optimized(self, url: str) -> Optional[np.ndarray]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            for attempt in range(self.config.max_image_download_retries):
                try:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        content = await resp.read()
                        return await asyncio.get_event_loop().run_in_executor(self.executor, self._decode_image, content)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                    if attempt < self.config.max_image_download_retries - 1:
                        await asyncio.sleep(0.5 * attempt)
        return None

    def _decode_image(self, content: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(content, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                return None
            height, width = image.shape[:2]
            scale = min(640 / width, 480 / height)
            if scale < 1:
                image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    async def process_image_optimized(self, staff: Dict) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
        image_url = staff.get('imageUrl')
        staff_id = staff.get('_id')
        name = staff.get('name')
        if not staff_id or not image_url:
            logger.warning(f"Missing ID or image for staff {name}")
            return None, None, None
        cached_encoding = self.cache.get_encoding(staff_id, image_url)
        if cached_encoding is not None:
            return cached_encoding, name, staff_id
        image = await self.load_image_optimized(image_url)
        if image is None:
            return None, None, None
        encoding = await asyncio.get_event_loop().run_in_executor(self.executor, self._extract_face_encoding, image)
        if encoding is not None:
            self.cache.set_encoding(staff_id, image_url, encoding)
        return encoding, name, staff_id

    def _extract_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image, model='small', num_jitters=1)
            return encodings[0] if encodings else None
        except Exception as e:
            logger.error(f"Error extracting face encoding: {e}")
            return None

    async def refresh_staff_data(self):
        staff_list = await self.fetch_staff()
        if not staff_list:
            return
        new_staff_ids = {s.get('_id') for s in staff_list if s.get('_id')}
        if new_staff_ids == self.last_staff_ids and self.known_encodings:
            logger.info("Staff data unchanged")
            return
        batch_size = 8
        new_encodings, new_names, new_ids = [], [], []
        for i in range(0, len(staff_list), batch_size):
            batch = staff_list[i:i + batch_size]
            results = await asyncio.gather(*[self.process_image_optimized(s) for s in batch], return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    continue
                enc, name, sid = result
                if enc is not None:
                    new_encodings.append(enc)
                    new_names.append(name)
                    new_ids.append(sid)
        with self.lock:
            self.known_encodings = new_encodings
            self.known_names = new_names
            self.known_ids = new_ids
            self.staff_list = staff_list
            self.last_staff_ids = new_staff_ids
            self._build_encoding_tree()
        self.cache.save_cache()
        logger.info(f"Staff data updated: {len(self.known_encodings)} encodings loaded")

    async def background_refresh(self):
        while self.running:
            try:
                await self.refresh_staff_data()
                await asyncio.sleep(self.config.refresh_interval)
            except Exception as e:
                logger.error(f"Background refresh error: {e}")
                await asyncio.sleep(5)

    async def record_attendance(self, staff_id: str, name: str, type: str = None, note: str = None):
        record_type = type or self.config.default_attendance_type
        valid_types = ['check-in', 'check-out']
        if record_type not in valid_types:
            logger.error(f"Invalid attendance type: {record_type}")
            return False
        now = time.time()
        buffer_key = f"{staff_id}_{record_type}"
        for key in list(self.recently_recorded.keys()):
            if key.startswith(f"{staff_id}_") and key != buffer_key:
                self.recently_recorded[key] = 0
        cooldown_seconds = self.config.attendance_cooldown.total_seconds()
        last_recorded = self.recently_recorded.get(buffer_key, 0)
        if now - last_recorded < cooldown_seconds:
            logger.info(f"⚠️ Cooldown active for {name} ({record_type})")
            return False
        self.recently_recorded[buffer_key] = now
        payload = {
            "staffId": staff_id,
            "name": name,
            "type": record_type,
            "timestamp": datetime.now().isoformat()
        }
        if note:
            payload["note"] = note
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.post(self.config.attendance_url, json=payload) as resp:
                    if resp.status == 201:
                        logger.info(f"✓ Attendance recorded: {name} - {record_type}")
                        return True
                    else:
                        text = await resp.text()
                        logger.error(f"❌ POST failed: {resp.status} - {text}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error recording attendance for {name}: {e}")
            return False

    async def detect_and_check_optimized(self, frame: np.ndarray, attendance_type: str) -> Tuple[np.ndarray, List[Dict]]:
        start_time = time.time()
        small_frame = cv2.resize(frame, (0, 0), fx=self.config.frame_resize_scale, fy=self.config.frame_resize_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model=self.config.detection_model)
        if not face_locations:
            self.performance.add_detection_time((time.time() - start_time) * 1000)
            return frame, []
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
        allowed_staff = []
        with self.lock:
            if not self.known_encodings:
                return self._draw_unknown_faces(frame, face_locations), []
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                if self.encoding_tree:
                    dist, idx = self.encoding_tree.query(face_encoding, k=1)
                    dist = dist[0] if isinstance(dist, np.ndarray) else dist
                    idx = idx[0] if isinstance(idx, np.ndarray) else idx
                    if dist <= self.config.max_face_distance:
                        name = self.known_names[idx]
                        staff_id = self.known_ids[idx]
                        confidence = 1 - dist
                        buffer_key = f"{staff_id}_{attendance_type}"
                        for key in list(self.recognition_buffer.keys()):
                            if key.startswith(f"{staff_id}_") and key != buffer_key:
                                self.recognition_buffer[key] = 0
                        self.recognition_buffer[buffer_key] = self.recognition_buffer.get(buffer_key, 0) + 1
                        if self.recognition_buffer[buffer_key] >= self.config.recognition_threshold:
                            allowed_staff.append({
                                "name": name,
                                "type": attendance_type,
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "confidence": f"{confidence:.2f}",
                                "staff_id": staff_id
                            })
                            asyncio.create_task(self.record_attendance(staff_id, name, type=attendance_type))
                            self.recognition_buffer[buffer_key] = 0
                        self._draw_face_box(frame, (top, right, bottom, left), name, confidence, (0, 255, 0))
                    else:
                        self._draw_face_box(frame, (top, right, bottom, left), "Unknown", 0, (0, 0, 255))
                else:
                    self._draw_face_box(frame, (top, right, bottom, left), "Unknown", 0, (0, 0, 255))
        self.performance.add_detection_time((time.time() - start_time) * 1000)
        return frame, allowed_staff

    def _draw_face_box(self, frame: np.ndarray, location: Tuple[int, int, int, int], name: str, confidence: float, color: Tuple[int, int, int]):
        top, right, bottom, left = [int(coord / self.config.frame_resize_scale) for coord in location]
        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
        label = f"{name} ({confidence:.0%})" if confidence > 0 else name
        cv2.putText(frame, label, (left + 5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_unknown_faces(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> np.ndarray:
        for (top, right, bottom, left) in face_locations:
            self._draw_face_box(frame, (top, right, bottom, left), "Loading...", 0, (255, 255, 0))
        return frame