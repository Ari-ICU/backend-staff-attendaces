import cv2
import numpy as np
import face_recognition
import asyncio
import aiohttp
import logging
import threading
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, font
import tkinter.font as tkfont
from PIL import Image, ImageTk
import warnings
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import json
import os
from dotenv import load_dotenv
from scipy.spatial import cKDTree
import pyttsx3
import pygame

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    staff_api_url: str = os.getenv('STAFF_API_URL', 'http://localhost:5000/api/staff')
    attendance_url: str = os.getenv('ATTENDANCE_URL', 'http://localhost:5000/api/attendance')
    tolerance: float = 0.5
    frame_resize_scale: float = 0.25
    base_process_frames: int = 3
    max_image_download_retries: int = 3
    detection_model: str = 'hog'
    refresh_interval: int = 30
    default_attendance_type: str = 'check-in'
    attendance_cooldown: timedelta = timedelta(seconds=120)
    cache_file: str = 'face_cache.json'
    max_face_distance: float = 0.5
    recognition_threshold: int = 3
    sound_debounce_interval: float = 8.0
    max_sound_queue_size: int = 10
    min_fps_threshold: float = 15.0
    sound_directory: str = 'sounds'
    voice_id: int = 0

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
        """
        Record attendance for a staff member while respecting cooldown per type.
        Resets cooldown for a type when switching between types.
        """
        record_type = type or self.config.default_attendance_type
        valid_types = ['check-in', 'check-out', 'leave']

        if record_type not in valid_types:
            logger.error(f"Invalid attendance type: {record_type}")
            return False

        now = time.time()
        buffer_key = f"{staff_id}_{record_type}"

        # Reset cooldown for other types when switching type
        for key in list(self.recently_recorded.keys()):
            if key.startswith(f"{staff_id}_") and key != buffer_key:
                self.recently_recorded[key] = 0

        # Check cooldown for this type
        cooldown_seconds = self.config.attendance_cooldown.total_seconds()
        last_recorded = self.recently_recorded.get(buffer_key, 0)
        if now - last_recorded < cooldown_seconds:
            logger.info(f"⚠️ Cooldown active for {name} ({record_type})")
            return False

        # Update last recorded time
        self.recently_recorded[buffer_key] = now

        # Prepare payload
        payload = {
            "staffId": staff_id,
            "name": name,
            "type": record_type,
            "timestamp": datetime.now().isoformat()
        }
        if note:
            payload["note"] = note

        # Send attendance data
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

                        # Reset recognition buffer for other types
                        for key in list(self.recognition_buffer.keys()):
                            if key.startswith(f"{staff_id}_") and key != buffer_key:
                                self.recognition_buffer[key] = 0

                        # Increment current type buffer
                        self.recognition_buffer[buffer_key] = self.recognition_buffer.get(buffer_key, 0) + 1

                        if self.recognition_buffer[buffer_key] >= self.config.recognition_threshold:
                            allowed_staff.append({
                                "name": name,
                                "type": attendance_type,
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "confidence": f"{confidence:.2f}",
                                "staff_id": staff_id
                            })
                            # Record attendance
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

class ModernFaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ Face Recognition Attendance System")
        self.root.geometry("1000x850")
        self.dark_mode = False

        # Initialize pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

        # Create sounds directory if not exists
        os.makedirs('sounds', exist_ok=True)

        # Generate default sounds if not exist
        self._generate_default_sounds_if_missing()

        # TTS Engine Setup (Professional Voice)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 100)     # Natural speaking pace
        self.tts_engine.setProperty('volume', 0.9)

        # Prefer female voice (Microsoft Zira or similar)
        voices = self.tts_engine.getProperty('voices')
        for v in voices:
            if "Samantha" in v.name:
                self.tts_engine.setProperty('voice', v.id)
                print(f"Using voice: {v.name}")
                break
        else:
            # Fallback to first voice if Samantha is not found
            self.tts_engine.setProperty('voice', voices[0].id)
            print(f"Samantha not found, using fallback voice: {voices[0].name}")

        # Sound Queue & Control
        self.sound_queue = queue.PriorityQueue(maxsize=Config.max_sound_queue_size)
        self.sound_thread = None
        self.sound_running = False
        self.last_sound_times = {}

        # Volume control
        self.volume_var = tk.DoubleVar(value=0.9)

        self.style = ttk.Style()
        self.configure_styles()
        self.config = Config()
        self.detector = FaceDetector(self.config)
        self.cap = None
        self.running = False
        self.async_loop = None
        self.thread = None
        self.attendance_type_var = tk.StringVar(value=self.config.default_attendance_type)
        self.fps_var = tk.StringVar(value="📊 FPS: 0")
        self.detection_time_var = tk.StringVar(value="⏱️ Detection: 0ms")
        self.status_var = tk.StringVar(value="🟢 System Idle")
        self.staff_count_var = tk.StringVar(value="👥 Registered Staff: 0")
        self.frame_queue = queue.Queue(maxsize=2)
        self.last_frame_time = time.time()

        self.setup_gui()
        self.update_performance_display()

    def configure_styles(self):
        self.bg_light = '#e6e9ef'
        self.bg_dark = '#1a1c22'
        self.card_light = '#ffffff'
        self.card_dark = '#25272e'
        self.accent = '#3b82f6'
        self.success = '#22c55e'
        self.warning = '#f59e0b'
        self.danger = '#ef4444'
        self.text_light = '#1f2937'
        self.text_dark = '#d1d5db'

        self.default_font = tkfont.Font(family="Inter", size=11)
        self.title_font = tkfont.Font(family="Inter", size=18, weight="bold")
        self.root.option_add("*Font", self.default_font)

        self.style.theme_use('default')
        self.apply_light_theme()

    def _generate_default_sounds_if_missing(self):
        """Generate soft professional chime sounds if not present."""
        import numpy as np

        sample_rate = 22050
        duration = 0.6  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))

        sounds = {
            'check-in.wav': np.sin(2 * np.pi * 600 * t) * np.exp(-t * 3),  # Rising tone
            'check-out.wav': np.sin(2 * np.pi * 400 * t) * np.exp(-t * 3),  # Falling tone
            'leave.wav': np.sin(2 * np.pi * 500 * t) * np.exp(-t * 4),     # Neutral tone
        }

        for fname, wave in sounds.items():
            path = os.path.join('sounds', fname)
            if not os.path.exists(path):
                try:
                    stereo_wave = np.column_stack([wave, wave])  # Stereo
                    stereo_wave = (stereo_wave * 32767).astype(np.int16)
                    pygame.mixer.Sound(buffer=stereo_wave).save(path)
                    logger.info(f"✅ Generated default sound: {path}")
                except Exception as e:
                    logger.warning(f"Could not generate {path}: {e}")
                    
    def apply_light_theme(self):
        self.root.configure(bg=self.bg_light)
        self.style.configure('TFrame', background=self.bg_light)
        self.style.configure('TLabel', background=self.bg_light, foreground=self.text_light)
        self.style.configure('TButton', font=('Inter', 11, 'bold'), padding=8, borderwidth=0)
        self.style.configure('Accent.TButton', foreground='white', background=self.accent)
        self.style.map('Accent.TButton', background=[('active', '#2563eb')])
        self.style.configure('Success.TButton', foreground='white', background=self.success)
        self.style.map('Success.TButton', background=[('active', '#16a34a')])
        self.style.configure('Warning.TButton', foreground='white', background=self.warning)
        self.style.map('Warning.TButton', background=[('active', '#d97706')])
        self.style.configure('Danger.TButton', foreground='white', background=self.danger)
        self.style.map('Danger.TButton', background=[('active', '#dc2626')])
        self.style.configure('Card.TFrame', background=self.card_light)
        self.style.configure('Card.TLabel', background=self.card_light, foreground=self.text_light)
        self.style.configure('Header.TLabel', font=self.title_font, foreground=self.accent, background=self.bg_light)
        self.style.configure('Treeview', background='#ffffff', fieldbackground='#ffffff', foreground=self.text_light, rowheight=32, font=('Inter', 10))
        self.style.configure('Treeview.Heading', font=('Inter', 11, 'bold'), background='#e5e7eb', foreground=self.text_light)
        self.style.map('Treeview', background=[('selected', '#3b82f6')], foreground=[('selected', 'white')])

    def apply_dark_theme(self):
        self.root.configure(bg=self.bg_dark)
        self.style.configure('TFrame', background=self.bg_dark)
        self.style.configure('TLabel', background=self.bg_dark, foreground=self.text_dark)
        self.style.configure('TButton', font=('Inter', 11, 'bold'), padding=8, borderwidth=0)
        self.style.configure('Accent.TButton', foreground='white', background=self.accent)
        self.style.map('Accent.TButton', background=[('active', '#2563eb')])
        self.style.configure('Success.TButton', foreground='white', background=self.success)
        self.style.map('Success.TButton', background=[('active', '#16a34a')])
        self.style.configure('Warning.TButton', foreground='white', background=self.warning)
        self.style.map('Warning.TButton', background=[('active', '#d97706')])
        self.style.configure('Danger.TButton', foreground='white', background=self.danger)
        self.style.map('Danger.TButton', background=[('active', '#dc2626')])
        self.style.configure('Card.TFrame', background=self.card_dark)
        self.style.configure('Card.TLabel', background=self.card_dark, foreground=self.text_dark)
        self.style.configure('Header.TLabel', font=self.title_font, foreground=self.accent, background=self.bg_dark)
        self.style.configure('Treeview', background='#2d2d2d', fieldbackground='#2d2d2d', foreground=self.text_dark, rowheight=32, font=('Inter', 10))
        self.style.configure('Treeview.Heading', font=('Inter', 11, 'bold'), background='#374151', foreground=self.text_dark)
        self.style.map('Treeview', background=[('selected', '#3b82f6')], foreground=[('selected', 'white')])

    def update_treeview_tags(self):
        if hasattr(self, 'detection_tree'):
            self.detection_tree.tag_configure('oddrow', background='#2d2d2d' if self.dark_mode else '#f9fafb')
            self.detection_tree.tag_configure('evenrow', background='#374151' if self.dark_mode else '#ffffff')

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
            self.theme_button.config(text="☀️ Light Mode")
        else:
            self.apply_light_theme()
            self.theme_button.config(text="🌙 Dark Mode")
        canvas_bg = self.card_dark if self.dark_mode else self.card_light
        canvas_fg = self.text_dark if self.dark_mode else self.text_light
        self.canvas.configure(bg=canvas_bg)
        self.canvas.itemconfig(self.canvas_loading, fill=canvas_fg)
        self.update_treeview_tags()
        for card in [self.cam_card, self.att_card, self.stat_card, self.action_card, self.feed_card, self.det_card]:
            card.configure(style='Card.TFrame')
            for child in card.winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(style='Card.TLabel')
        self.root.update()

    def create_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+1000+1000")
        bg = '#fef3c7' if not self.dark_mode else '#4b5563'
        fg = self.text_light if not self.dark_mode else self.text_dark
        label = tk.Label(tooltip, text=text, background=bg, foreground=fg, relief='solid', borderwidth=1, font=('Inter', 10), padx=8, pady=4)
        label.pack()
        def enter(event):
            x = widget.winfo_rootx() + 30
            y = widget.winfo_rooty() + 30
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()
        def leave(event):
            tooltip.withdraw()
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
        tooltip.withdraw()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="25")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=4)
        main_frame.rowconfigure(2, weight=2)

        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 25))
        ttk.Label(header_frame, text="✨ Face Recognition Attendance", style='Header.TLabel').pack(side="left")
        self.theme_button = ttk.Button(header_frame, text="🌙 Dark Mode", command=self.toggle_theme, style='Warning.TButton')
        self.theme_button.pack(side="right", padx=10)
        self.create_tooltip(self.theme_button, "Toggle between light and dark themes")

        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=1, column=0, rowspan=2, sticky="nswe", padx=(0, 20), pady=(0, 15))
        left_panel.columnconfigure(0, weight=1)

        self.cam_card = ttk.Frame(left_panel, style='Card.TFrame', padding=20)
        self.cam_card.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        ttk.Label(self.cam_card, text="📹 Camera Controls", font=('Inter', 12, 'bold'), style='Card.TLabel').pack(anchor="w", pady=(0, 15))
        self.start_button = ttk.Button(self.cam_card, text="▶️ Start Camera", command=self.start_detection, style='Success.TButton')
        self.start_button.pack(fill="x", pady=6)
        self.create_tooltip(self.start_button, "Start the camera and face detection")
        self.stop_button = ttk.Button(self.cam_card, text="⏹️ Stop Camera", command=self.stop_detection, state="disabled", style='Danger.TButton')
        self.stop_button.pack(fill="x", pady=6)
        self.create_tooltip(self.stop_button, "Stop the camera and face detection")

        self.att_card = ttk.Frame(left_panel, style='Card.TFrame', padding=20)
        self.att_card.grid(row=1, column=0, sticky="ew", pady=15)
        ttk.Label(self.att_card, text="📝 Attendance Type", font=('Inter', 12, 'bold'), style='Card.TLabel').pack(anchor="w", pady=(0, 15))
        self.type_combo = ttk.Combobox(self.att_card, textvariable=self.attendance_type_var, values=["check-in", "check-out", "leave"], state="readonly", font=('Inter', 11))
        self.type_combo.pack(fill="x", pady=6)
        self.type_combo.current(0)
        self.create_tooltip(self.type_combo, "Select the type of attendance to record")

        ttk.Label(self.att_card, text="🔊 Volume", font=('Inter', 12, 'bold'), style='Card.TLabel').pack(anchor="w", pady=(10, 5))
        volume_slider = ttk.Scale(self.att_card, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.volume_var, command=self.update_volume)
        volume_slider.pack(fill="x", pady=6)
        self.create_tooltip(volume_slider, "Adjust the volume for audio announcements")

        self.stat_card = ttk.Frame(left_panel, style='Card.TFrame', padding=20)
        self.stat_card.grid(row=2, column=0, sticky="ew", pady=15)
        ttk.Label(self.stat_card, text="📊 System Status", font=('Inter', 12, 'bold'), style='Card.TLabel').pack(anchor="w", pady=(0, 15))
        ttk.Label(self.stat_card, textvariable=self.staff_count_var, style='Card.TLabel').pack(anchor="w", pady=2)
        ttk.Label(self.stat_card, textvariable=self.fps_var, style='Card.TLabel').pack(anchor="w", pady=2)
        ttk.Label(self.stat_card, textvariable=self.detection_time_var, style='Card.TLabel').pack(anchor="w", pady=2)
        ttk.Label(self.stat_card, textvariable=self.status_var, style='Card.TLabel').pack(anchor="w", pady=(10, 0))

        self.action_card = ttk.Frame(left_panel, style='Card.TFrame', padding=20)
        self.action_card.grid(row=3, column=0, sticky="ew", pady=15)
        ttk.Label(self.action_card, text="⚙️ System Actions", font=('Inter', 12, 'bold'), style='Card.TLabel').pack(anchor="w", pady=(0, 15))
        refresh_button = ttk.Button(self.action_card, text="🔄 Refresh Staff", command=self.manual_refresh, style='Accent.TButton')
        refresh_button.pack(fill="x", pady=6)
        self.create_tooltip(refresh_button, "Refresh staff data from the server")
        clear_cache_button = ttk.Button(self.action_card, text="🗑️ Clear Cache", command=self.clear_cache, style='Warning.TButton')
        clear_cache_button.pack(fill="x", pady=6)
        self.create_tooltip(clear_cache_button, "Clear the cached face encodings")

        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, rowspan=2, sticky="nsew", pady=(0, 15))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=4)
        right_panel.rowconfigure(1, weight=1)

        self.feed_card = ttk.Frame(right_panel, style='Card.TFrame', padding=20)
        self.feed_card.grid(row=0, column=0, sticky="nsew", pady=(0, 15))
        self.feed_card.columnconfigure(0, weight=1)
        self.feed_card.rowconfigure(1, weight=1)
        ttk.Label(self.feed_card, text="📷 Live Camera Feed", font=('Inter', 12, 'bold'), style='Card.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 15))

        canvas_bg = self.card_light
        canvas_fg = self.text_light
        self.canvas = tk.Canvas(self.feed_card, bg=canvas_bg, highlightthickness=1, highlightbackground='#d1d5db')
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas_loading = self.canvas.create_text(
            400, 200, text="🎥 Initializing Camera...", font=('Inter', 14, 'bold'), fill=canvas_fg, anchor="center"
        )

        self.det_card = ttk.Frame(right_panel, style='Card.TFrame', padding=20)
        self.det_card.grid(row=1, column=0, sticky="nsew")
        self.det_card.columnconfigure(0, weight=1)
        self.det_card.rowconfigure(1, weight=1)
        ttk.Label(self.det_card, text="✅ Recent Detections", font=('Inter', 12, 'bold'), style='Card.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 15))

        columns = ('Name', 'Type', 'Time', 'Confidence')
        self.detection_tree = ttk.Treeview(self.det_card, columns=columns, show='headings', height=8)
        self.update_treeview_tags()
        for col in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, anchor="center", width=160 if col == 'Time' else 120)
        self.detection_tree.grid(row=1, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(self.det_card, orient="vertical", command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=1, column=1, sticky="ns")

    def update_volume(self, value):
        volume = float(value)
        self.volume_var.set(volume)
        pygame.mixer.music.set_volume(volume)
        self.tts_engine.setProperty('volume', volume)

    def manual_refresh(self):
        if self.detector.running:
            self.status_var.set("🔄 Refreshing staff data...")
            self.start_button.config(state="disabled")
            asyncio.run_coroutine_threadsafe(self.detector.refresh_staff_data(), self.async_loop)
            self.root.after(3000, lambda: self.start_button.config(state="normal"))
            self.root.after(3000, lambda: self.status_var.set("🟢 System Idle"))

    def clear_cache(self):
        self.detector.cache.cache.clear()
        self.detector.cache.save_cache()
        self.status_var.set("🗑️ Cache cleared successfully")
        self.root.after(2000, lambda: self.status_var.set("🟢 System Idle"))

    def process_sound_queue(self):
        """Background thread to play sounds professionally."""
        while self.sound_running:
            try:
                priority, (name, attendance_type, staff_id) = self.sound_queue.get(timeout=1.0)

                # Construct sound path: personal > generic
                personal_sound = os.path.join(self.config.sound_directory, f"staff_{staff_id}_{attendance_type}.wav")
                generic_sound = os.path.join(self.config.sound_directory, f"{attendance_type}.wav")

                sound_file = personal_sound if os.path.exists(personal_sound) else generic_sound

                if os.path.exists(sound_file):
                    try:
                        pygame.mixer.music.load(sound_file)
                        pygame.mixer.music.set_volume(self.volume_var.get())
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() and self.sound_running:
                            time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Failed to play {sound_file}: {e}")
                        self._play_tts_fallback(name, attendance_type)
                else:
                    logger.warning(f"Sound file not found: {sound_file}")
                    self._play_tts_fallback(name, attendance_type)

                self.sound_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Sound queue error: {e}")
                
    def _play_tts_fallback(self, name: str, attendance_type: str):
        """
        Professional voice announcements using Microsoft David.
        Dynamically handles all attendance types.
        """
        # Define TTS messages for known attendance types
        attendance_messages = {
            "check-in": f"Good morning, {name}. You're checked in. Have a productive day.",
            "check-out": f"Goodbye, {name}. You're checked out. Thank you for your work today.",
            "leave": f"{name}, your leave has been recorded. Take care and see you soon.",
            "break": f"{name}, your break has started. Enjoy your time.",
            "overtime": f"{name}, your overtime has been recorded. Keep up the great work."
        }

        # Get message or fallback
        message = attendance_messages.get(
            attendance_type,
            f"{name}, your attendance type '{attendance_type.replace('-', ' ')}' has been recorded."
        )

        try:
            # Log voice info and message
            current_voice = self.tts_engine.getProperty('voice')
            logger.info(f"🔊 TTS ({current_voice}): {message}")

            # Speak the message
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")

    def queue_sound_alert(self, name: str, attendance_type: str, staff_id: str):
        """Queue a sound alert with debounce and priority."""
        current_type = self.attendance_type_var.get()
        if current_type not in ["check-in", "check-out", "leave"]:
            return

        now = time.time()
        key = f"{staff_id}_{current_type}"

        # Debounce: prevent repeat within cooldown
        if now - self.last_sound_times.get(key, 0) < self.config.sound_debounce_interval:
            return

        try:
            # Priority: check-in > check-out > leave
            priority = {"check-in": 1, "check-out": 2, "leave": 3}.get(current_type, 2)

            if not self.sound_queue.full():
                self.sound_queue.put((priority, (name, current_type, staff_id)))
                self.last_sound_times[key] = now
                logger.info(f"🔔 Alert queued: {name} - {current_type}")
            else:
                logger.warning("Sound queue full. Alert dropped.")
        except Exception as e:
            logger.error(f"Error queuing sound: {e}")

    def start_detection(self):
        if not self.running:
            self.running = True
            self.sound_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_var.set("🎥 Camera starting...")
            self.canvas.delete(self.canvas_loading)

            os.makedirs(self.config.sound_directory, exist_ok=True)

            self.sound_thread = threading.Thread(target=self.process_sound_queue, daemon=True)
            self.sound_thread.start()

            self.async_loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self.run_async_loop, args=(self.async_loop,), daemon=True)
            self.thread.start()

    def stop_detection(self):
        if self.running:
            self.running = False
            self.sound_running = False
            self.detector.running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_var.set("⏹️ Stopping camera...")
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            cv2.destroyAllWindows()
            canvas_fg = self.text_dark if self.dark_mode else self.text_light
            self.canvas.delete("all")
            self.canvas.create_text(400, 200, text="🛑 Camera Stopped", font=('Inter', 14, 'bold'), fill=canvas_fg, anchor="center")
            self.fps_var.set("📊 FPS: 0")
            self.detection_time_var.set("⏱️ Detection: 0ms")
            self.status_var.set("🟢 System Idle")
            if self.sound_thread:
                self.sound_thread.join(timeout=2.0)
                pygame.mixer.quit()

    def run_async_loop(self, loop):
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.main_loop())
        except Exception as e:
            logger.error(f"Async loop error: {e}")
            self.root.after(0, lambda: self.status_var.set(f"❌ Error: {str(e)}"))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def main_loop(self):
        try:
            await self.detector.initialize()
            refresh_task = asyncio.create_task(self.detector.background_refresh())
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Cannot open webcam")
                self.root.after(0, lambda: self.status_var.set("🔴 Webcam not available"))
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.root.after(0, lambda: self.status_var.set("🟢 Camera running"))
            frame_count = 0
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                frame_count += 1
                self.detector.performance.update_process_frames()
                if frame_count % self.detector.performance.process_frames == 0:
                    att_type = self.attendance_type_var.get()
                    processed_frame, detections = await self.detector.detect_and_check_optimized(frame, att_type)
                    if detections:
                        for detection in detections:
                            self.root.after(0, lambda d=detection: self.queue_sound_alert(d['name'], d['type'], d['staff_id']))
                        self.root.after(0, lambda d=detections: self.update_detection_display(d))
                    self.root.after(0, lambda f=processed_frame: self.update_camera_display(f))
                self.detector.performance.add_frame_time((time.time() - self.last_frame_time) * 1000)
                self.last_frame_time = time.time()
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            self.root.after(0, lambda: self.status_var.set(f"❌ Error: {str(e)}"))
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            if 'refresh_task' in locals():
                refresh_task.cancel()
                try:
                    await refresh_task
                except asyncio.CancelledError:
                    pass
            self.root.after(0, lambda: self.status_var.set("🛑 Camera Stopped"))

    def update_camera_display(self, frame):
        try:
            if frame is None:
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            canvas_width = max(self.canvas.winfo_width(), 200)
            canvas_height = max(self.canvas.winfo_height(), 150)
            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height
            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
        except Exception as e:
            logger.error(f"Camera display error: {e}")

    def update_detection_display(self, staff_data):
        try:
            for i, entry in enumerate(staff_data):
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                self.detection_tree.insert('', 0, values=(
                    entry['name'],
                    entry['type'].title(),
                    entry['timestamp'],
                    entry['confidence']
                ), tags=(tag,))
            for child in self.detection_tree.get_children()[15:]:
                self.detection_tree.delete(child)
        except Exception as e:
            logger.error(f"Detection display error: {e}")

    def update_performance_display(self):
        try:
            if self.detector and self.detector.performance:
                fps = self.detector.performance.get_avg_fps()
                detect_time = self.detector.performance.get_avg_detection_time()
                self.fps_var.set(f"📊 FPS: {fps:.1f}")
                self.detection_time_var.set(f"⏱️ Detection: {detect_time:.0f}ms")
                with self.detector.lock:
                    count = len(self.detector.known_encodings)
                    self.staff_count_var.set(f"👥 Registered Staff: {count}")
            self.root.after(500, self.update_performance_display)
        except Exception as e:
            logger.error(f"Performance update error: {e}")
            self.root.after(500, self.update_performance_display)

    def on_closing(self):
        try:
            if self.running:
                self.stop_detection()
            if hasattr(self, 'detector'):
                self.detector.cache.save_cache()
            if hasattr(self, 'tts_engine'):
                self.tts_engine.stop()
            self.sound_running = False
            if self.sound_thread:
                self.sound_thread.join(timeout=2.0)
            if self.async_loop and self.async_loop.is_running():
                self.async_loop.call_soon_threadsafe(self.async_loop.stop)
                self.thread.join(timeout=2.0)
            pygame.mixer.quit()
            self.root.destroy()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            self.root.destroy()

def main():
    root = tk.Tk()
    app = ModernFaceRecognitionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        messagebox.showerror("Critical Error", f"System error: {str(e)}")

if __name__ == "__main__":
    main()