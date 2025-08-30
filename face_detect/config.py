from dataclasses import dataclass
from datetime import timedelta
import os
from dotenv import load_dotenv

load_dotenv()

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