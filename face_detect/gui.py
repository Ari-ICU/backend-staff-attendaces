import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont
from PIL import Image, ImageTk
import cv2
import asyncio
import queue
import threading
import time
import os
import pygame
import pyttsx3
import logging
from config import Config
from face_detector import FaceDetector
from scipy.io import wavfile
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModernFaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ Face Recognition Attendance System")
        self.root.geometry("1000x850")
        self.dark_mode = False
        self.latest_detections = {}  # Dictionary to store the most recent detection per staff_id

        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        os.makedirs('sounds', exist_ok=True)
        self._generate_default_sounds_if_missing()

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 100)
        self.tts_engine.setProperty('volume', 0.9)
        voices = self.tts_engine.getProperty('voices')
        for v in voices:
            if "Zira" in v.name or "Samantha" in v.name:
                self.tts_engine.setProperty('voice', v.id)
                logger.info(f"Using voice: {v.name}")
                break
        else:
            self.tts_engine.setProperty('voice', voices[0].id)
            logger.info(f"Fallback voice: {voices[0].name}")

        self.sound_queue = queue.PriorityQueue(maxsize=Config.max_sound_queue_size)
        self.sound_thread = None
        self.sound_running = False
        self.last_sound_times = {}
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
        sample_rate = 22050
        duration = 0.6
        t = np.linspace(0, duration, int(sample_rate * duration))
        sounds = {
            'check-in.wav': np.sin(2 * np.pi * 600 * t) * np.exp(-t * 3),
            'check-out.wav': np.sin(2 * np.pi * 400 * t) * np.exp(-t * 3),
        }
        for fname, wave in sounds.items():
            path = os.path.join('sounds', fname)
            if not os.path.exists(path):
                try:
                    stereo_wave = np.column_stack([wave, wave])
                    stereo_wave = (stereo_wave * 32767).astype(np.int16)
                    wavfile.write(path, sample_rate, stereo_wave)
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
        self.type_combo = ttk.Combobox(self.att_card, textvariable=self.attendance_type_var, values=["check-in", "check-out"], state="readonly", font=('Inter', 11))
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
            self.root.after(5000, lambda: self.start_button.config(state="normal"))
            self.root.after(5000, lambda: self.status_var.set("🟢 System Idle"))

    def clear_cache(self):
        self.detector.cache.cache.clear()
        self.detector.cache.save_cache()
        self.status_var.set("🗑️ Cache cleared successfully")
        self.root.after(2000, lambda: self.status_var.set("🟢 System Idle"))

    def process_sound_queue(self):
        while self.sound_running:
            try:
                priority, (name, attendance_type, staff_id) = self.sound_queue.get(timeout=1.0)
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
        attendance_messages = {
            "check-in": f"Good morning, {name}. You're checked in. Have a productive day.",
            "check-out": f"Goodbye, {name}. You're checked out. Thank you for your work today.",
        }
        message = attendance_messages.get(
            attendance_type,
            f"{name}, your attendance type '{attendance_type.replace('-', ' ')}' has been recorded."
        )
        try:
            current_voice = self.tts_engine.getProperty('voice')
            logger.info(f"🔊 TTS ({current_voice}): {message}")
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")

    def queue_sound_alert(self, name: str, attendance_type: str, staff_id: str):
        current_type = self.attendance_type_var.get()
        if current_type not in ["check-in", "check-out"]:
            return
        now = time.time()
        key = f"{staff_id}_{current_type}"
        if now - self.last_sound_times.get(key, 0) < self.config.sound_debounce_interval:
            return
        try:
            priority = {"check-in": 1, "check-out": 2}.get(current_type, 2)
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
            self.root.after(0, lambda error=e: self.status_var.set(f"❌ Error: {str(error)}"))
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
            self.root.after(0, lambda error=e: self.status_var.set(f"❌ Error: {str(error)}"))
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
            # Update the latest_detections dictionary with new detections
            for entry in staff_data:
                staff_id = entry.get('staff_id')
                if staff_id:
                    self.latest_detections[staff_id] = {
                        'name': entry.get('name', 'Unknown'),
                        'type': entry.get('type', '').title(),
                        'timestamp': entry.get('timestamp', ''),
                        'confidence': entry.get('confidence', '')
                    }

            # Clear the Treeview
            self.detection_tree.delete(*self.detection_tree.get_children())

            # Add the latest detection for each staff member
            for i, (staff_id, entry) in enumerate(self.latest_detections.items()):
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                self.detection_tree.insert('', 'end', values=(
                    entry['name'],
                    entry['type'],
                    entry['timestamp'],
                    entry['confidence']
                ), tags=(tag,))

            # Limit to the most recent 15 staff members
            items = self.detection_tree.get_children()
            for item in items[15:]:
                self.detection_tree.delete(item)

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