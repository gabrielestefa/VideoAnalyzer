import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime, timedelta
import numpy as np
import collections
import mediapipe as mp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import MultiCursor
import networkx as nx
from audio_wheel_analyzer import TOPIC_COLORS  # Shared color palette
import pandas as pd
from video_analysis_engine import VideoAnalysisEngine

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

GESTURE_MODEL_PATH = r"D:\personal\OpenCV\VideoHandTracker\Models\gesture_recognizer.task"

# Manual definition of HAND_CONNECTIONS since mp.solutions is not available
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
])

import math

class AudioWaveVisualizer(tk.Canvas):
    def __init__(self, master, width=200, height=40, bg_color="black"):
        super().__init__(master, width=width, height=height, bg=bg_color, highlightthickness=0)
        self.width = width
        self.height = height
        self.is_running = False
        self.phase = 0
        self.lines = []
        
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.animate()
            
    def stop(self):
        self.is_running = False
        self.delete("all")
        
    def animate(self):
        if not self.is_running:
            return
            
        self.delete("all")
        self.phase += 0.15
        
        # Spectrum Bar Style
        # Colors from user preference
        colors = ["#2E3674", "#5D8AA8", "#768993"]
        
        cy = self.height / 2
        num_bars = 28
        gap = 3
        
        # Calculate width dynamically
        total_gap = (num_bars - 1) * gap
        bar_width = (self.width - total_gap - 10) / num_bars 
        
        for i in range(num_bars):
            # Simulate generic audio activity using intersecting sine waves
            # x is the "frequency" index
            # Add phase shift to make it move
            val = math.sin(i * 0.3 + self.phase) * 0.5 + \
                  math.sin(i * 0.7 - self.phase * 2) * 0.3 + \
                  math.sin(i * 0.1 + self.phase * 0.5) * 0.2
            
            # Normalize to 0-1 somewhat, Val is approx -1 to 1
            h_norm = abs(val) 
            
            # Height scaling (keep some min height)
            h = max(4, h_norm * (self.height * 0.85))
            
            x_center = 5 + i * (bar_width + gap) + (bar_width / 2)
            
            # Cycle colors
            col = colors[i % len(colors)]
            
            # Draw vertical line with round caps to simulate the "pill" shape
            self.create_line(x_center, cy - h/2, x_center, cy + h/2, 
                           width=bar_width, capstyle=tk.ROUND, fill=col)
        
        self.after(30, self.animate)

class ModernHandTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hand Gesture Analytics | Modern UI")
        self.geometry("1400x900")
        
        # --- Top Right Controls ---
        # Load Icon
        try:
            lib_img = ctk.CTkImage(light_image=Image.open(r"D:\personal\OpenCV\VideoHandTracker\Media\Library.png"),
                                   dark_image=Image.open(r"D:\personal\OpenCV\VideoHandTracker\Media\Library.png"),
                                   size=(20, 20))
        except Exception as e:
            print(f"Icon warn: {e}")
            lib_img = None

        self.btn_save = ctk.CTkButton(self, text="Save Analysis", command=self.save_current_analysis, width=100, height=30)
        self.btn_save.place(relx=1.0, rely=0.01, x=-160, anchor="ne")

        self.btn_library = ctk.CTkButton(self, text="", image=lib_img, command=self.open_library, width=40, height=30)
        self.btn_library.place(relx=1.0, rely=0.01, x=-110, anchor="ne")
        
        # Engine
        self.engine = VideoAnalysisEngine(GESTURE_MODEL_PATH)
        self.is_analyzing = False
        self.rotation_angle = 0
        
        # UI Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.create_sidebar()
        self.create_main_view()
        
    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(20, weight=1)

        logo_label = ctk.CTkLabel(self.sidebar_frame, text="HandTracker\nAnalytics", 
                                font=ctk.CTkFont(size=20, weight="bold"))
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Actions
        self.upload_btn = ctk.CTkButton(self.sidebar_frame, text="Upload Video", command=self.upload_video)
        self.upload_btn.grid(row=1, column=0, padx=20, pady=10)
        
        # Video Analysis Button
        self.analyze_video_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Analyze Video", 
            command=self.start_video_analysis, 
            state="disabled",
            fg_color="#3498db",
            hover_color="#2980b9",
            text_color="#ffffff",
            border_width=2
        )
        self.analyze_video_btn.grid(row=2, column=0, padx=20, pady=5)
        
        # Audio Analysis Button
        self.analyze_audio_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Analyze Audio", 
            command=self.start_audio_analysis, 
            state="disabled",
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            text_color="#ffffff",
            border_width=2
        )
        self.analyze_audio_btn.grid(row=3, column=0, padx=20, pady=5)


        # Filters
        ctk.CTkLabel(self.sidebar_frame, text="Heatmap Settings", font=ctk.CTkFont(size=14, weight="bold")).grid(row=4, column=0, padx=20, pady=(20, 0))
        
        self.source_var = ctk.StringVar(value="Whole Hand")
        self.source_combo = ctk.CTkComboBox(self.sidebar_frame, values=["Whole Hand", "Wrist", "Thumb", "Index", "Median"],
                                            command=self.on_filter_change, variable=self.source_var)
        self.source_combo.grid(row=5, column=0, padx=20, pady=5)
        
        self.heat_type_var = ctk.StringVar(value="Presence")
        self.type_combo = ctk.CTkComboBox(self.sidebar_frame, values=["Presence", "Speed"], 
                                          command=self.on_filter_change, variable=self.heat_type_var)
        self.type_combo.grid(row=6, column=0, padx=20, pady=5)
        
        # Advanced Filters
        filter_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        filter_frame.grid(row=7, column=0, padx=20, pady=5)
        
        ctk.CTkLabel(filter_frame, text="Min Trace Len:").pack(anchor="w")
        self.min_len_slider = ctk.CTkSlider(filter_frame, from_=0, to=50, number_of_steps=50, command=self.on_filter_change)
        self.min_len_slider.set(5)
        self.min_len_slider.pack(fill="x")
        
        ctk.CTkLabel(filter_frame, text="Max Jump %:").pack(anchor="w")
        self.max_jump_slider = ctk.CTkSlider(filter_frame, from_=1, to=50, number_of_steps=49, command=self.on_filter_change)
        self.max_jump_slider.set(10)
        self.max_jump_slider.pack(fill="x")

        # Opacity
        ctk.CTkLabel(self.sidebar_frame, text="Opacity").grid(row=8, column=0, padx=20, pady=(10, 0))
        self.opacity_slider = ctk.CTkSlider(self.sidebar_frame, from_=0, to=1, command=self.update_view)
        self.opacity_slider.set(0.5)
        self.opacity_slider.grid(row=9, column=0, padx=20, pady=5)
        
        # Sensitivity
        ctk.CTkLabel(self.sidebar_frame, text="Sensitivity").grid(row=10, column=0, padx=20, pady=(10, 0))
        self.sensitivity_slider = ctk.CTkSlider(self.sidebar_frame, from_=0, to=0.5, command=self.update_view)
        self.sensitivity_slider.set(0.1)
        self.sensitivity_slider.grid(row=11, column=0, padx=20, pady=5)

        # Toggles
        self.check_heatmap = ctk.CTkCheckBox(self.sidebar_frame, text="Show Heatmap", command=self.update_view)
        self.check_heatmap.select()
        self.check_heatmap.grid(row=12, column=0, padx=20, pady=5, sticky="n")
        
        self.check_trace = ctk.CTkCheckBox(self.sidebar_frame, text="Show Trace", command=self.update_view)
        self.check_trace.grid(row=13, column=0, padx=20, pady=5, sticky="n")

        self.check_skeleton = ctk.CTkCheckBox(self.sidebar_frame, text="Show Skeleton", command=self.update_view)
        self.check_skeleton.select()
        self.check_skeleton.grid(row=14, column=0, padx=20, pady=5, sticky="n")

        # Audio
        self.check_audio = ctk.CTkCheckBox(self.sidebar_frame, text="Enable Audio Transcribe")
        self.check_audio.grid(row=15, column=0, padx=20, pady=5, sticky="n")

        # Rotate
        self.rotate_btn = ctk.CTkButton(self.sidebar_frame, text="Rotate 90°", command=self.rotate_view, fg_color="#555")
        self.rotate_btn.grid(row=16, column=0, padx=20, pady=10)

        # Status
        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame)
        self.progress_bar.grid(row=17, column=0, padx=20, pady=10)
        self.progress_bar.set(0)
        
        # Audio Wave Visualization (Hidden by default)
        # Match background to sidebar frame color
        bg_color = self.sidebar_frame._apply_appearance_mode(self.sidebar_frame._fg_color)
        self.audio_wave = AudioWaveVisualizer(self.sidebar_frame, width=160, height=30, bg_color=bg_color)
        self.audio_wave.grid(row=18, column=0, padx=20, pady=10)
        self.audio_wave.grid_remove() # Hide initially
        
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Ready", text_color="gray")
        self.status_label.grid(row=19, column=0, padx=20, pady=10)

    def create_main_view(self):
        # Tabs for Video / Dashboard
        self.tab_view = ctk.CTkTabview(self, width=1100)
        self.tab_view.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.tab_video = self.tab_view.add("Video Analysis")
        self.tab_dash = self.tab_view.add("Dashboard")
        
        # --- Video Tab ---
        self.tab_video.grid_columnconfigure(0, weight=1)
        self.tab_video.grid_rowconfigure(0, weight=1)
        
        # Canvas Container
        self.canvas_frame = ctk.CTkFrame(self.tab_video, fg_color="black")
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Actual Canvas
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Controls below canvas
        self.controls_frame = ctk.CTkFrame(self.tab_video, height=50)
        self.controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        self.play_slider = ctk.CTkSlider(self.controls_frame, from_=0, to=100, command=self.on_slider_move)
        self.play_slider.set(0)
        self.play_slider.pack(fill=tk.X, padx=20, pady=15)
        
        # --- Dashboard Tab Container ---
        self.dash_tabs = ctk.CTkTabview(self.tab_dash)
        self.dash_tabs.pack(fill=tk.BOTH, expand=True)
        self.dash_tab_gen = self.dash_tabs.add("General Stats")
        self.dash_tab_hmi = self.dash_tabs.add("HMI Analytics")
        self.dash_tab_timeline = self.dash_tabs.add("Timeline")
        self.dash_tab_3d = self.dash_tabs.add("3D Visualization")
        self.dash_tab_audio = self.dash_tabs.add("Audio Log")
        
        # Dashboard Configs
        self.viz_3d_var = ctk.StringVar(value="Trace Lines")

        # Bring top-right buttons to front
        self.btn_save.lift()
        self.btn_library.lift()
        
    def upload_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            if self.engine.load_video(path):
                self.rotation_angle = 0
                self.status_label.configure(text="Video Loaded")
                # Enable both analysis buttons
                self.analyze_video_btn.configure(state="normal")
                self.analyze_audio_btn.configure(state="normal")
                self.play_slider.configure(to=self.engine.total_frames-1)
                self.show_frame(0)
            else:
                messagebox.showerror("Error", "Could not load video")

    def start_video_analysis(self):
        """Analyze video only (hand tracking, heatmap, etc.) WITHOUT audio"""
        self.analyze_video_btn.configure(state="disabled")
        self.analyze_audio_btn.configure(state="disabled")
        self.status_label.configure(text="Processing Video...")
        self.is_analyzing = True
        
        thread = threading.Thread(target=self.run_video_analysis, daemon=True)
        thread.start()
    
    def start_audio_analysis(self):
        """Analyze ONLY audio (transcription, sentiment) - skips video processing"""
        self.analyze_video_btn.configure(state="disabled")
        self.analyze_audio_btn.configure(state="disabled")
        self.status_label.configure(text="Processing Audio...")
        self.is_analyzing = True
        
        # UI Toggle: Wave ON, Progress OFF
        self.progress_bar.grid_remove()
        self.audio_wave.grid()
        self.audio_wave.start()
        
        thread = threading.Thread(target=self.run_audio_analysis, daemon=True)
        thread.start()

    def run_video_analysis(self):
        """Process video WITHOUT audio"""
        try:
            def on_progress(percent, msg):
                self.after(0, lambda: self.update_progress(percent, msg))
            
            # Video only - no audio
            self.engine.process_video(progress_callback=on_progress, enable_audio=False)
            self.after(0, self.on_analysis_complete)
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.after(0, lambda: self.analyze_video_btn.configure(state="normal"))
            self.after(0, lambda: self.analyze_audio_btn.configure(state="normal"))
    
    def run_audio_analysis(self):
        """Process ONLY audio - skip video hand tracking entirely"""
        try:
            def on_progress(percent, msg):
                self.after(0, lambda p=percent, m=msg: self.update_progress(p, m))
            
            # Check if engine has audio processing capability
            if not hasattr(self.engine, 'transcribe_audio'):
                raise Exception("Audio transcription not available in this engine version")
            
            if not self.engine.video_path:
                raise Exception("No video loaded")
            
            on_progress(0, "Starting audio analysis...")
            
            # Call the engine's transcribe method with progress callback
            self.engine.transcribe_audio(progress_callback=on_progress)
            
            on_progress(100, "Audio analysis complete!")
            
            self.after(0, self.on_audio_analysis_complete)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("Audio Analysis Error", str(e)))
            self.after(0, lambda: self.analyze_video_btn.configure(state="normal"))
            self.after(0, lambda: self.analyze_audio_btn.configure(state="normal"))

    def run_analysis(self, enable_audio=None):
        """Legacy method - kept for compatibility"""
        try:
            def on_progress(percent, msg):
                self.after(0, lambda: self.update_progress(percent, msg))
            
            use_audio = enable_audio if enable_audio is not None else (self.check_audio.get() == 1)
            self.engine.process_video(progress_callback=on_progress, enable_audio=use_audio)
            self.after(0, self.on_analysis_complete)
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.after(0, lambda: self.analyze_video_btn.configure(state="normal"))
            self.after(0, lambda: self.analyze_audio_btn.configure(state="normal"))

    def update_progress(self, percent, msg):
        self.progress_bar.set(percent / 100.0)
        self.status_label.configure(text=msg)
        self.update_idletasks()  # Force UI refresh

    def on_video_analysis_complete(self):
        """Called after video analysis completes"""
        self.progress_bar.set(0)
        self.status_label.configure(text="Analysis Complete")
        self.analyze_video_btn.configure(state="normal")
        self.analyze_audio_btn.configure(state="normal")
        self.engine.regenerate_data(self.source_var.get())
        self.update_view()
        self.generate_dashboard()
    
    def on_audio_analysis_complete(self):
        """Called after audio-only analysis completes"""
        # UI Toggle: Wave OFF, Progress ON
        self.audio_wave.stop()
        self.audio_wave.grid_remove()
        self.progress_bar.grid()
        self.progress_bar.set(0)
        
        self.status_label.configure(text="Audio Analysis Complete")
        self.analyze_video_btn.configure(state="normal")
        self.analyze_audio_btn.configure(state="normal")
        
        # Regenerate dashboard to show transcript data
        try:
            self.generate_dashboard()
            print("✓ Dashboard refreshed with audio transcript")
        except Exception as e:
            print(f"Warning: Could not regenerate dashboard: {e}")
    
    def populate_audio_log(self):
        """Refresh the audio log/transcript display after transcription completes"""
        # Find the OCR scroll container and refresh it
        if not hasattr(self, 'engine') or not self.engine:
            return
            
        # The audio tab content is regenerated in the dashboard
        # For now, just trigger a dashboard refresh if possible
        if hasattr(self, 'tabview') and self.tabview:
            try:
                # Trigger tab switch to force refresh
                current = self.tabview.get()
                self.tabview.set("Transcript")
                self.update_idletasks()
            except:
                pass  # Tab may not exist


    def rotate_view(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.update_view()

    def update_view(self, _=None):
        idx = int(self.play_slider.get())
        self.show_frame(idx)

    def rotate_pt(self, pt, angle, w, h):
         x, y = pt
         if angle == 0: return x, y
         elif angle == 90: return h-y, x
         elif angle == 180: return w-x, h-y
         elif angle == 270: return y, w-x
         return x, y

    def show_frame(self, idx):
        if not self.engine.video_path: return
        
        frame = None
        # Get Frame
        if idx < len(self.engine.frames):
            frame = self.engine.frames[idx].copy()
        else:
            # Fallback: Read from video file directly for preview
            try:
                cap = cv2.VideoCapture(self.engine.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, f = cap.read()
                cap.release()
                if ret:
                    frame = f
            except:
                pass
        
        if frame is None: return

        # Rotation
        if self.rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Draw Overlays
        # 1. Heatmap
        if self.check_heatmap.get() == 1 and self.engine.heatmap is not None:
            hmap = self.engine.heatmap
            if self.heat_type_var.get() == "Speed":
                hmap = self.engine.speed_heatmap
                
            if np.max(hmap) > 0:
                hmap_norm = hmap / np.max(hmap)
                # Use jet_r (reversed)
                hmap_color = (plt.cm.jet_r(hmap_norm)[:, :, :3] * 255).astype(np.uint8)
                
                alpha = self.opacity_slider.get()
                threshold = self.sensitivity_slider.get()
                mask = hmap_norm > threshold
                
                # Check rotation for heatmap match
                if self.rotation_angle == 90:
                    hmap_color = cv2.rotate(hmap_color, cv2.ROTATE_90_CLOCKWISE)
                    mask = cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool)
                elif self.rotation_angle == 180:
                    hmap_color = cv2.rotate(hmap_color, cv2.ROTATE_180)
                    mask = cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_180).astype(bool)
                elif self.rotation_angle == 270:
                    hmap_color = cv2.rotate(hmap_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    mask = cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool)

                # Resize if minor pixel mismatches
                if hmap_color.shape[:2] != frame.shape[:2]:
                    hmap_color = cv2.resize(hmap_color, (frame.shape[1], frame.shape[0]))
                    mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0])).astype(bool)

                overlay = frame.copy()
                overlay[mask] = cv2.addWeighted(frame[mask], 1-alpha, hmap_color[mask], alpha, 0)
                frame = overlay

        # 2. Skeleton (Full Hand)
        if self.check_skeleton.get() == 1:
            if idx < len(self.engine.landmarks_list):
                 for landmarks, label in self.engine.landmarks_list[idx]:
                     points = []
                     for lm in landmarks:
                         x, y = int(lm.x * self.engine.video_width), int(lm.y * self.engine.video_height)
                         points.append((x, y))
                     
                     r_points = [self.rotate_pt(p, self.rotation_angle, self.engine.video_width, self.engine.video_height) for p in points]

                     connections = HAND_CONNECTIONS
                     for start_idx, end_idx in connections:
                         cv2.line(frame, r_points[start_idx], r_points[end_idx], (0, 255, 0), 2)
                     for p in r_points:
                         cv2.circle(frame, p, 3, (0, 0, 255), -1)

        # 3. Traces
        if self.check_trace.get() == 1:
            for side, path in [('L', self.engine.trace_path_left), ('R', self.engine.trace_path_right)]:
                color = (255, 0, 0) if side == 'R' else (0, 0, 255) # RGB
                start = max(0, idx - 50)
                for i in range(start, idx):
                     if i < len(path) and path[i] and path[i+1]:
                         p1 = (int(path[i][0]), int(path[i][1]))
                         p2 = (int(path[i+1][0]), int(path[i+1][1]))
                         
                         p1 = self.rotate_pt(p1, self.rotation_angle, self.engine.video_width, self.engine.video_height)
                         p2 = self.rotate_pt(p2, self.rotation_angle, self.engine.video_width, self.engine.video_height)
                         
                         cv2.line(frame, p1, p2, color, 2)
        
        # Display on Canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: cw, ch = 800, 600
        
        h, w = frame.shape[:2]
        ratio = min(cw/w, ch/h)
        nw, nh = int(w*ratio), int(h*ratio)
        
        resized = cv2.resize(frame, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=imgtk, anchor=tk.CENTER)
        self.canvas.image = imgtk

    def on_slider_move(self, val):
        self.show_frame(int(val))

    def on_filter_change(self, _):
        if self.engine.frames:
            self.engine.regenerate_data(
                source=self.source_var.get(),
                min_len=int(self.min_len_slider.get()),
                max_jump_pct=self.max_jump_slider.get()/100.0
            )
            self.update_view()
            self.generate_dashboard()

    def generate_dashboard(self):
        # Clear all tabs
        for tab in [self.dash_tab_gen, self.dash_tab_hmi, self.dash_tab_timeline, self.dash_tab_3d, self.dash_tab_audio]:
            for w in tab.winfo_children(): w.destroy()
            
        # Determine strict audio-only mode
        has_hands = self.engine.has_hands
        
        # Helper placeholders
        def show_placeholder(parent, text):
            ctk.CTkLabel(parent, text=text, font=("Arial", 16, "bold"), text_color="gray").pack(pady=50)
            
        if not has_hands:
            show_placeholder(self.dash_tab_gen, "No hand data available.\n(Audio analysis only)")
            show_placeholder(self.dash_tab_hmi, "No hand data available.\n(Audio analysis only)")
            show_placeholder(self.dash_tab_timeline, "No hand data for timeline.\n(Audio analysis only)")
            show_placeholder(self.dash_tab_3d, "No 3D data.\n(Audio analysis only)")
            
            # Switch to Transcript tab automatically
            try:
                self.tabview.set("Transcript")
            except: pass
            
        # Calculate metrics irrespective (will return defaults if no hands)
        metrics = self.engine.calculate_metrics()
        hmi_left = self.engine.calculate_hmi_metrics('left')
        hmi_right = self.engine.calculate_hmi_metrics('right')

        # --- Tab 1: General (Skip if no hands) ---
        if has_hands:
            # 1. Heatmap
            fig1 = plt.Figure(figsize=(10, 6), dpi=100)
            ax1 = fig1.add_subplot(221)
            # Check source for heatmap
            hmap_to_show = self.engine.heatmap
            if self.heat_type_var.get() == "Speed":
                hmap_to_show = self.engine.speed_heatmap if self.engine.speed_heatmap is not None else self.engine.heatmap
                
            ax1.imshow(hmap_to_show, cmap="hot")
            ax1.set_title(f"Heatmap ({self.heat_type_var.get()})")
            ax1.axis("off")
            
            # 2. Velocity
            ax2 = fig1.add_subplot(222)
            ax2.plot(metrics['left']['speeds'], label='L', color='magenta')
            ax2.plot(metrics['right']['speeds'], label='R', color='cyan')
            ax2.set_title("Velocity Configuration")
            ax2.legend()
            
            # 3. Trajectory
            ax3 = fig1.add_subplot(223)
            ax3.set_title("Trajectory Map")
            if self.engine.trace_path_left:
                 valid_l = [p for p in self.engine.trace_path_left if p]
                 if valid_l: ax3.plot([p[0] for p in valid_l], [p[1] for p in valid_l], '.', color='blue', markersize=1)
            if self.engine.trace_path_right:
                 valid_r = [p for p in self.engine.trace_path_right if p]
                 if valid_r: ax3.plot([p[0] for p in valid_r], [p[1] for p in valid_r], '.', color='red', markersize=1)
            ax3.set_xlim(0, self.engine.video_width)
            ax3.set_ylim(self.engine.video_height, 0)
    
            # 4. Gesture Bar
            ax4 = fig1.add_subplot(224)
            gestures = list(self.engine.gesture_counts.keys())
            counts = list(self.engine.gesture_counts.values())
            if gestures:
                ax4.bar(gestures, counts, color=['blue' if 'Left' in g else 'red' for g in gestures])
                ax4.tick_params(axis='x', rotation=45, labelsize=8)
                ax4.set_title("Detected Gestures")
            else:
                ax4.text(0.5, 0.5, "No Gestures", ha='center')
                
            fig1.tight_layout()
            canvas1 = FigureCanvasTkAgg(fig1, master=self.dash_tab_gen)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill="both", expand=True)

            # --- Tab 2: HMI ---
            fig2 = plt.Figure(figsize=(10, 6), dpi=100)
            h_ax1 = fig2.add_subplot(221) # Pie
            h_ax2 = fig2.add_subplot(222) # Efficiency
            h_ax3 = fig2.add_subplot(223) # Hover
            
            # Pie
            total_counts = collections.Counter()
            if hmi_left: total_counts.update(hmi_left['counts'])
            if hmi_right: total_counts.update(hmi_right['counts'])
            sizes = [total_counts[0], total_counts[1], total_counts[2]]
            if sum(sizes) > 0:
                h_ax1.pie(sizes, labels=['Rest', 'Reach', 'Hover'], autopct='%1.1f%%', colors=['gray', 'green', 'orange'])
                h_ax1.set_title("Phase Distribution")
    
            # Efficiency
            eff_vals, eff_labels = [], []
            if hmi_left: 
                eff_vals.append(hmi_left['efficiency'])
                eff_labels.append("Left")
            if hmi_right:
                eff_vals.append(hmi_right['efficiency'])
                eff_labels.append("Right")
            if eff_vals:
                h_ax2.bar(eff_labels, eff_vals, color=['blue', 'red'])
                h_ax2.set_title("Path Efficiency (Tortuosity)")
                h_ax2.axhline(y=1.0, linestyle='--', color='k')
                
            # Hover
            h_ax3.set_title("Hover Hotspots")
            h_ax3.set_xlim(0, self.engine.video_width)
            h_ax3.set_ylim(self.engine.video_height, 0)
            if hmi_left and hmi_left['hover_points']:
                pts = np.array(hmi_left['hover_points'])
                h_ax3.scatter(pts[:,0], pts[:,1], c='orange', alpha=0.3, s=10)
            
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=self.dash_tab_hmi)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)
        
        # Add Legend & Timeline (Shared Figure)
        frame_t_ctrl = ctk.CTkFrame(self.dash_tab_timeline, height=30)
        frame_t_ctrl.pack(side=tk.TOP, fill=tk.X)
        
        if has_hands:
            ctk.CTkLabel(frame_t_ctrl, text="Legend: Gray=Rest, Green=Reach, Orange=Hover").pack(side=tk.LEFT, padx=10)
            self.ctx_label = ctk.CTkLabel(frame_t_ctrl, text="Hover over timeline for details...", text_color="yellow")
            self.ctx_label.pack(side=tk.RIGHT, padx=10)
        
        fig3 = plt.Figure(figsize=(10, 6), dpi=100)
        # Adjust layout based on data availability
        if has_hands:
            gs = fig3.add_gridspec(3, 1, height_ratios=[1.5, 1, 0.5], hspace=0.3)
            ax_speed = fig3.add_subplot(gs[0])
            ax_phase = fig3.add_subplot(gs[1], sharex=ax_speed)
            ax_sent = fig3.add_subplot(gs[2], sharex=ax_speed)
            
            # FPS calculation
            fps = 30
            if self.engine.video_path:
                 cap = cv2.VideoCapture(self.engine.video_path)
                 if cap.isOpened():
                     fps = cap.get(cv2.CAP_PROP_FPS) or 30
                 cap.release()
                 
            # 1. Speed Plot
            speeds_l = metrics['left']['speeds']
            speeds_r = metrics['right']['speeds']
            times_l = [i/fps for i in range(len(speeds_l))]
            times_r = [i/fps for i in range(len(speeds_r))]
            
            ax_speed.plot(times_l, speeds_l, color='magenta', label='Left Speed', linewidth=1)
            ax_speed.plot(times_r, speeds_r, color='cyan', label='Right Speed', linewidth=1)
            ax_speed.set_ylabel("Speed")
            ax_speed.legend(loc='upper right')
            ax_speed.grid(True, alpha=0.3)
            ax_speed.set_title("Timeline: Speed, Phases, and Sentiment")
        else:
            # Audio only timeline
            gs = fig3.add_gridspec(1, 1)
            ax_sent = fig3.add_subplot(gs[0])
            ax_sent.set_title("Audio Sentiment Timeline")
            # Create dummy speed/phase axes to prevent errors if referenced later
            ax_speed = None
            ax_phase = None
        
        # 2. Phases & Activity Bars (Middle)
        
        # Helper for Gantt
        def plot_gantt(ax, phases, y_pos, label, is_phase=True):
            if not phases: return
            start_frame = 0
            curr_phase = phases[0]
            xranges = []
            colors = []
            
            # Colors
            # Phase: 0=Rest(Gray), 1=Reach(Green), 2=Hover(Orange)
            phase_colors = {0: 'silver', 1: '#77dd77', 2: '#ffb347'}
            
            for i, p in enumerate(phases):
                if p != curr_phase:
                     # End of segment
                     duration = (i - start_frame) / fps
                     start_sec = start_frame / fps
                     xranges.append((start_sec, duration))
                     colors.append(phase_colors.get(curr_phase, 'white'))
                     
                     curr_phase = p
                     start_frame = i
            # Add last segment
            duration = (len(phases) - start_frame) / fps
            start_sec = start_frame / fps
            xranges.append((start_sec, duration))
            colors.append(phase_colors.get(curr_phase, 'white'))
            
            if xranges:
                ax.broken_barh(xranges, (y_pos, 0.8), facecolors=colors) # Height=0.8
            
        def plot_active(ax, path, y_pos, color):
            xranges = []
            if not path: return
            
            # Find active segments (where trace exists)
            start_f = -1
            for i, pt in enumerate(path):
                if pt and start_f == -1: start_f = i
                elif not pt and start_f != -1:
                    dur_sec = (i - start_f) / fps
                    start_sec = start_f / fps
                    xranges.append((start_sec, dur_sec))
                    start_f = -1
            if start_f != -1: # Unfinished
                 xranges.append((start_f/fps, (len(path)-start_f)/fps))
                 
            if xranges:
                ax.broken_barh(xranges, (y_pos, 0.8), facecolors=color)

        # Plot Phases (Only if hands exist)
        if has_hands and ax_phase is not None:
            if hmi_left: plot_gantt(ax_phase, hmi_left['phases'], 3, "L Phase")
            if hmi_right: plot_gantt(ax_phase, hmi_right['phases'], 2, "R Phase")
            
            plot_active(ax_phase, self.engine.trace_path_left, 1, '#6a5acd') # SlateBlue
            plot_active(ax_phase, self.engine.trace_path_right, 0, '#ff6961') # Pastel Red
            
            ax_phase.set_yticks([0.4, 1.4, 2.4, 3.4])
            ax_phase.set_yticklabels(["R Act", "L Act", "R Phs", "L Phs"])
            ax_phase.set_ylim(0, 4.5)
            ax_phase.grid(True, axis='x', alpha=0.3)
        
        # 3. Sentiment (Bottom) - Runs for both modes
        if hasattr(self.engine, 'transcript') and self.engine.transcript:
            xranges_pos = []
            xranges_neg = []
            xranges_neu = []
            
            for item in self.engine.transcript:
                if len(item) == 4:
                    start, end, text, cat_str = item
                else:
                    start, end, text = item
                    cat_str = "Neutral"
                
                cat_lower = cat_str.lower()
                dur = end - start
                
                if "positive" in cat_lower:
                    xranges_pos.append((start, dur))
                elif "negative" in cat_lower:
                    xranges_neg.append((start, dur))
                else:
                    xranges_neu.append((start, dur))
            
            # Plot
            if xranges_pos: ax_sent.broken_barh(xranges_pos, (0, 1), facecolors='#77dd77', label='Positive') # Green
            if xranges_neg: ax_sent.broken_barh(xranges_neg, (0, 1), facecolors='#ff6961', label='Negative') # Red
            if xranges_neu: ax_sent.broken_barh(xranges_neu, (0, 1), facecolors='lightgray', label='Neutral')
            
        ax_sent.set_yticks([]) # No Y labels needed
        ax_sent.set_ylabel("Sentiment")
        ax_sent.set_xlabel("Time (seconds)")
        ax_sent.set_ylim(0, 1)
        ax_sent.grid(True, axis='x', alpha=0.3)
        ax_sent.legend(loc='upper right', fontsize='small')

        # Legend (Only for phases if they exist)
        if ax_phase is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='silver', label='Rest/Neutral'),
                Patch(facecolor='#77dd77', label='Reach/Positive'),
                Patch(facecolor='#ff6961', label='Neg/Active'),
                Patch(facecolor='#ffb347', label='Hover'),
                Patch(facecolor='#6a5acd', label='Blue Hand')
            ]
            ax_phase.legend(handles=legend_elements, loc='upper right', fontsize='x-small', ncol=5)

        fig3.tight_layout()
        canvas3 = FigureCanvasTkAgg(fig3, master=self.dash_tab_timeline)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill="both", expand=True)
        
        # Cursor "Lens"
        axes_to_cursor = [ax for ax in [ax_speed, ax_phase, ax_sent] if ax is not None]
        if axes_to_cursor:
            self.cursor = MultiCursor(canvas3.figure.canvas, tuple(axes_to_cursor), color='r', lw=1, horizOn=False, vertOn=True)

        # INTERACTIVE HOVER
        def on_timeline_hover(event):
            if event.inaxes in axes_to_cursor:
                # event.xdata is time in seconds
                t = event.xdata
                if t is not None and t >= 0:
                    ctx = self.engine.get_context_at(t)
                    if ctx:
                        gesture = f"L:{ctx['gesture_left']} R:{ctx['gesture_right']}"
                        # Try to get sentiment from transcript context? 
                        # get_context_at gives "spoken" text, but not category explicitly.
                        # We can just show the spoken text which is already there.
                        # If we want the category, we'd need to update get_context_at or just parse it here if needed,
                        # but parsing here is hard. 
                        # Ideally get_context_at returns category too.
                        
                        spoken = f"Audio: {ctx['spoken']}" if ctx['spoken'] else ""
                        self.ctx_label.configure(text=f"Time: {t:.1f}s | {gesture} | {spoken}")
        
        canvas3.mpl_connect("motion_notify_event", on_timeline_hover)
        
        # --- Tab 4: 3D ---
        ctrl_frame = ctk.CTkFrame(self.dash_tab_3d)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        
        ctk.CTkLabel(ctrl_frame, text="Visualization Type:").pack(side=tk.LEFT, padx=5)
        self.viz_3d_combo = ctk.CTkComboBox(ctrl_frame, values=["Trace Lines", "Point Cloud", "Density Heatmap"], 
                                            command=lambda x: self.generate_dashboard(), # triggers redraw
                                            variable=self.viz_3d_var)
        self.viz_3d_combo.pack(side=tk.LEFT, padx=5)
        
        fig4 = plt.Figure(figsize=(10, 6), dpi=100)
        ax_3d = fig4.add_subplot(111, projection='3d')
        style = self.viz_3d_var.get()
        show_speed_3d = (self.heat_type_var.get() == "Speed")
        
        # Colors / CMP
        cmap = plt.get_cmap('jet') # Blue to Red
        
        # Max Speed for normalization
        all_speeds = metrics['left']['speeds'] + metrics['right']['speeds']
        max_speed = max(all_speeds) if all_speeds else 1
        
        def plot_3d_trace(ax, trace, speeds, hand_color):
            valid_indices = [i for i, p in enumerate(trace) if p is not None]
            if not valid_indices: return
            
            xs = [trace[i][0] for i in valid_indices]
            ys = [trace[i][1] for i in valid_indices]
            zs = [trace[i][2] for i in valid_indices]
            
            colors = hand_color
            if show_speed_3d and speeds:
                # Map Speed to Color
                # Speed array might be slightly shorter due to diff?
                # Usually len(speeds) = len(points) - 1. Or smoothed same len.
                # Let's handle index safely
                c_vals = []
                for i in valid_indices:
                     s = speeds[i] if i < len(speeds) else 0
                     c_vals.append(s / max_speed)
                colors = cmap(c_vals)

            if style == "Trace Lines":
                if show_speed_3d:
                     # For lines with varying color, we prefer scatter or segment Loop
                     # Segment loop is slow. Scatter is acceptable fallback or dense points.
                     # Let's use dense scatter with small s for 'Lines' look if speed on
                     ax.scatter(xs, ys, zs, c=colors, s=1)
                else:
                     ax.plot(xs, ys, zs, c=hand_color, alpha=0.5, linewidth=1)
            elif style == "Point Cloud":
                ax.scatter(xs, ys, zs, c=colors, alpha=0.3, s=2)

        if style in ["Trace Lines", "Point Cloud"]:
             if self.engine.trace_path_left:
                  plot_3d_trace(ax_3d, self.engine.trace_path_left, metrics['left']['speeds'], 'blue')
             if self.engine.trace_path_right:
                  plot_3d_trace(ax_3d, self.engine.trace_path_right, metrics['right']['speeds'], 'red')

        elif style == "Density Heatmap":
             # Voxel / Extrusion Logic
             bins = 50
             hmap_source = self.engine.heatmap
             current_cmap = 'hot'
             
             if show_speed_3d and self.engine.speed_heatmap is not None:
                 hmap_source = self.engine.speed_heatmap
                 current_cmap = 'jet'
                 
             x_grid = np.linspace(0, self.engine.video_width - 1, bins)
             y_grid = np.linspace(0, self.engine.video_height - 1, bins)
             z_steps = 20
             z_range = np.linspace(0, 100, z_steps)
             
             max_heat = np.max(hmap_source) if np.max(hmap_source) > 0 else 1
             grid_points, colors_list = [], []
             
             # Sample
             for gx in x_grid:
                 for gy in y_grid:
                     cx = int(np.clip(gx, 0, self.engine.video_width - 1))
                     cy = int(np.clip(gy, 0, self.engine.video_height - 1))
                     heat_val = hmap_source[cy, cx] / max_heat
                     
                     if heat_val > 0.02:
                         max_z_idx = int(heat_val * (z_steps - 1))
                         for z_idx in range(max_z_idx + 1):
                             grid_points.append((gx, gy, z_range[z_idx]))
                             colors_list.append(heat_val)
             
             if grid_points:
                 px, py, pz = zip(*grid_points)
                 # Map colors
                 sc = ax_3d.scatter(px, py, pz, c=colors_list, cmap=current_cmap, s=10, alpha=0.8)

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Depth/Intensity')
        ax_3d.set_title(f"3D Visualization ({style})")
        ax_3d.invert_yaxis()
        
        canvas4 = FigureCanvasTkAgg(fig4, master=self.dash_tab_3d)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill="both", expand=True)

        # --- Tab 5: Audio Log ---
        # self.dash_tabs.tab("Text Log").configure(title="Audio Log") # Rename if possible or just use new name
        ocr_scroll = ctk.CTkScrollableFrame(self.dash_tab_audio, label_text="Audio Transcript")
        ocr_scroll.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary
        if hasattr(self.engine, 'audio_summary') and self.engine.audio_summary:
            sum_frame = ctk.CTkFrame(ocr_scroll, fg_color="#442222")
            sum_frame.pack(fill=tk.X, pady=5)
            ctk.CTkLabel(sum_frame, text="SUMMARY", font=("Arial", 12, "bold"), text_color="orange").pack()
            ctk.CTkLabel(sum_frame, text=self.engine.audio_summary, wraplength=400, justify="left").pack(padx=5, pady=5)

        if hasattr(self.engine, 'transcript') and self.engine.transcript:
            for item in self.engine.transcript:
                # Handle varying tuple sizes just in case (start, end, text) or (start, end, text, category)
                category = ""
                if len(item) == 4:
                    start, end, text, category = item
                else:
                    start, end, text = item
                    
                row = ctk.CTkFrame(ocr_scroll, fg_color="#333")
                row.pack(fill=tk.X, pady=2)
                
                t_lbl = ctk.CTkLabel(row, text=f"[{start:.2f}s - {end:.2f}s]", width=120, text_color="cyan")
                t_lbl.pack(side=tk.LEFT, padx=5)
                
                if category and category != "Neutral":
                    c_lbl = ctk.CTkLabel(row, text=f"[{category}]", width=120, text_color="#ff77ff", font=("Arial", 11, "bold"))
                    c_lbl.pack(side=tk.LEFT, padx=2)
                
                txt_lbl = ctk.CTkLabel(row, text=text, text_color="white", wraplength=400, justify="left")
                txt_lbl.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        else:
            ctk.CTkLabel(ocr_scroll, text="No audio transcribed or disabled.").pack(pady=20)

    def save_current_analysis(self):
        if not self.engine.video_path:
            messagebox.showwarning("Save", "No video analyzed yet.")
            return
            
        success, msg = self.engine.save_analysis()
        if success:
            messagebox.showinfo("Saved", f"Analysis saved to:\n{msg}")
        else:
            messagebox.showerror("Error", f"Failed to save: {msg}")

    def open_library(self):
        # Create Toplevel Window
        lib_win = ctk.CTkToplevel(self)
        lib_win.title("Analysis Library")
        lib_win.geometry("600x700")
        lib_win.attributes('-topmost', True) # Keep on top
        lib_win.lift()
        lib_win.focus_force()
        
        # List files
        files_frame = ctk.CTkScrollableFrame(lib_win, label_text="Saved Sessions (Sorted by Date)")
        files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Use absolute path relative to script to ensure safely finding the folder
        lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Library")
        
        if os.path.exists(lib_dir):
            all_files = [f for f in os.listdir(lib_dir) if f.endswith(".pkl")]
            
            # Sort by modification time (newest first)
            # Store tuples: (timestamp, filename)
            files_with_time = []
            for f in all_files:
                path = os.path.join(lib_dir, f)
                mtime = os.path.getmtime(path)
                files_with_time.append((mtime, f))
            
            files_with_time.sort(key=lambda x: x[0], reverse=True)
            
            if not files_with_time:
                ctk.CTkLabel(files_frame, text="No saved analyses found.").pack(pady=20)
            
            self.lib_selections = {}
            current_date_group = None
            group_frame = None
            
            for mtime, filename in files_with_time:
                # Determine Date Label
                dt = datetime.fromtimestamp(mtime)
                date_str = dt.strftime("%Y-%m-%d")
                
                # Fancy date string (Today, Yesterday, etc.)
                today_str = datetime.now().strftime("%Y-%m-%d")
                yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                
                if date_str == today_str:
                    display_date = f"Today ({dt.strftime('%B %d')})"
                elif date_str == yesterday_str:
                    display_date = f"Yesterday ({dt.strftime('%B %d')})"
                else:
                    display_date = dt.strftime("%B %d, %Y")
                
                # Create New Group if Date Changed
                if display_date != current_date_group:
                    current_date_group = display_date
                    
                    # Collapsible Header (Button that toggles frame)
                    # We'll use a frame container for the header + content
                    container = ctk.CTkFrame(files_frame, fg_color="transparent")
                    container.pack(fill=tk.X, pady=5)
                    
                    # Content Frame (initially visible)
                    content_frame = ctk.CTkFrame(container)
                    
                    # Toggle Function
                    def toggle_group(frame, btn):
                        if frame.winfo_viewable():
                            frame.pack_forget()
                            btn.configure(text=f"▶ {btn.cget('text')[2:]}")
                        else:
                            frame.pack(fill=tk.X, padx=10, pady=2)
                            btn.configure(text=f"▼ {btn.cget('text')[2:]}")

                    header_btn = ctk.CTkButton(
                        container, 
                        text=f"▼ {display_date}", 
                        fg_color="#333", 
                        hover_color="#444",
                        anchor="w",
                        command=lambda f=content_frame, b=None: toggle_group(f, b)
                    )
                    header_btn.configure(command=lambda f=content_frame, b=header_btn: toggle_group(f, b))
                    header_btn.pack(fill=tk.X)
                    
                    content_frame.pack(fill=tk.X, padx=10, pady=2)
                    group_frame = content_frame # Set current target
                
                # Add File Item to Current Group Frame
                row = ctk.CTkFrame(group_frame)
                row.pack(fill=tk.X, pady=1)
                
                # Selection Var
                var = ctk.BooleanVar(value=False)
                self.lib_selections[filename] = var
                
                # Display filename with Time
                time_str = dt.strftime("%H:%M")
                
                # Clean up filename for display (remove ext and date suffix if possible)
                # pattern: {name}_Analysis_{date}.pkl
                display_name = filename
                try:
                    name_part = filename.replace(".pkl", "")
                    if "_Analysis_" in name_part:
                        base = name_part.split("_Analysis_")[0]
                        display_name = base
                    elif name_part.startswith("Analysis_"):
                         # Old format: Analysis_{name}_{date}
                         parts = name_part.split("_")
                         if len(parts) >= 3:
                             display_name = "_".join(parts[1:-2]) # Approximate
                except:
                    pass
                    
                label_text = f"[{time_str}] {display_name}  ({filename})"
                ctk.CTkCheckBox(row, text=label_text, variable=var).pack(side=tk.LEFT, padx=5, pady=2)
                
        else:
             ctk.CTkLabel(files_frame, text="Library folder not found.").pack(pady=20)
        
        # Bottom Controls
        btn_frame = ctk.CTkFrame(lib_win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        
        ctk.CTkButton(btn_frame, text="Load Selected", command=lambda: self.load_selected(lib_win)).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(btn_frame, text="Compare Selected", command=lambda: self.compare_selected(lib_win)).pack(side=tk.LEFT, padx=5)

    def load_selected(self, window):
        selected = [f for f, var in self.lib_selections.items() if var.get()]
        if len(selected) != 1:
            messagebox.showwarning("Load", "Please select exactly one file to load.")
            return
            
        filename = selected[0]
        lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Library")
        filepath = os.path.join(lib_dir, filename)
        
        success, msg = self.engine.load_analysis(filepath)
        if success:
            messagebox.showinfo("Success", "Analysis loaded!")
            window.destroy()
            self.lift() # Bring main window to front
            self.focus_force()
            self.update_view()
            self.generate_dashboard()
            # Reset Play slider
            if self.engine.total_frames > 0:
                self.play_slider.configure(to=self.engine.total_frames-1)
                self.play_slider.set(0)
                self.show_frame(0)
        else:
            messagebox.showerror("Error", f"Failed to load: {msg}")

    def compare_selected(self, lib_window=None):
        selected = [f for f, var in self.lib_selections.items() if var.get()]
        if len(selected) < 2:
            messagebox.showwarning("Compare", "Please select at least 2 files to compare.")
            return

        # Close library window if provided
        if lib_window:
            lib_window.destroy()

        comp_win = ctk.CTkToplevel(self)
        comp_win.title("Comparison View")
        comp_win.geometry("1400x900")
        comp_win.attributes('-topmost', True) # Keep on top
        comp_win.lift()
        comp_win.focus_force()
        
        # Load Data for all selected
        lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Library") 
        
        loaded_objects = [] # List of (filename, engine_instance)
        
        progress = ctk.CTkProgressBar(comp_win)
        progress.pack(pady=10)
        progress.set(0)
        lbl = ctk.CTkLabel(comp_win, text="Loading and Analyzing...")
        lbl.pack()
        comp_win.update()
        
        try:
            for i, f in enumerate(selected):
                path = os.path.join(lib_dir, f)
                # New engine instance for each to keep data separate
                eng = VideoAnalysisEngine(GESTURE_MODEL_PATH)
                eng.load_analysis(path) 
                loaded_objects.append({'name': f, 'engine': eng})
                progress.set((i+1)/len(selected))
                comp_win.update()
        except:
            lbl.configure(text="Error loading files.")
            return

        progress.destroy()
        lbl.destroy()
        
        # Tabs
        tab_view = ctk.CTkTabview(comp_win)
        tab_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tab_metrics = tab_view.add("Performance Metrics")
        tab_spatial = tab_view.add("Spatial Analysis")
        tab_semantic = tab_view.add("Semantic (Speech)")
        
        # --- TAB 1: Metrics ---
        self.plot_comparison_metrics(tab_metrics, loaded_objects)
        self.plot_gesture_comparison(tab_metrics, loaded_objects) # Add below metrics
        
        # --- TAB 2: Spatial ---
        self.plot_spatial_comparison(tab_spatial, loaded_objects)
        
        # --- TAB 3: Semantic ---
        print("\n ABOUT TO CALL plot_semantic_comparison...")
        self.plot_semantic_comparison(tab_semantic, loaded_objects)
        print(" plot_semantic_comparison RETURNED")

        # --- EPISODE ANALYSIS TABS (NEW) ---
        try:
            from episode_comparison import add_episode_tabs
            add_episode_tabs(tab_view, loaded_objects, comp_win)
        except Exception as e:
            print(f"Episode tabs integration failed: {e}")


    def get_comparison_metrics_figure(self, objects):
        stats_list = []
        for obj in objects:
            eng = obj['engine']
            metrics = eng.calculate_metrics()
            hmi_l = eng.calculate_hmi_metrics('left')
            hmi_r = eng.calculate_hmi_metrics('right')
            
            stats_list.append({
                'name': obj['name'][:15],
                'avg_speed_l': metrics['left']['avg_speed'],
                'efficiency_l': hmi_l['efficiency'] if hmi_l else 0,
                'jerk_l': hmi_l['avg_jerk'] if hmi_l else 0,
                'avg_speed_r': metrics['right']['avg_speed'],
                'efficiency_r': hmi_r['efficiency'] if hmi_r else 0,
                'jerk_r': hmi_r['avg_jerk'] if hmi_r else 0,
            })
        
        # Create Plotly 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Movement Tortuosity (Arc/Disp - Lower is Better)', 'Average Speed (Normalized)', 
                           'Movement Smoothness (Jerk Magnitude - Lower is Better)', 'Summary'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'table'}]]
        )
        
        names = [d['name'] for d in stats_list]
        
        # Efficiency bars
        fig.add_trace(go.Bar(
            x=names, y=[d['efficiency_l'] for d in stats_list],
            name='Left', marker_color='#636EFA', legendgroup='Left',
            hovertemplate='<b>%{x}</b><br>Left: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=names, y=[d['efficiency_r'] for d in stats_list],
            name='Right', marker_color='#EF553B', legendgroup='Right',
            hovertemplate='<b>%{x}</b><br>Right: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        
        # Speed bars
        fig.add_trace(go.Bar(
            x=names, y=[d['avg_speed_l'] for d in stats_list],
            name='Left', marker_color='#636EFA', showlegend=False, legendgroup='Left',
            hovertemplate='<b>%{x}</b><br>Left: %{y:.3f}<extra></extra>'
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=names, y=[d['avg_speed_r'] for d in stats_list],
            name='Right', marker_color='#EF553B', showlegend=False, legendgroup='Right',
            hovertemplate='<b>%{x}</b><br>Right: %{y:.3f}<extra></extra>'
        ), row=1, col=2)
        
        # Jerk bars
        fig.add_trace(go.Bar(
            x=names, y=[d['jerk_l'] for d in stats_list],
            name='Left', marker_color='#636EFA', showlegend=False, legendgroup='Left',
            hovertemplate='<b>%{x}</b><br>Left: %{y:.3f}<extra></extra>'
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=names, y=[d['jerk_r'] for d in stats_list],
            name='Right', marker_color='#EF553B', showlegend=False, legendgroup='Right',
            hovertemplate='<b>%{x}</b><br>Right: %{y:.3f}<extra></extra>'
        ), row=2, col=1)
        
        # Summary table
        if stats_list:
            best_eff = min(stats_list, key=lambda x: (x['efficiency_l'] + x['efficiency_r'])/2)
            fig.add_trace(go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[['Most Efficient', 'Score'],
                                  [best_eff['name'], 
                                   f"{(best_eff['efficiency_l'] + best_eff['efficiency_r'])/2:.3f}"]],
                          fill_color='lavender',
                          align='left')
            ), row=2, col=2)
        
        fig.update_layout(
            height=700,
            title="Performance Metrics Comparison",
            showlegend=True,
            hovermode='closest',
            barmode='group',
            template='plotly_white'
        )
        return fig

    def plot_comparison_metrics(self, parent, objects):
        fig = self.get_comparison_metrics_figure(objects)
        self._embed_plotly_figure(fig, parent)
    
    def _embed_matplotlib_figure(self, fig, parent, title="Chart"):
        """Embed Matplotlib figure with interactive navigation toolbar."""
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        
        # Create frame for chart
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add title
        ctk.CTkLabel(chart_frame, text=f"📈 {title}", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        
        # Add interactive toolbar
        toolbar_frame = tk.Frame(chart_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Pack canvas
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add hover tooltips if mplcursors available
        try:
            import mplcursors
            mplcursors.cursor(hover=True)
        except ImportError:
            pass
    
    def _force_new_window(self, url):
        """
        Aggressively attempt to open a new browser window.
        Standard webbrowser.open_new() often just opens a tab.
        We try to invoke the browser executable directly with --new-window.
        """
        import subprocess
        import shutil
        
        # 1. Try to find common browsers on Windows
        browsers = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]
        
        selected_browser = None
        for b in browsers:
            if os.path.exists(b):
                selected_browser = b
                break
        
        # 2. If found, use subprocess
        if selected_browser:
            try:
                subprocess.Popen([selected_browser, "--new-window", url])
                return
            except Exception as e:
                print(f"Failed to launch browser directly: {e}")
        
        # 3. Fallback to standard method if custom launch fails
        import webbrowser
        webbrowser.open_new(url)

    def _embed_plotly_figure(self, fig, parent):
        """Display Plotly figure in browser with auto-open and reopen button"""
        import tempfile
        
        # Generate HTML with full Plotly features
        html_str = fig.to_html(include_plotlyjs='cdn', config={
            'responsive': True,
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'hand_tracking_chart',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        })
        
        # Inject "Save as PDF" button
        pdf_button_code = """
        <style>
            #pdf-export-btn {
                position: fixed; top: 15px; right: 15px; z-index: 9999;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; border: none; padding: 12px 24px;
                border-radius: 8px; cursor: pointer; font-size: 15px;
                font-weight: 600; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            #pdf-export-btn:hover { transform: translateY(-2px); }
        </style>
        <button id="pdf-export-btn" onclick="exportToPDF()">📄 Save as PDF</button>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <script>
            function exportToPDF() {
                const btn = document.getElementById('pdf-export-btn');
                btn.textContent = '⏳ Generating...';
                const opt = {
                    margin: 0.5,
                    filename: 'topic_analysis_' + new Date().toISOString().slice(0,10) + '.pdf',
                    image: { type: 'jpeg', quality: 0.95 },
                    html2canvas: { scale: 2 },
                    jsPDF: { unit: 'in', format: 'a4', orientation: 'landscape' }
                };
                html2pdf().set(opt).from(document.body).save().then(() => {
                    btn.textContent = '✓ Saved!'; setTimeout(() => btn.textContent = '📄 Save as PDF', 2000);
                });
            }
        </script>
        """
        if '</body>' in html_str:
            html_str = html_str.replace('</body>', pdf_button_code + '</body>')
        elif '</html>' in html_str:
            html_str = html_str.replace('</html>', pdf_button_code + '</html>')
        else:
            html_str += pdf_button_code
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
            f.write(html_str)
            temp_path = f.name
        
        url = 'file:///' + temp_path.replace('\\', '/')
        
        # Auto-open in new window
        self._force_new_window(url)
        
        # Show confirmation UI
        frame = ctk.CTkFrame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Success message
        ctk.CTkLabel(frame, text="✓ Interactive Chart Opened in Browser", 
                    font=("Arial", 18, "bold"), text_color="#00aa00").pack(pady=(40, 20))
        
        # Info box
        info_frame = ctk.CTkFrame(frame, fg_color=("#E8F5E9", "#1B5E20"))
        info_frame.pack(pady=10, padx=40, fill=tk.X)
        
        ctk.CTkLabel(info_frame, text="Chart Features:", 
                    font=("Arial", 13, "bold")).pack(anchor="w", padx=15, pady=(10, 5))
        
        features = [
            "🖱️  Hover over data points for exact values",
            "🔍  Click and drag to zoom into regions",
            "↩️  Double-click to reset view",
            "📷  Camera icon (top-right) to export PNG",
            "👁️  Click legend items to hide/show data"
        ]
        
        for feature in features:
            ctk.CTkLabel(info_frame, text=feature, 
                        font=("Arial", 11), anchor="w").pack(anchor="w", padx=25, pady=2)
        
        ctk.CTkLabel(info_frame, text="").pack(pady=5)  # Spacer
        
        # Reopen button
        ctk.CTkButton(frame, text="🔄 Reopen Chart in New Window", 
                     command=lambda: self._force_new_window(url),
                     height=45, font=("Arial", 13, "bold"),
                     fg_color=("#2196F3", "#1976D2")).pack(pady=20)

    def get_spatial_comparison_figure(self, objects):
        """Generates an interactive spatial density plot with source switching."""
        
        sources = ["Whole Hand", "Wrist", "Index", "Thumb", "Median"]
        fig = go.Figure()
        
        # Pre-calculate histograms for all sources
        for i, source in enumerate(sources):
            all_points = []
            for obj in objects:
                eng = obj['engine']
                pts = eng.get_normalized_points(source)
                all_points.extend(pts)
            
            if not all_points:
                # Add empty dummy trace to keep indices aligned
                fig.add_trace(go.Histogram2dContour(x=[], y=[], visible=False))
                continue
                
            xs = [p[0] for p in all_points]
            ys = [1-p[1] for p in all_points]  # Invert Y
            
            # Setup visibility: only first source visible by default
            is_visible = (i == 0)
            
            fig.add_trace(go.Histogram2dContour(
                x=xs, y=ys,
                colorscale='Hot',
                reversescale=False,
                contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
                hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z}<extra></extra>',
                colorbar=dict(title="Dwell Time<br>Density"),
                visible=is_visible,
                name=source
            ))

        # Create buttons for the dropdown
        buttons = []
        for i, source in enumerate(sources):
            # Create visibility mask: only the i-th trace is True
            visibility = [False] * len(sources)
            visibility[i] = True
            
            button = dict(
                label=source,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Spatial Occupancy Density (Heatmap) - {source}"}]
            )
            buttons.append(button)

        fig.update_layout(
            title=f"Spatial Occupancy Density (Heatmap) - {sources[0]}",
            xaxis_title="X Position (Normalized)",
            yaxis_title="Y Position (Normalized)",
            height=700,
            template='plotly_white',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor="y", scaleratio=1),
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    x=1.12,
                    y=1,
                    xanchor='left',
                    yanchor='top'
                )
            ]
        )
        return fig

    def plot_spatial_comparison(self, parent, objects):
        # Frame for interactive display
        self._spatial_plot_frame = ctk.CTkFrame(parent)
        self._spatial_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # New interactive figure handles logic internally, so just embed it
        fig = self.get_spatial_comparison_figure(objects)
        self._embed_plotly_figure(fig, self._spatial_plot_frame)

    def get_semantic_comparison_figure(self, objects):
        """
        Dual-view topic analysis:
        1. Aggregated: Topics across ALL sessions (common themes)
        2. Per-Session: Topics unique to each session (differences)
        """
        from topic_modeling import HybridTopicModeler
        
        
        # Create simple session labels: "Session 1", "Session 2", etc.
        session_labels = {obj_idx: f"Session {obj_idx + 1}" for obj_idx in range(len(objects))}
        
        # Collect segments with metadata
        all_segments = []
        segment_metadata = []
        
        for obj_idx, obj in enumerate(objects):
            eng = obj['engine']
            for entry in eng.transcript:
                text = entry[2] if len(entry) > 2 else ""
                sentiment = entry[3] if len(entry) > 3 else "neutral"
                timestamp = entry[0] if len(entry) > 0 else 0
                
                if text and len(text) > 10:
                    all_segments.append(text)
                    segment_metadata.append({
                        'session': session_labels[obj_idx],
                        'session_idx': obj_idx,
                        'timestamp': timestamp,
                        'sentiment': sentiment,
                        'text': text
                    })
        
        if len(all_segments) < 3:
            return self._create_empty_topic_figure()
        
        # === 1. AGGREGATED ANALYSIS (all sessions combined) ===
        print(f"\n=== AGGREGATED TOPIC ANALYSIS ===")
        print(f"Analyzing {len(all_segments)} total segments across {len(objects)} sessions...")
        
        modeler_aggregated = HybridTopicModeler(n_topics=min(8, len(all_segments) // 4))
        aggregated_result = modeler_aggregated.extract_topics(all_segments)
        
        # === 2. BUILD SESSION INFO (no per-session BERTopic needed) ===
        # Derive session-level breakdowns from the aggregated result + metadata
        session_names = list(set(m['session'] for m in segment_metadata))
        session_names.sort()  # Deterministic order
        n_sessions = len(session_names)
        print(f"\n  [Info] {n_sessions} session(s) detected: {session_names}")
        
        # === CREATE DUAL-VIEW VISUALIZATION ===
        dual_view_fig = self._create_dual_view_figure(
            aggregated_result, 
            segment_metadata, 
            objects
        )
        
        # === CREATE NETWORK GRAPH ===
        try:
            print("\n=== GENERATING NETWORK GRAPH ===")
            network_fig = self.get_topic_network_graph(aggregated_result, segment_metadata)
            
            # Store for later reference
            self._last_aggregated_result = aggregated_result
            self._last_metadata = segment_metadata
            self._last_network_fig = network_fig
            print("✓ Network graph generated successfully")
        except Exception as e:
            print(f"⚠️ Network graph generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # === GENERATE EXECUTIVE SUMMARY HTML ===
        try:
            topic_summaries = aggregated_result.get('topic_summaries', {})
            topic_labels = aggregated_result.get('topic_label_map', {})
            topic_words = aggregated_result.get('topic_words', {})
            
            if topic_summaries:
                print("\n=== GENERATING EXECUTIVE SUMMARY ===")
                self._last_executive_summary_html = self._create_executive_summary_html(
                    topic_summaries, topic_labels, topic_words
                )
                print(f"✓ Executive summary generated for {len(topic_summaries)} topics")
            else:
                self._last_executive_summary_html = ""
        except Exception as e:
            print(f"⚠️ Executive summary generation failed: {e}")
            self._last_executive_summary_html = ""
        
        return dual_view_fig
    
    def get_topic_network_graph(self, aggregated_result, metadata):
        """
        Create 'Knowledge Graph' of Topics and Concepts (Keywords).
        - Nodes: Topics (Anchors), Keywords (Concepts).
        - Edges: Topic-Keyword (Definition), Topic-Topic (Similarity).
        - Layout: Force-directed (Spring) with Topics anchored to semantic UMAP coords.
        """
        import networkx as nx
        import numpy as np
        
        topics = aggregated_result['topics']
        topic_labels = aggregated_result['topic_labels']
        topic_words = aggregated_result['topic_words']
        topic_sizes = aggregated_result['topic_sizes']
        topic_coords = aggregated_result.get('topic_coords', {})
        
        # Create network graph
        G = nx.Graph()
        
        # 1. Add Topic Nodes (Anchors)
        # We will use the UMAP coordinates (topic_coords) as FIXED positions for these nodes
        # to ensure the "map" is semantically meaningful.
        topic_nodes = []
        fixed_positions = {}
        
        # Normalize UMAP coords to [-1, 1] range for Spring Layout
        has_coords = False
        if topic_coords:
            raw_coords = np.array([c for tid, c in topic_coords.items() if tid < len(topic_labels)])
            if len(raw_coords) > 0:
                has_coords = True
                min_c = raw_coords.min(axis=0)
                max_c = raw_coords.max(axis=0)
                range_c = max_c - min_c
                range_c[range_c == 0] = 1
                
                # Normalize and center
                for tid, coord in topic_coords.items():
                    if tid < len(topic_labels) and tid != -1:
                        norm = (np.array(coord) - min_c) / range_c
                        norm = (norm - 0.5) * 2.0 # [-1, 1]
                        
                        label = topic_labels[tid]
                        node_id = f"T_{tid}" # Use ID for robustness
                        
                        G.add_node(node_id, 
                                   node_type='topic', 
                                   label=label, 
                                   size=topic_sizes.get(tid, 10),
                                   color='#e74c3c') # Red
                        
                        fixed_positions[node_id] = norm.tolist()
                        topic_nodes.append(node_id)
        
        # FALLBACK: If no coords (NMF Mode), just add nodes
        if not has_coords:
            for tid, label in enumerate(topic_labels):
                 node_id = f"T_{tid}"
                 G.add_node(node_id, 
                           node_type='topic', 
                           label=label, 
                           size=topic_sizes.get(tid, 10),
                           color='#e74c3c') # Red
                 topic_nodes.append(node_id)

        # 2. Add Keyword Nodes (Concepts)
        # Top 5 keywords for each topic
        for tid, words in topic_words.items():
            if tid not in topic_coords or tid == -1: continue # Skip outliers or missing topics
            
            node_id = f"T_{tid}"
            if node_id not in G.nodes: continue
            
            for word, score in words[:5]:
                kw_node_id = f"K_{word}"
                
                if kw_node_id not in G.nodes:
                    G.add_node(kw_node_id, 
                               node_type='keyword', 
                               label=word, 
                               size=5, 
                               color='#3498db') # Blue
                
                # Edge: Topic defines Concept
                # Weight by score so strong definitions are shorter
                G.add_edge(node_id, kw_node_id, weight=score*5)

        # 3. Add Topic-Topic Edges (Travel Lines)
        # Connect topics that are semantically close (distance < threshold)
        # This creates the "Knowledge Web"
        
        if len(topic_nodes) > 1:
            import itertools
            for t1, t2 in itertools.combinations(topic_nodes, 2):
                # Only draw similarity edges if we have coordinates (fixed_positions)
                # If we are in NMF mode (no coords), we rely on spring layout and don't force similarity layout
                if t1 in fixed_positions and t2 in fixed_positions:
                    p1 = np.array(fixed_positions[t1])
                    p2 = np.array(fixed_positions[t2])
                    dist = np.linalg.norm(p1 - p2)
                    
                    # Heuristic: Connect if closer than 0.4 (in normalized -1,1 space)
                    if dist < 0.4:
                        G.add_edge(t1, t2, weight=(1.0 - dist)*2, edge_type='similarity')

        # 4. Compute Layout
        # Topics are fixed anchors (if coords exist). Keywords float around them.
        print(f"  [Knowledge Graph] Calculating layout for {len(G.nodes)} nodes...")
        
        pos = {}
        if not G.nodes:
            print("  [Warning] Graph has no nodes, skipping layout.")
        else:
            if fixed_positions:
                pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_positions.keys(), k=0.15, iterations=50, seed=42)
            else:
                # Fallback layout if no UMAP coords (NMF mode)
                pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # 5. Build Plotly Figure
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Subtle grey for concepts, bolder for similarity
            if edge[2].get('edge_type') == 'similarity':
                edge_colors.extend(['#cfd8dc', '#cfd8dc', '#cfd8dc']) # Thicker/Visible? Plotly lines are single style batch usually.
            else:
                edge_colors.extend(['#ecf0f1', '#ecf0f1', '#ecf0f1'])

        # Draw Edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#bdc3c7'), # Light Grey
            hoverinfo='none',
            mode='lines'
        )

        # Draw Nodes
        node_x = []
        node_y = []
        node_text = []
        node_marker_size = []
        node_marker_color = []
        custom_data = [] # For hover
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            data = G.nodes[node]
            node_text.append(data['label'])
            
            if data['node_type'] == 'topic':
                # Scale topic size: 20 -> 50
                s = data['size']
                # functional scale for visibility
                sz = 15 + np.log(s+1)*8 
                node_marker_size.append(sz)
                node_marker_color.append('#e74c3c') # Red
                custom_data.append(f"<b>Topic: {data['label']}</b><br>Size: {s}")
            else:
                node_marker_size.append(8) # Small steady size for keywords
                node_marker_color.append('#3498db') # Blue
                custom_data.append(f"Concept: {data['label']}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            textfont=dict(family='Arial', size=10, color='#2c3e50'),
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                color=node_marker_color,
                size=node_marker_size,
                line_width=1,
                line_color='white'
            ),
            customdata=custom_data,
            hovertemplate='%{customdata}<extra></extra>'
        )

        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(
                    text='Topic Knowledge Graph',
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Network of Topics (Red) and Concepts (Blue)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_white',
                height=600
            )
        )
        return fig
    
    def _create_executive_summary_html(
        self, 
        topic_summaries: dict, 
        topic_labels: dict, 
        topic_words: dict
    ) -> str:
        """
        Generate styled HTML cards for topic executive summaries.
        
        Args:
            topic_summaries: {topic_id: [takeaway1, takeaway2, ...]}
            topic_labels: {topic_id: "Label"}
            topic_words: {topic_id: [(word, score), ...]}
            
        Returns:
            HTML string with styled summary cards
        """
        if not topic_summaries:
            return ""
        
        html_parts = ["""
        <style>
            .executive-summary { 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); 
                gap: 20px; 
                padding: 20px;
            }
            .summary-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 20px;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                max-height: 400px;
                overflow-y: auto;
                position: relative;
            }
            
            /* Custom Scrollbar */
            .summary-card::-webkit-scrollbar {
                width: 8px;
            }
            .summary-card::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
            }
            .summary-card::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px; 
            }
            .summary-card::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 255, 255, 0.5); 
            }

            .summary-card h3 {
                margin: 0 0 10px 0;
                font-size: 1.2em;
                border-bottom: 1px solid rgba(255,255,255,0.3);
                padding-bottom: 8px;
                padding-right: 60px; /* Space for copy button */
            }
            
            .copy-btn {
                position: absolute;
                top: 15px;
                right: 15px;
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.4);
                color: white;
                border-radius: 6px;
                padding: 4px 10px;
                cursor: pointer;
                font-size: 0.8em;
                transition: all 0.2s;
            }
            .copy-btn:hover {
                background: rgba(255, 255, 255, 0.4);
                transform: scale(1.05);
            }

            .summary-card .keywords {
                font-size: 0.85em;
                opacity: 0.8;
                margin-bottom: 12px;
            }
            .summary-card ul {
                margin: 0;
                padding-left: 20px;
            }
            .summary-card li {
                margin-bottom: 6px;
                line-height: 1.4;
            }
        </style>
        
        <script>
        function copyCard(btn) {
            const card = btn.parentElement;
            // Clone to avoid modifying UI, and exclude button text
            const clone = card.cloneNode(true);
            const btnClone = clone.querySelector('.copy-btn');
            if(btnClone) btnClone.remove();
            
            const text = clone.innerText.trim();
            
            // Create hidden textarea for copy command
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            
            try {
                document.execCommand('copy');
                const originalText = btn.innerText;
                btn.innerText = 'Copied!';
                setTimeout(() => { btn.innerText = originalText; }, 2000);
            } catch (err) {
                console.error('Failed to copy', err);
                alert('Failed to copy to clipboard');
            }
            document.body.removeChild(textarea);
        }
        </script>

        <div class="executive-summary">
        """]
        
        for tid, takeaways in topic_summaries.items():
            label = topic_labels.get(tid, f"Topic {tid}")
            keywords = [w[0] for w in topic_words.get(tid, [])[:5]]
            keywords_str = ", ".join(keywords) if keywords else "No keywords"
            
            takeaway_items = "".join([f"<li>{t}</li>" for t in takeaways])
            
            html_parts.append(f"""
            <div class="summary-card">
                <button class="copy-btn" onclick="copyCard(this)">Copy</button>
                <h3>{label}</h3>
                <div class="keywords">Keywords: {keywords_str}</div>
                <ul>{takeaway_items}</ul>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _create_empty_topic_figure(self):
        """Return empty figure when no data"""
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient transcript data for topic modeling<br>(Need at least 3 segments)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            title="Topic Analysis",
            height=700,
            template='plotly_white'
        )
        return fig
    
    def _create_dual_view_figure(self, aggregated, metadata, sessions):
        """
        REWRITTEN: Full 3x2 grid with proper BERTopic data handling.
        Row 1: Aggregated Topics (Bubble), Aggregated Sentiment Analysis
        Row 2: Topic Distribution by Session, Topic Timeline
        Row 3: Top Words, Session Summary
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Extract data
        topics = aggregated['topics']
        topic_label_map = aggregated.get('topic_label_map', {})
        topic_sizes = aggregated['topic_sizes']
        topic_words = aggregated.get('topic_words', {})
        
        unique_topics = sorted([t for t in set(topics) if t != -1])
        
        print(f"  [Dual-View V2] Creating 3x2 figure for {len(unique_topics)} topics")
        
        # Create 3x2 subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'Aggregated Topics ({aggregated["method"]} - {len(unique_topics)} topics)',
                'Aggregated Sentiment Analysis',
                'Topic Distribution by Session',
                'Topic Timeline (Aggregated)',
                'Top Words (Most Discussed Topic)',
                'Session Summary'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'table'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
            row_heights=[0.35, 0.35, 0.30]
        )
        
        # === ROW 1, COL 1: Aggregated Topic Bubble Map ===
        try:
            topic_sentiments = {}
            for topic_id in unique_topics:
                topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
                sentiments = [metadata[i]['sentiment'].lower() for i in topic_docs if i < len(metadata)]
                sent_score = sum(1 if ('positive' in s or 'joy' in s) else (-1 if ('negative' in s or 'anger' in s or 'sadness' in s) else 0) for s in sentiments)
                topic_sentiments[topic_id] = sent_score / len(sentiments) if sentiments else 0
            
            for topic_id in unique_topics:
                label = topic_label_map.get(topic_id, f"Topic {topic_id}")
                count = topic_sizes.get(topic_id, 0)
                sent = topic_sentiments.get(topic_id, 0)
                
                max_count = max(topic_sizes.values()) if topic_sizes else 1
                marker_size = (count / max_count) * 50 + 15
                color = '#2ecc71' if sent > 0.3 else ('#e74c3c' if sent < -0.3 else '#95a5a6')
                
                fig.add_trace(go.Scatter(
                    x=[count], y=[sent],
                    mode='markers+text',
                    marker=dict(size=marker_size, color=color, opacity=0.7),
                    text=[label[:20]],
                    textposition="top center",
                    textfont=dict(size=9),
                    hovertemplate=f"<b>{label}</b><br>Count: {count}<br>Sentiment: {sent:.2f}<extra></extra>",
                    showlegend=False
                ), row=1, col=1)
            
            fig.update_xaxes(title="Frequency", showgrid=True, row=1, col=1)
            fig.update_yaxes(title="Sentiment", showgrid=True, row=1, col=1, range=[-1.1, 1.1])
        except Exception as e:
            print(f"  [Error] Bubble map: {e}")
        
        # === ROW 1, COL 2: Sentiment Analysis (Avg Sentiment per Topic) ===
        try:
            # Group by sentiment ranges
            pos_topics = [(tid, topic_label_map.get(tid, f"T{tid}"), topic_sizes.get(tid, 0)) 
                         for tid in unique_topics if topic_sentiments.get(tid, 0) > 0.3]
            neg_topics = [(tid, topic_label_map.get(tid, f"T{tid}"), topic_sizes.get(tid, 0)) 
                         for tid in unique_topics if topic_sentiments.get(tid, 0) < -0.3]
            neu_topics = [(tid, topic_label_map.get(tid, f"T{tid}"), topic_sizes.get(tid, 0)) 
                         for tid in unique_topics if -0.3 <= topic_sentiments.get(tid, 0) <= 0.3]
            
            all_sentiment_topics = pos_topics + neu_topics + neg_topics
            x_vals = [item[2] for item in all_sentiment_topics]  # mentions
            y_vals = [topic_sentiments.get(item[0], 0) for item in all_sentiment_topics]  # avg sentiment
            labels = [item[1][:25] for item in all_sentiment_topics]
            colors = ['#2ecc71' if y > 0.3 else '#e74c3c' if y < -0.3 else '#95a5a6' for y in y_vals]
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers',
                marker=dict(size=12, color=colors, opacity=0.7),
                text=labels,
                hovertemplate="<b>%{text}</b><br>Mentions: %{x}<br>Avg Sentiment: %{y:.2f}<extra></extra>",
                showlegend=False
            ), row=1, col=2)
            
            fig.update_xaxes(title="Mentions", showgrid=True, row=1, col=2)
            fig.update_yaxes(title="Avg Sentiment", showgrid=True, row=1, col=2, range=[-1.1, 1.1])
        except Exception as e:
            print(f"  [Error] Sentiment analysis: {e}")
        
        # === ROW 2, COL 1: Topic Distribution by Session (Stacked Bar) ===
        try:
            # Derive session names from metadata
            session_names_unique = sorted(set(m['session'] for m in metadata))
            n_sessions = len(session_names_unique)
            
            if n_sessions > 1:
                # Multiple sessions: show stacked bar derived from aggregated topics + metadata
                for topic_id in unique_topics:
                    label = topic_label_map.get(topic_id, f"Topic {topic_id}")
                    counts_per_session = []
                    
                    for sess_name in session_names_unique:
                        # Count how many segments in this session were assigned to this topic
                        count = sum(1 for i, m in enumerate(metadata) 
                                   if m['session'] == sess_name and i < len(topics) and topics[i] == topic_id)
                        counts_per_session.append(count)
                    
                    fig.add_trace(go.Bar(
                        x=session_names_unique,
                        y=counts_per_session,
                        name=label[:25],
                        hovertemplate=f"<b>{label}</b><br>%{{y}}<extra></extra>"
                    ), row=2, col=1)
                
                fig.update_xaxes(title="Session", row=2, col=1)
                fig.update_yaxes(title="Topic Mentions", row=2, col=1)
            else:
                # Single session: show simple topic frequency bar chart
                labels = [topic_label_map.get(tid, f"Topic{tid}")[:30] for tid in unique_topics]
                counts = [topic_sizes.get(tid, 0) for tid in unique_topics]
                
                fig.add_trace(go.Bar(
                    x=labels,
                    y=counts,
                    marker_color='#4A90E2',
                    showlegend=False,
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
                ), row=2, col=1)
                
                fig.update_xaxes(title="Topic", tickangle=-45, row=2, col=1)
                fig.update_yaxes(title="Frequency", row=2, col=1)
        except Exception as e:
            print(f"  [Error] Topic distribution: {e}")
        
        # === ROW 2, COL 2: Topic Timeline ===
        try:
            for topic_id in unique_topics:
                label = topic_label_map.get(topic_id, f"Topic {topic_id}")
                topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
                times = [metadata[i]['timestamp'] for i in topic_docs if i < len(metadata)]
                
                if times:
                    fig.add_trace(go.Scatter(
                        x=times, y=[topic_id] * len(times),
                        mode='markers',
                        marker=dict(size=5, opacity=0.6),
                        name=label,
                        showlegend=False,
                        hovertemplate=f"<b>{label}</b><br>Time: %{{x}}<extra></extra>"
                    ), row=2, col=2)
            
            fig.update_xaxes(title="Time (seconds)", showgrid=True, row=2, col=2)
            fig.update_yaxes(title="Topic", showgrid=True, row=2, col=2)
        except Exception as e:
            print(f"  [Error] Timeline: {e}")
        
        # === ROW 3, COL 1: Top Words ===
        try:
            if topic_sizes:
                top_topic = max(topic_sizes, key=topic_sizes.get)
                words = topic_words.get(top_topic, [])[:10]
                
                if words:
                    fig.add_trace(go.Bar(
                        y=words[::-1], x=[1]*len(words),
                        orientation='h',
                        marker_color='#4A90E2',
                        showlegend=False
                    ), row=3, col=1)
                    
                    label = topic_label_map.get(top_topic, f"Topic {top_topic}")
                    fig.update_xaxes(title=f"Relevance Score ({label[:30]})", showticklabels=False, row=3, col=1)
                    fig.update_yaxes(title="Words", row=3, col=1)
        except Exception as e:
            print(f"  [Error] Top words: {e}")
        
        # === ROW 3, COL 2: Session Summary Table ===
        try:
            if metadata:
                # Derive session summary from aggregated topics + metadata
                sess_names_sorted = sorted(set(m['session'] for m in metadata))
                seg_counts = []
                primary_topics = []
                
                for sess_name in sess_names_sorted:
                    # Count segments in this session
                    sess_indices = [i for i, m in enumerate(metadata) if m['session'] == sess_name]
                    seg_counts.append(len(sess_indices))
                    
                    # Find primary topic from aggregated assignments
                    topic_counts = {}
                    for idx in sess_indices:
                        if idx < len(topics):
                            t = topics[idx]
                            if t != -1:
                                topic_counts[t] = topic_counts.get(t, 0) + 1
                    if topic_counts:
                        primary_tid = max(topic_counts, key=topic_counts.get)
                        primary_topics.append(topic_label_map.get(primary_tid, f"Topic {primary_tid}")[:40])
                    else:
                        primary_topics.append("N/A")
                
                fig.add_trace(go.Table(
                    header=dict(values=['<b>Session</b>', '<b>Segments</b>', '<b>Primary Topic</b>', '<b>Method</b>'],
                               fill_color='#4A90E2', font=dict(color='white', size=12)),
                    cells=dict(values=[sess_names_sorted, seg_counts, primary_topics, [aggregated['method']]*len(sess_names_sorted)],
                              fill_color='#f9f9f9', align='left')
                ), row=3, col=2)
        except Exception as e:
            print(f"  [Error] Summary table: {e}")
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f"<b>Topic Analysis Report</b><br>"
                     f"<sub>Aggregated: {aggregated['method']} (Quality: {aggregated.get('quality_score', 0):.2f}) | "
                     f"Sessions: {len(sessions)}</sub>",
                font=dict(size=18)
            ),
            height=1600,
            showlegend=True,
            template='plotly_white'
        )
        
        print(f"  [Dual-View V2] Complete with {len(fig.data)} traces")
        return fig
    



    def _add_aggregated_bubble_map(self, fig, result, metadata, row, col):
        """Bubble map of aggregated topics"""
        topics = result['topics']
        topic_labels = result['topic_labels']
        topic_sizes = result['topic_sizes']
        topic_label_map = result.get('topic_label_map', {})  # BERTopic provides this
        
        print(f"    [Bubble Map Debug] topics type: {type(topics)}, len: {len(topics) if hasattr(topics, '__len__') else 'N/A'}")
        print(f"    [Bubble Map Debug] unique topics: {set(topics) if hasattr(topics, '__iter__') else topics}")
        print(f"    [Bubble Map Debug] topic_labels type: {type(topic_labels)}, len: {len(topic_labels)}")
        print(f"    [Bubble Map Debug] topic_sizes: {topic_sizes}")
        print(f"    [Bubble Map Debug] topic_label_map: {topic_label_map}")
        print(f"    [Bubble Map Debug] metadata len: {len(metadata)}")
        
        if len(set(topics)) < 2:
            print(f"    [Bubble Map Debug] Exiting early: only {len(set(topics))} unique topic(s)")
            fig.add_annotation(
                text="Only 1 topic found",
                xref=f"x{col}", yref=f"y{row}",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="gray")
            )
            return
        
        # Calculate sentiment per topic
        topic_sentiments = {}
        for topic_id in set(topics):
            topic_segments = [i for i, t in enumerate(topics) if t == topic_id]
            sentiments = [metadata[i]['sentiment'].lower() for i in topic_segments if i < len(metadata)]
            
            sent_score = 0
            for s in sentiments:
                if 'positive' in s or 'joy' in s:
                    sent_score += 1
                elif 'negative' in s or 'anger' in s or 'sadness' in s:
                    sent_score -= 1
            avg_sent = sent_score / len(sentiments) if sentiments else 0
            topic_sentiments[topic_id] = avg_sent
        
        # Determine strict scaling factor
        max_count = max(topic_sizes.values()) if topic_sizes else 1
        
        #Position and plot bubbles
        bubbles_added = 0
        for topic_id in set(topics):
            if topic_id not in topic_sizes:
                continue
            
            # Normalize topic_id to regular int (handles np.int64)
            topic_id_int = int(topic_id)
            
            # Get label: use map if available, otherwise list indexing (for backwards compat)
            if topic_label_map and topic_id_int in topic_label_map:
                label = topic_label_map[topic_id_int]
            elif topic_id_int >= 0 and topic_id_int < len(topic_labels):
                label = topic_labels[topic_id_int]
            else:
                label = f"Topic {topic_id_int}"
            
            count = topic_sizes[topic_id]
            sent = topic_sentiments.get(topic_id, 0)
            
            # Logarithmic-like scaling: (val / max) * max_pixel + min_pixel
            # But making sure big topics don't explode
            norm_size = (count / max_count) 
            marker_size = (norm_size * 60) + 15  # Range: 15px to 75px
            
            # Position: frequency with jitter
            x = count + np.random.uniform(-0.5, 0.5)
            y = sent + np.random.uniform(-0.05, 0.05)
            
            color = '#2ecc71' if sent > 0.3 else ('#e74c3c' if sent < -0.3 else '#95a5a6')
            
            bubbles_added += 1
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=marker_size, color=color, opacity=0.7, line=dict(width=1, color='white')),
                text=[label],
                textposition="middle center",
                textfont=dict(size=min(12, int(marker_size/3)), color='white', family='Arial Black'), # Scale font too
                hovertext=f"<b>{label}</b><br>Mentions: {count}<br>Sentiment: {sent:.2f}",
                hoverinfo='text',
                showlegend=False
            ), row=row, col=col)
        
        print(f"    [Bubble Map Debug] Added {bubbles_added} bubbles to figure")
        fig.update_xaxes(title="Frequency", showgrid=True, row=row, col=col)
        fig.update_yaxes(title="Sentiment", showgrid=True, row=row, col=col, range=[-1.1, 1.1])
    
    def _add_session_topic_distribution(self, fig, aggregated, per_session_results, metadata, row, col):
        """Show how aggregated topics are distributed across sessions"""
        if not per_session_results:
            return
        
        # For each session, count which aggregated topics appear
        session_names = []
        topic_distributions = {topic_id: [] for topic_id in set(aggregated['topics'])}
        
        for session_result in per_session_results:
            session_names.append(session_result['session_name'])
            
            # Count topic occurrences in this session's segments
            session_topic_counts = {tid: 0 for tid in set(aggregated['topics'])}
            
            # Find this session's segments in metadata
            session_segments_idx = [i for i, m in enumerate(metadata) if m['session'] == session_result['session_name']]
            
            for idx in session_segments_idx:
                if idx < len(aggregated['topics']):
                    topic_id = aggregated['topics'][idx]
                    session_topic_counts[topic_id] += 1
            
            for topic_id in topic_distributions:
                topic_distributions[topic_id].append(session_topic_counts.get(topic_id, 0))
        
        # Plot stacked bars
        for topic_id, counts in topic_distributions.items():
            if topic_id < len(aggregated['topic_labels']):
                label = aggregated['topic_labels'][topic_id]
                
                fig.add_trace(go.Bar(
                    x=session_names,
                    y=counts,
                    name=label,
                    hovertemplate=f'{label}<br>%{{y}} mentions<extra></extra>'
                ), row=row, col=col)
        
        fig.update_xaxes(title="Session", row=row, col=col)
        fig.update_yaxes(title="Topic Mentions", row=row, col=col)
        fig.update_layout(barmode='stack')
    
    def _create_topic_multiview(self, topic_result: dict, metadata: list, sessions: list):
        """Create comprehensive topic visualization with multiple views"""
        
        # Create subplot figure with 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Topic Overview ({topic_result["method"]} - {topic_result["n_topics"]} topics)',
                'Topic Sentiment Analysis',
                'Topic Timeline',
                'Top Words per Topic'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Extract data
        topics = topic_result['topics']
        topic_labels = topic_result['topic_labels']
        topic_words = topic_result['topic_words']
        topic_sizes = topic_result['topic_sizes']
        
        # View 1:  Topic Bubble Map (pyLDAvis-style)
        self._add_topic_bubble_map(fig, topic_result, metadata, row=1, col=1)
        
        # View 2: Sentiment Analysis
        self._add_sentiment_analysis(fig, topics, topic_labels, metadata, row=1, col=2)
        
        # View 3: Timeline
        self._add_topic_timeline(fig, topics, topic_labels, metadata, row=2, col=1)
        
        # View 4: Word Bars
        self._add_top_words(fig, topic_words, topic_labels, topic_sizes, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>Topic Analysis Report</b><br><sub>Method: {topic_result['method']} | Quality: {topic_result.get('quality_score', 0):.2f} | Topics: {topic_result['n_topics']}</sub>",
                font=dict(size=18)
            ),
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _add_topic_bubble_map(self, fig, topic_result, metadata, row, col):
        """Add pyLDAvis-style topic bubble map"""
        topics = topic_result['topics']
        topic_labels = topic_result['topic_labels']
        topic_sizes = topic_result['topic_sizes']
        
        if len(set(topics)) <2:
            return
        
        # Position topics in 2D space (simplified - using frequency and sentiment)
        topic_positions = {}
        topic_sentiments = {}
        
        for topic_id in set(topics):
            if topic_id not in topic_sizes:
                continue
                
            # Calculate average sentiment for this topic
            topic_segments = [i for i, t in enumerate(topics) if t == topic_id]
            sentiments = [metadata[i]['sentiment'].lower() for i in topic_segments if i < len(metadata)]
            
            # Map to score
            sent_score = 0
            for s in sentiments:
                if 'positive' in s or 'joy' in s:
                    sent_score += 1
                elif 'negative' in s or 'anger' in s or 'sadness' in s:
                    sent_score -= 1
            avg_sent = sent_score / len(sentiments) if sentiments else 0
            
            # Positioning: x = frequency, y = sentiment (with jitter)
            topic_positions[topic_id] = (
                topic_sizes[topic_id] + np.random.uniform(-2, 2),
                avg_sent + np.random.uniform(-0.1, 0.1)
            )
            topic_sentiments[topic_id] = avg_sent
        
        # Determine strict scaling factor
        max_count = max(topic_sizes.values()) if topic_sizes else 1

        # Plot bubbles
        for topic_id, (x, y) in topic_positions.items():
            if topic_id >= len(topic_labels):
                continue
                
            size = topic_sizes.get(topic_id, 1)
            label = topic_labels[topic_id] if topic_id < len(topic_labels) else f"Topic {topic_id}"
            
            # Dynamic scaling
            norm_size = (size / max_count)
            marker_size = (norm_size * 60) + 15
            
            # Color by sentiment
            color = '#2ecc71' if topic_sentiments[topic_id] > 0.3 else ('#e74c3c' if topic_sentiments[topic_id] < -0.3 else '#95a5a6')
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(size=marker_size, color=color, opacity=0.6, line=dict(width=2, color='white')),
                text=[label],
                textposition="middle center",
                textfont=dict(size=9, color='white'),
                hovertext=f"<b>{label}</b><br>Size: {size} mentions<br>Sentiment: {topic_sentiments[topic_id]:.2f}",
                hoverinfo='text',
                showlegend=False
            ), row=row, col=col)
        
        fig.update_xaxes(title="Frequency", showgrid=True, row=row, col=col)
        fig.update_yaxes(title="Sentiment", showgrid=True, row=row, col=col)
    
    def _add_sentiment_analysis(self, fig, topics, topic_labels, metadata, row, col):
        """Add sentiment vs frequency scatter"""
        topic_data = {}
        
        for i, topic_id in enumerate(topics):
            if topic_id not in topic_data:
                topic_data[topic_id] = {'freq': 0, 'sentiment_scores': []}
            
            topic_data[topic_id]['freq'] += 1
            
            # Map sentiment
            if i < len(metadata):
                sent = metadata[i]['sentiment'].lower()
                if 'positive' in sent or 'joy' in sent:
                    topic_data[topic_id]['sentiment_scores'].append(1)
                elif 'negative' in sent or 'anger' in sent:
                    topic_data[topic_id]['sentiment_scores'].append(-1)
                else:
                    topic_data[topic_id]['sentiment_scores'].append(0)
        
        # Determine strict scaling factor
        max_freq = max([d['freq'] for d in topic_data.values()]) if topic_data else 1
        
        # Plot
        for topic_id, data in topic_data.items():
            if topic_id >= len(topic_labels):
                continue
            avg_sent = np.mean(data['sentiment_scores']) if data['sentiment_scores'] else 0
            label = topic_labels[topic_id]
            
            # Dynamic scaling
            norm_size = (data['freq'] / max_freq)
            marker_size = (norm_size * 60) + 15
            
            color = '#2ecc71' if avg_sent > 0.3 else ('#e74c3c' if avg_sent < -0.3 else '#95a5a6')
            
            fig.add_trace(go.Scatter(
                x=[data['freq']],
                y=[avg_sent],
                mode='markers+text',
                marker=dict(size=marker_size, color=color, opacity=0.7),
                text=[label],
                textposition="top center",
                textfont=dict(size=9),
                hovertext=f"{label}<br>Frequency: {data['freq']}<br>Sentiment: {avg_sent:.2f}",
                hoverinfo='text',
                showlegend=False
            ), row=row, col=col)
        
        fig.update_xaxes(title="Mentions", row=row, col=col)
        fig.update_yaxes(title="Avg Sentiment", range=[-1.2, 1.2], row=row, col=col)
    
    def _add_topic_timeline(self, fig, topics, topic_labels, metadata, row, col):
        """Add timeline showing topic evolution"""
        if not metadata:
            return
        
        # Group by topic
        for topic_id in set(topics):
            if topic_id >= len(topic_labels):
                continue
                
            timestamps = []
            for i, t in enumerate(topics):
                if t == topic_id and i < len(metadata):
                    timestamps.append(metadata[i]['timestamp'])
            
            if timestamps:
                label = topic_labels[topic_id]
                
                # Create histogram of occurrences over time
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=[topic_id] * len(timestamps),
                    mode='markers',
                    marker=dict(size=8, opacity=0.6),
                    name=label,
                    hovertext=[f"{label} at {t:.1f}s" for t in timestamps],
                    hoverinfo='text'
                ), row=row, col=col)
        
        fig.update_xaxes(title="Time (seconds)", row=row, col=col)
        fig.update_yaxes(title="Topic", row=row, col=col)
    
    def _add_top_words(self, fig, topic_words, topic_labels, topic_sizes, row, col):
        """Add bar chart of top words for largest topic"""
        if not topic_words or not topic_sizes:
            return
        
        # Find largest topic
        largest_topic = max(topic_sizes, key=topic_sizes.get)
        
        if largest_topic in topic_words:
            words_data = topic_words[largest_topic][:8]  # Top 8 words
            words = [w[0] for w in words_data]
            scores = [w[1] for w in words_data]
            
            label = topic_labels[largest_topic] if largest_topic < len(topic_labels) else f"Topic {largest_topic}"
            
            fig.add_trace(go.Bar(
                y=words,
                x=scores,
                orientation='h',
                marker=dict(color='#3498db'),
                hovertemplate='%{y}: %{x:.3f}<extra></extra>',
                showlegend=False
            ), row=row, col=col)
            
            fig.update_xaxes(title=f"Relevance Score ({label})", row=row, col=col)
            fig.update_yaxes(title="Words", row=row, col=col)
    
    def _add_session_summary_table(self, fig, per_session_results, metadata, row, col):
        """Add table summarizing each session's topic breakdown"""
        if not per_session_results:
            return
        
        # Prepare table data
        session_names = []
        n_segments = []
        top_topics = []
        methods = []
        
        for result in per_session_results:
            session_names.append(result['session_name'])
            
            # Count segments for this session
            seg_count = sum(1 for m in metadata if m['session'] == result['session_name'])
            n_segments.append(str(seg_count))
            
            # Get top topic
            if result['topic_labels']:
                top_topic = result['topic_labels'][0] if len(result['topic_labels']) > 0 else "N/A"
                top_topics.append(top_topic)
            else:
                top_topics.append("N/A")
            
            methods.append(result.get('method', 'N/A'))
        
        # Create table
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Session</b>', '<b>Segments</b>', '<b>Primary Topic</b>', '<b>Method</b>'],
                fill_color='#3498db',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[session_names, n_segments, top_topics, methods],
                fill_color=[['#ecf0f1', 'white'] * len(session_names)],
                align='left',
                font=dict(size=11)
            )
        ), row=row, col=col)

    def plot_semantic_comparison(self, parent, objects):
        print("\n=== plot_semantic_comparison CALLED ===")
        fig = self.get_semantic_comparison_figure(objects)
        print(f"Main figure generated, now embedding...")
        self._embed_plotly_figure(fig, parent)
        print(f"Main figure embedded")
        
        # Auto-display network graph in separate browser window
        print(f"\n=== CHECKING FOR NETWORK GRAPH ===")
        print(f"Has _last_network_fig: {hasattr(self, '_last_network_fig')}")
        
        if hasattr(self, '_last_network_fig'):
            print("Opening network graph in browser...")
            import time
            time.sleep(0.5)  # Small delay so windows don't overlap
            self._embed_plotly_figure(self._last_network_fig, parent)
            print("✓ Network graph displayed")
        else:
            print("No network graph available")
        
        # Display executive summary if available
        if hasattr(self, '_last_executive_summary_html') and self._last_executive_summary_html:
            print("\n=== EXECUTIVE SUMMARY AVAILABLE ===")
            # Store for consolidated report
            self._figures_for_report = getattr(self, '_figures_for_report', {})
            self._figures_for_report['Executive Summary'] = self._last_executive_summary_html
            print("✓ Executive summary stored for consolidated report")

    def get_gesture_comparison_figure(self, objects):
        # Aggregate Top Gestures
        all_gestures_found = set()
        names = []
        
        for obj in objects:
            names.append(obj['name'][:10])
            counts = obj['engine'].gesture_counts
            for g in counts:
                if g != "None" and g != "Unknown":
                    all_gestures_found.add(g)
                    
        if not all_gestures_found:
            return go.Figure()

        sorted_gestures = sorted(list(all_gestures_found))
        
        fig = go.Figure()
        
        for g in sorted_gestures:
            counts = []
            for obj in objects:
                counts.append(obj['engine'].gesture_counts.get(g, 0))
            
            fig.add_trace(go.Bar(
                x=names,
                y=counts,
                name=g
            ))
            
        fig.update_layout(
            title="Gesture Frequency Distribution",
            barmode='group',
            xaxis_title="Session",
            yaxis_title="Count",
            template='plotly_white',
            height=500
        )
        return fig

    def plot_gesture_comparison(self, parent, objects):
        # Fallback to existing matplotlib for tab view or use new plotly
        # Using new Plotly for consistency in tab view too
        fig = self.get_gesture_comparison_figure(objects)
        self._embed_plotly_figure(fig, parent)

    def _create_zeroshot_chart(self, aggregated_result):
        """
        Creates a bar chart + details list for Guided BERTopic Categories.
        Uses 'document_info' from BERTopic to find topics matching our guided seeds.
        """
        # 1. Check if we have the new Guided BERTopic data
        has_guided = 'document_info' in aggregated_result and 'guided_topics' in aggregated_result
        
        # 2. Check if we have legacy Zero-Shot data (fallback)
        has_legacy = 'zero_shot' in aggregated_result and aggregated_result['zero_shot']
        
        if not has_guided and not has_legacy:
            return None
            
        cat_to_texts = {}
        
        if has_guided:
            # --- NEW GUIDED LOGIC ---
            df = aggregated_result['document_info'] # Columns: Document, Topic, Name, ...
            seeds = aggregated_result['guided_topics']
            
            # Map Topic IDs to Texts
            # We want to group by our "Seeds" (e.g. "Bug") if the topic name matches
            # Or just list the dominant topics if they don't match (Discover Mode)
            
            for _, row in df.iterrows():
                topic_id = row['Topic']
                if topic_id == -1: continue # Skip outliers
                
                topic_name = row['Name'] # e.g. "0_bug_error"
                text = row['Document']
                
                # Use Clean CustomLabel if available (from LLM)
                if 'CustomLabel' in row and pd.notna(row['CustomLabel']):
                    category = str(row['CustomLabel'])
                else:
                    # Fallback: Default category is the topic name (cleaned)
                    category = " ".join(topic_name.split("_")[1:])
                
                # Try to match with a Seed Key for cleaner grouping
                matched_seed = None
                for seed in seeds:
                    # Simple check: does the topic name or category contain key words from the seed?
                    seed_keywords = seed.replace("/", " ").lower().split()
                    
                    # Check keywords against the category name
                    if any(k in category.lower() for k in seed_keywords if len(k) > 3):
                         matched_seed = seed
                         break
                    
            
                # If we found a seed match, use it. Otherwise use the clean category
                # User requested exact match with LLM labels, so we remove "Discovered:" prefix
                final_cat = matched_seed if matched_seed else category
                
                if final_cat not in cat_to_texts:
                    cat_to_texts[final_cat] = []
                cat_to_texts[final_cat].append(text)
                
        else:
            # --- LEGACY LOGIC ---
            results = aggregated_result['zero_shot']
            for r in results:
                cat = r['category']
                if cat not in cat_to_texts:
                    cat_to_texts[cat] = []
                cat_to_texts[cat].append(r['text'])
            
        # Sort categories by frequency
        sorted_cats = sorted(cat_to_texts.keys(), key=lambda c: len(cat_to_texts[c]), reverse=True)
        sorted_counts = [len(cat_to_texts[c]) for c in sorted_cats]
        
        # Generate Hover Text (Top 3 examples)
        hover_texts = []
        for cat in sorted_cats:
            texts = cat_to_texts[cat]
            # Pick up to 3 shortest/cleanest examples to avoid separate huge tooltips
            # Sort by length for readability
            examples = sorted(list(set(texts)), key=len)[:3] 
            example_str = "<br>".join([f"<i>'{t[:50]}...'</i>" for t in examples])
            hover_texts.append(f"<b>{cat}</b><br>Count: {len(texts)}<br><br>Examples:<br>{example_str}<extra></extra>")
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_cats, 
                y=sorted_counts,
                marker_color='#FF6B6B',  # Distinct color
                hovertemplate='%{text}',
                text=hover_texts # Pass custom hover HTML here
            )
        ])
        
        fig.update_layout(
            title="Intelligent Category Detection (Zero-Shot)",
            xaxis_title="Category",
            yaxis_title="Frequency",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        # Generate Detailed HTML List for Report
        html_details = "<div style='margin-top:20px; border-top:1px solid #eee; padding-top:10px;'>"
        html_details += "<h3>Category Details</h3>"
        
        for cat in sorted_cats:
            texts = cat_to_texts[cat]
            html_details += f"""
            <details style="margin-bottom:10px; background:#f9f9f9; padding:5px; border-radius:5px;">
                <summary style="cursor:pointer; font-weight:bold; color:#333;">
                    {cat} ({len(texts)})
                </summary>
                <div style="padding:10px; padding-left:20px; font-size:0.9em; color:#555;">
                    <ul style="margin:5px 0;">
            """
            for t in texts:
                html_details += f"<li style='margin-bottom:3px;'>{t}</li>"
                
            html_details += """
                    </ul>
                </div>
            </details>
            """
        html_details += "</div>"
        
        return (fig, html_details)

    def _create_interactive_clusters_chart(self, aggregated_result):
        """Creates interactive Scatter Plot with Dropdown for K=(3,5,8,10)"""
        if not aggregated_result or 'semantic_clusters' not in aggregated_result or not aggregated_result['semantic_clusters']:
            return None
            
        cluster_data = aggregated_result['semantic_clusters']
        if not cluster_data:
            return None
            
        fig = go.Figure()
        
        # We need to add all traces for all K values, but set visible=False for most
        # Dropdown will toggle visibility
        
        # Buttons configuration
        buttons = []
        
        # Colors for clusters (safe palette)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c', '#34495e', '#7f8c8d', '#c0392b']
        
        # Track trace indices to toggle visibility efficiently
        current_trace_idx = 0
        
        # Sort K values
        k_values = sorted([k for k in cluster_data.keys() if isinstance(k, int)])
        
        if not k_values:
            return None
            
        first_k = k_values[0]
        
        for k in k_values:
            data = cluster_data[k]
            labels = data['labels']
            coords = np.array(data['coords'])
            names = data['cluster_names'] 
            
            # For this K, add one trace PER cluster to allow legend
            # But to keep dropdown simple, we might just add ONE trace with different colors?
            # Actually, one trace per K is easier for "visible" toggling
            
            # Using discrete colors in a single trace is cleaner for large N
            trace_colors = [colors[l % len(colors)] for l in labels]
            trace_hover = [f"Cluster {l+1}: {names.get(str(l), names.get(l, 'Unknown'))}" for l in labels]
            
            visible = (k == first_k)
            
            fig.add_trace(go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers',
                marker=dict(size=8, color=trace_colors, opacity=0.7, line=dict(width=1, color='white')),
                text=trace_hover,
                hoverinfo='text',
                name=f"{k} Clusters",
                visible=visible
            ))
            
            # Add centroids (optional, skipping for clean look)
            
            # Create button for this K
            # We have N traces (one per K). 
            # If K=3 is at index 0, K=5 is at index 1...
            visibility = [False] * len(k_values)
            visibility[current_trace_idx] = True
            
            buttons.append(dict(
                label=f"{k} Clusters",
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Semantic Structure ({k} Clusters)"}]
            ))
            current_trace_idx += 1
            
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    x=1.15,
                    y=1.15
                )
            ],
            title=f"Semantic Structure ({first_k} Clusters)",
            xaxis_title="Semantic Dimension 1",
            yaxis_title="Semantic Dimension 2",
            template='plotly_white',
            height=500,
            showlegend=False,
            margin=dict(r=150) # Space for dropdown
        )
        
        return fig

    def open_consolidated_report(self, figures):
        """Generates and opens a single HTML report with all figures"""
        import tempfile
        
        # HTML Header
        html_content = ["""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Comparison Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 95%; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
                .chart-section { margin-bottom: 50px; border: 1px solid #eee; padding: 15px; border-radius: 5px; }
                h2 { color: #0066cc; margin-top: 0; }
                .timestamp { color: #888; font-size: 0.9em; margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analysis Comparison Report</h1>
                <div class="timestamp">Generated on: """ + str(np.datetime64('now')) + """</div>
        """]
        
        # Embed each figure
        # Embed each figure
        first_fig = True
        for title, fig_data in figures.items():
            if fig_data is None: continue
            
            # Handle tuple return (fig, extra_html_details)
            extra_html = ""
            if isinstance(fig_data, tuple):
                fig = fig_data[0]
                extra_html = fig_data[1]
            else:
                fig = fig_data
            
            # Embed JS library in the first figure ONLY to keep file size reasonable but robust
            # This fixes "empty chart" issues if CDN fails
            include_js = True if first_fig else False
            
            # Check for DataMapPlot explicitly
            if title == 'Interactive DataMapPlot':
                # DataMapPlot returns a full HTML page. We MUST iframe it to prevent CSS collisions
                import html
                escaped_html = html.escape(str(fig))
                div_str = f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 800px; border: none;"></iframe>'
            
            # Charts that need iframe embedding to prevent layout conflicts
            elif title in ["Semantic & Sentiment", "Semantic Structure (Interactive)", 
                          "BERTopic Top Words (Bar Chart)", "BERTopic Document Clusters", 
                          "BERTopic Intertopic Map"] and hasattr(fig, 'to_html'):
                import tempfile
                import os
                # Create safe filename from title
                safe_name = title.replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and').lower()
                temp_dir = tempfile.gettempdir()
                chart_path = os.path.join(temp_dir, f'{safe_name}_standalone.html')
                with open(chart_path, 'w', encoding='utf-8') as sf:
                    sf.write(fig.to_html(include_plotlyjs=True, full_html=True))
                
                # Different heights for different charts
                height_map = {
                    "Semantic & Sentiment": 1700,
                    "Semantic Structure (Interactive)": 800,
                    "BERTopic Top Words (Bar Chart)": 700,
                    "BERTopic Document Clusters": 800,
                    "BERTopic Intertopic Map": 800
                }
                iframe_height = height_map.get(title, 800)
                
                # Embed as iframe pointing to the file
                file_url = 'file:///' + chart_path.replace('\\', '/')
                div_str = f'<iframe src="{file_url}" style="width: 100%; height: {iframe_height}px; border: 1px solid #ddd; border-radius: 5px;"></iframe>'
                print(f"  [Iframe Embed] {title} -> {chart_path}")
            
            # Check if fig is a Plotly figure object
            elif hasattr(fig, 'to_html'):
                div_str = fig.to_html(include_plotlyjs=include_js, full_html=False, config={'responsive': True})
            else:
                # Assume it's raw HTML content if not a Plotly figure
                div_str = str(fig)
            
            first_fig = False
            
            html_content.append(f"""
                <div class="chart-section">
                    <h2>{title}</h2>
                    {div_str}
                    {extra_html}
                </div>
            """)
            
        html_content.append("""
            </div>
        </body>
        </html>
        """)
        
        # Write file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
            f.write("\n".join(html_content))
            path = f.name
            
        # FORCE NEW WINDOW
        url = 'file:///' + path.replace('\\', '/')
        self._force_new_window(url)


    def compare_selected(self, lib_window=None):
        selected = [f for f, var in self.lib_selections.items() if var.get()]
        if len(selected) < 1:
            messagebox.showwarning("Compare", "Please select at least 2 files to compare.")
            return

        if lib_window:
            lib_window.destroy()

        # Loading Progress
        progress_win = ctk.CTkToplevel(self)
        progress_win.title("Generating Report...")
        progress_win.geometry("400x150")
        progress_win.attributes('-topmost', True)
        
        lbl = ctk.CTkLabel(progress_win, text="Loading sessions...", font=("Arial", 14))
        lbl.pack(pady=20)
        progress = ctk.CTkProgressBar(progress_win)
        progress.pack(pady=10, padx=20)
        progress.set(0)
        progress_win.update()
        
        # Load Data
        lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Library") 
        loaded_objects = []
        
        try:
            for i, f in enumerate(selected):
                path = os.path.join(lib_dir, f)
                eng = VideoAnalysisEngine(GESTURE_MODEL_PATH)
                eng.load_analysis(path) 
                loaded_objects.append({'name': f, 'engine': eng})
                progress.set((i+1)/len(selected))
                progress_win.update()
        except Exception as e:
            progress_win.destroy()
            messagebox.showerror("Error", f"Failed to load: {e}")
            return
            
            
        lbl.configure(text="Generating Graphs...")
        progress_win.update()
        
        # Collect Figures
        figures = {}
        
        # Check if we have hand data in any session
        any_hands = any(obj['engine'].has_hands for obj in loaded_objects)
        if not any_hands:
            print("Notice: No hand data detected. Generating Audio-Only report.")
        
        try:
            # Audio Wheel will be generated AFTER topic analysis to include discovered topics
            
            # 1. Standard Metrics (Hand Dependent)
            if any_hands:
                figures['Performance Comparison'] = self.get_comparison_metrics_figure(loaded_objects)
                figures['Gesture Frequencies'] = self.get_gesture_comparison_figure(loaded_objects)
            
                # 2. Spatial (Hand Dependent)
                figures['Spatial Occupancy Density'] = self.get_spatial_comparison_figure(loaded_objects)
            else:
                figures['Visualization Note'] = "<b> Audio Only Analysis </b> <br> Hand tracking metrics are hidden because no hand data was found in the selected sessions."
            
            # 3. Semantic (Audio - Always Run)
            try:
                print("Generating Semantic & Sentiment figure...")
                semantic_fig = self.get_semantic_comparison_figure(loaded_objects)
                if semantic_fig is not None:
                    figures['Semantic & Sentiment'] = semantic_fig
                    print("✓ Semantic & Sentiment figure generated successfully")
                else:
                    print("⚠️  Semantic & Sentiment figure returned None")
                    figures['Semantic & Sentiment'] = "<b>Error:</b> Semantic comparison figure generation returned None"
            except Exception as e:
                print(f"⚠️  Semantic & Sentiment figure generation failed: {e}")
                import traceback
                traceback.print_exc()
                figures['Semantic & Sentiment'] = f"<b>Error:</b> {str(e)}"
            
            # 3b. Interactive Clusters (K-Means)
            if hasattr(self, '_last_aggregated_result'):
                cluster_fig = self._create_interactive_clusters_chart(self._last_aggregated_result)
                if cluster_fig:
                     figures['Semantic Structure (Interactive)'] = cluster_fig
                     print("✓ Added Interactive Clusters chart to report")
            
            
            # 3c. Zero-Shot Categories
            if hasattr(self, '_last_aggregated_result'):
                zs_fig = self._create_zeroshot_chart(self._last_aggregated_result)
                if zs_fig:
                     figures['Zero-Shot Categories'] = zs_fig
                     print("✓ Added Zero-Shot chart to report")
            
            # 3b. Network Graph (if available)
            if hasattr(self, '_last_network_fig') and self._last_network_fig is not None:
                figures['Topic Network'] = self._last_network_fig
                print("✓ Added network graph to report")
                
            # 3d. BERTopic Native Visualizations (if enabled)
            if hasattr(self, '_last_aggregated_result'):
                res = self._last_aggregated_result
                if res.get('viz_intertopic'):
                    figures['BERTopic Intertopic Map'] = res['viz_intertopic']
                    print("✓ Added Intertopic Map to report")
                
                if res.get('viz_documents'):
                    figures['BERTopic Document Clusters'] = res['viz_documents']
                    print("✓ Added Document Clusters to report")
                    
                if res.get('viz_barchart'):
                    figures['BERTopic Top Words (Bar Chart)'] = res['viz_barchart']
                    print("✓ Added Bar Chart to report")
                
                if res.get('viz_datamap'):
                    figures['Interactive DataMapPlot'] = res['viz_datamap']
                    print("✓ Added DataMapPlot to report")
            
            # 3e. Executive Summary (Topic Takeaways)
            if hasattr(self, '_last_executive_summary_html') and self._last_executive_summary_html:
                figures['Executive Summary'] = self._last_executive_summary_html
                print("Executive Summary added to report")
            
            # 3f. Audio Wheel Analyzer (Generated AFTER topic analysis to include topics)
            # We need to map the aggregated topic results back to the individual session transcripts
            # The aggregation logic used a filter: if text and len(text) > 10
            if hasattr(self, '_last_aggregated_result'):
                res = self._last_aggregated_result
                
                # Get the sequence of topic IDs assigned to the filtered documents
                # 'topics' is a list of topic_ids corresponding to the filtered all_segments list
                assigned_topic_ids = res.get('topics', [])
                topic_label_map = res.get('topic_label_map', {})
                topic_labels_list = res.get('topic_labels', [])
                
                # Global counter for the filtered segments
                global_filtered_idx = 0
                
                # Collect generated wheels
                audio_wheels = []
                
                for obj in loaded_objects:
                    engine = obj['engine']
                    if hasattr(engine, 'transcript') and engine.transcript:
                        try:
                            updated_transcript = []
                            
                            # Iterate transcript and match with topic assignments using the same filter
                            for start, end, text, _ in engine.transcript:
                                topic_label = "General"  # Default
                                
                                # Check if this segment was included in topic modeling
                                if text and len(text) > 10:
                                    # It was included, so grab the next topic from the list
                                    if global_filtered_idx < len(assigned_topic_ids):
                                        topic_id = assigned_topic_ids[global_filtered_idx]
                                        
                                        # Resolve label
                                        if topic_label_map and topic_id in topic_label_map:
                                            topic_label = topic_label_map[topic_id]
                                        elif 0 <= topic_id < len(topic_labels_list):
                                            topic_label = topic_labels_list[topic_id]
                                        elif topic_id == -1:
                                            topic_label = "General"
                                        else:
                                            topic_label = f"Topic {topic_id}"
                                            
                                        global_filtered_idx += 1
                                
                                updated_transcript.append((start, end, text, topic_label))
                                
                            # Get video title from path
                            video_title = os.path.splitext(os.path.basename(engine.video_path))[0]
                            
                            # Generate wheel with topics
                            from audio_wheel_analyzer import generate_audio_wheel_html
                            wheel_path = generate_audio_wheel_html(
                                engine.video_path,
                                updated_transcript,
                                video_title=video_title
                            )
                            
                            if wheel_path:
                                file_url = 'file:///' + wheel_path.replace('\\', '/')
                                audio_wheel_fig = f'<iframe src="{file_url}" style="width: 100%; height: 900px; border: 1px solid #ddd; border-radius: 10px;"></iframe>'
                                audio_wheels.append((video_title, audio_wheel_fig))
                                
                        except Exception as e:
                            print(f"Audio Wheel generation failed for {engine.video_path}: {e}")
                            import traceback
                            traceback.print_exc()

                # Add all collected wheels to figures
                if audio_wheels:
                    from collections import OrderedDict
                    new_figures = OrderedDict()
                    
                    for title, iframe in audio_wheels:
                        # Use unique keys if multiple, otherwise generic
                        key_name = "Interactive Audio Analyzer" if len(audio_wheels) == 1 else f"Audio Analysis: {title}"
                        new_figures[key_name] = iframe
                        
                    new_figures.update(figures)
                    figures = new_figures
                    print(f"Added {len(audio_wheels)} Audio Wheel(s) to report (with synced topic labels)")

            # 4. Episode Analysis
            from episode_comparison import get_episode_figures
            ep_figs = get_episode_figures(loaded_objects)
            figures.update(ep_figs)
            
            # Open Report
            lbl.configure(text="Opening Browser...")
            progress_win.update()
            self.open_consolidated_report(figures)
            
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {e}")
            print(e)
            
        progress_win.destroy()

if __name__ == "__main__":
    app = ModernHandTrackerApp()
    app.mainloop()
