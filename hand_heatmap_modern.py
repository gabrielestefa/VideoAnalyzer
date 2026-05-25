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

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL_PATH = os.path.join(_BASE_DIR, "Models", "gesture_recognizer.task")

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
            lib_img = ctk.CTkImage(light_image=Image.open(os.path.join(_BASE_DIR, "Media", "Library.png")),
                                   dark_image=Image.open(os.path.join(_BASE_DIR, "Media", "Library.png")),
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
        """Render the sampled frame at position idx onto the main canvas."""
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
        """Scrub the video preview to the frame corresponding to slider position val."""
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
                # entry[4] is the signed sentiment score in [-1, 1] if classifier ran
                sentiment_score = float(entry[4]) if len(entry) > 4 else None
                timestamp = entry[0] if len(entry) > 0 else 0

                if text and len(text) > 10:
                    all_segments.append(text)
                    segment_metadata.append({
                        'session': session_labels[obj_idx],
                        'session_idx': obj_idx,
                        'timestamp': timestamp,
                        'sentiment': sentiment,
                        'sentiment_score': sentiment_score,
                        'text': text,
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
                topic_sizes_local = aggregated_result.get('topic_sizes', {})
                self._last_executive_summary_html = self._create_executive_summary_html(
                    topic_summaries, topic_labels, topic_words, topic_sizes_local
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
        Knowledge Graph: Topics (anchors) + Keywords (concepts) + similarity edges.

        Visual encoding:
          - Topic nodes: large, coloured (per-topic Wong colour), labelled
          - Keyword nodes: small grey dots, labelled at smaller font
          - Topic-keyword edges: thin light grey ("definition")
          - Topic-topic edges: thicker orange dashed ("semantic similarity")
          - Legend explains the three layers

        Keywords are limited to top-K per topic, with K adapting to topic count
        so the canvas doesn't turn into a hairball.
        """
        import networkx as nx
        import numpy as np
        from viz_palette import WONG, SEMANTIC

        topics = aggregated_result['topics']
        topic_labels = aggregated_result['topic_labels']
        topic_words = aggregated_result['topic_words']
        topic_sizes = aggregated_result['topic_sizes']
        topic_coords = aggregated_result.get('topic_coords', {})

        # Adaptive keyword limit: more topics → fewer keywords per topic (avoid hairball)
        n_topics_est = len([t for t in topic_sizes.keys() if t != -1])
        if n_topics_est <= 4:
            kw_per_topic = 6
        elif n_topics_est <= 8:
            kw_per_topic = 4
        else:
            kw_per_topic = 3

        G = nx.Graph()
        topic_nodes, fixed_positions = [], {}
        topic_color_map = {}

        # 1. Topic nodes — use semantic UMAP coords if available
        has_coords = False
        if topic_coords:
            raw_coords = np.array([c for tid, c in topic_coords.items() if tid < len(topic_labels)])
            if len(raw_coords) > 0:
                has_coords = True
                min_c, max_c = raw_coords.min(axis=0), raw_coords.max(axis=0)
                range_c = np.where(max_c - min_c == 0, 1, max_c - min_c)
                for i, (tid, coord) in enumerate(topic_coords.items()):
                    if tid < len(topic_labels) and tid != -1:
                        norm = ((np.array(coord) - min_c) / range_c - 0.5) * 2.0
                        color = WONG[(i + 1) % len(WONG)]
                        topic_color_map[tid] = color
                        node_id = f"T_{tid}"
                        G.add_node(node_id, node_type='topic',
                                   label=topic_labels[tid],
                                   size=topic_sizes.get(tid, 10), color=color)
                        fixed_positions[node_id] = norm.tolist()
                        topic_nodes.append(node_id)

        if not has_coords:
            for i, (tid, label) in enumerate(enumerate(topic_labels)):
                color = WONG[(i + 1) % len(WONG)]
                topic_color_map[tid] = color
                node_id = f"T_{tid}"
                G.add_node(node_id, node_type='topic', label=label,
                           size=topic_sizes.get(tid, 10), color=color)
                topic_nodes.append(node_id)

        # 2. Keyword nodes — limit per topic and filter low-weight words
        for tid, words in topic_words.items():
            node_id = f"T_{tid}"
            if node_id not in G.nodes or tid == -1:
                continue
            # Words may be (word, score) tuples or plain strings
            entries = words[:kw_per_topic]
            for entry in entries:
                if isinstance(entry, (list, tuple)):
                    word, score = entry[0], float(entry[1])
                else:
                    word, score = entry, 1.0
                if not word or len(word) < 2:
                    continue
                kw_node_id = f"K_{word}"
                if kw_node_id not in G.nodes:
                    G.add_node(kw_node_id, node_type='keyword', label=word,
                               size=4, color="#9aa3ad", weight=score)
                G.add_edge(node_id, kw_node_id, weight=max(score, 0.1) * 5,
                           edge_type='definition')

        # 3. Topic-topic similarity edges (only with coords)
        if len(topic_nodes) > 1:
            import itertools
            for t1, t2 in itertools.combinations(topic_nodes, 2):
                if t1 in fixed_positions and t2 in fixed_positions:
                    dist = float(np.linalg.norm(np.array(fixed_positions[t1]) -
                                                np.array(fixed_positions[t2])))
                    if dist < 0.4:
                        G.add_edge(t1, t2, weight=(1.0 - dist) * 2,
                                   edge_type='similarity', distance=dist)

        # 4. Layout
        print(f"  [Knowledge Graph] Calculating layout for {len(G.nodes)} nodes "
              f"({kw_per_topic} keywords/topic)...")
        if not G.nodes:
            print("  [Warning] Graph has no nodes, skipping layout.")
            return go.Figure()
        if fixed_positions:
            pos = nx.spring_layout(G, pos=fixed_positions,
                                   fixed=fixed_positions.keys(),
                                   k=0.18, iterations=60, seed=42)
        else:
            pos = nx.spring_layout(G, k=0.3, iterations=60, seed=42)

        # 5. Build edge traces — split by type for distinct styling
        def_edge_x, def_edge_y = [], []
        sim_edge_x, sim_edge_y = [], []
        for u, v, edata in G.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            if edata.get('edge_type') == 'similarity':
                sim_edge_x.extend([x0, x1, None]); sim_edge_y.extend([y0, y1, None])
            else:
                def_edge_x.extend([x0, x1, None]); def_edge_y.extend([y0, y1, None])

        def_edges = go.Scatter(
            x=def_edge_x, y=def_edge_y,
            line=dict(width=0.6, color='#dde2e7'),
            hoverinfo='none', mode='lines',
            name='topic → keyword', showlegend=True,
        )
        sim_edges = go.Scatter(
            x=sim_edge_x, y=sim_edge_y,
            line=dict(width=2.0, color=SEMANTIC["highlight"], dash='dot'),
            hoverinfo='none', mode='lines',
            name='topic ↔ topic (semantic similarity)', showlegend=True,
        )

        # 6. Build node traces — split for legend clarity
        topic_x, topic_y, topic_text, topic_sz, topic_col, topic_hover = [], [], [], [], [], []
        kw_x, kw_y, kw_text, kw_hover = [], [], [], []

        for node in G.nodes():
            x, y = pos[node]
            data = G.nodes[node]
            if data['node_type'] == 'topic':
                topic_x.append(x); topic_y.append(y)
                topic_text.append(data['label'])
                topic_sz.append(18 + np.log(data['size'] + 1) * 9)
                topic_col.append(data['color'])
                topic_hover.append(f"<b>{data['label']}</b><br>"
                                   f"Segments: {data['size']}")
            else:
                kw_x.append(x); kw_y.append(y)
                kw_text.append(data['label'])
                kw_hover.append(f"Keyword: <b>{data['label']}</b>")

        topic_trace = go.Scatter(
            x=topic_x, y=topic_y, mode='markers+text',
            text=topic_text, textposition="top center",
            textfont=dict(family='Inter, Segoe UI, sans-serif', size=12, color='#1a1a1a'),
            marker=dict(color=topic_col, size=topic_sz,
                        line=dict(width=2, color='white'),
                        opacity=0.92),
            hovertext=topic_hover, hoverinfo='text',
            name='topic', showlegend=True,
        )
        keyword_trace = go.Scatter(
            x=kw_x, y=kw_y, mode='markers+text',
            text=kw_text, textposition="top center",
            textfont=dict(family='Inter, Segoe UI, sans-serif', size=9, color='#5a6470'),
            marker=dict(color='#aab1b8', size=7, line=dict(width=0.5, color='white')),
            hovertext=kw_hover, hoverinfo='text',
            name='keyword', showlegend=True,
        )

        # 7. Compose
        fig = go.Figure(data=[def_edges, sim_edges, keyword_trace, topic_trace])

        n_topics = len(topic_x)
        n_kw = len(kw_x)
        insight = (f"{n_topics} topics · {n_kw} keywords shown · "
                   f"orange dotted lines link semantically related topics")

        fig.update_layout(
            title=dict(
                text=("<b>Topic Knowledge Graph</b><br>"
                      f"<span style='font-size:13px;color:#666;font-weight:normal'>{insight}</span>"),
                font=dict(size=20, family="Inter, Segoe UI, sans-serif", color="#1a1a1a"),
                x=0.02, xanchor="left",
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.05,
                        xanchor="center", x=0.5,
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#e5e5e5", borderwidth=1),
            hovermode='closest',
            margin=dict(b=80, l=20, r=20, t=90),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white', paper_bgcolor='white',
            height=700,
        )
        return fig
    
    def _create_executive_summary_html(
        self,
        topic_summaries: dict,
        topic_labels: dict,
        topic_words: dict,
        topic_sizes: dict = None,
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
                grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
                gap: 20px;
                padding: 20px;
            }
            .summary-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 20px;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                max-height: 420px;
                overflow-y: auto;
                position: relative;
            }
            .coherence-row {
                display: flex; flex-wrap: wrap; gap: 8px;
                margin: 4px 0 12px 0; align-items: center;
            }
            .badge {
                background: rgba(255,255,255,0.18);
                border: 1px solid rgba(255,255,255,0.32);
                border-radius: 999px;
                padding: 2px 10px;
                font-size: 0.78em;
                letter-spacing: 0.2px;
            }
            .badge.strong   { background: rgba(0,158,115,0.55);  border-color: rgba(0,158,115,0.7); }
            .badge.moderate { background: rgba(230,159,0,0.55);  border-color: rgba(230,159,0,0.7); }
            .badge.weak     { background: rgba(204,121,167,0.55); border-color: rgba(204,121,167,0.7); }
            .badge.stars { font-family: monospace; letter-spacing: 1px; }
            
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
        
        # Coherence indicators: derive a quality signal from segment count.
        # Larger topics are statistically more robust; tiny topics are likely noise.
        sizes = topic_sizes or {}
        total_segments = sum(sizes.values()) or 1
        max_size = max(sizes.values()) if sizes else 1

        def _coherence_meta(tid):
            n = sizes.get(tid, 0)
            pct = (n / total_segments) * 100 if total_segments else 0
            # 5-star scale: prominence relative to the biggest topic.
            ratio = n / max_size if max_size else 0
            stars_n = max(1, min(5, int(round(ratio * 5))))
            stars = "★" * stars_n + "☆" * (5 - stars_n)
            if n >= 10 and pct >= 10:
                strength = "strong"
            elif n >= 5:
                strength = "moderate"
            else:
                strength = "weak"
            return n, pct, stars, strength

        for tid, takeaways in topic_summaries.items():
            label = topic_labels.get(tid, f"Topic {tid}")
            keywords = []
            for w in topic_words.get(tid, [])[:5]:
                keywords.append(w[0] if isinstance(w, (list, tuple)) else w)
            keywords_str = ", ".join(keywords) if keywords else "No keywords"

            takeaway_items = "".join([f"<li>{t}</li>" for t in takeaways])

            n, pct, stars, strength = _coherence_meta(tid)
            coherence_row = (
                f'<div class="coherence-row" title="Indicators derived from topic prominence">'
                f'  <span class="badge {strength}">{n} segments</span>'
                f'  <span class="badge">{pct:.0f}% of corpus</span>'
                f'  <span class="badge stars" title="Quality (prominence-based)">{stars}</span>'
                f'</div>'
            )

            html_parts.append(f"""
            <div class="summary-card">
                <button class="copy-btn" onclick="copyCard(this)">Copy</button>
                <h3>{label}</h3>
                {coherence_row}
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
        REWRITTEN v3: Research-driven layout with semantic colours, insight-stating titles,
        topic labels (not IDs) on axes, and reference lines for quick interpretation.

        Layout (3x2):
          Row 1: Topic landscape (frequency × sentiment), Topic-by-topic sentiment ranking
          Row 2: Per-session distribution / topic frequency, Topic timeline (LABELS as y)
          Row 3: Top keywords for dominant topic, Session summary table
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from viz_palette import WONG, SEMANTIC, apply_clean_layout

        # Extract data
        topics = aggregated['topics']
        topic_label_map = aggregated.get('topic_label_map', {})
        topic_sizes = aggregated['topic_sizes']
        topic_words = aggregated.get('topic_words', {})

        unique_topics = sorted([t for t in set(topics) if t != -1])
        print(f"  [Dual-View V3] Creating 3x2 figure for {len(unique_topics)} topics")

        def trim(label, n=28):
            return label if len(label) <= n else label[:n - 1] + "…"

        # Precompute topic sentiments. Prefer the numeric `sentiment_score` field
        # produced by the real classifier; fall back to keyword matching for legacy data.
        def _seg_score(m):
            s = m.get('sentiment_score')
            if s is not None:
                return float(s)
            label = (m.get('sentiment') or "").lower()
            if 'pos' in label or 'joy' in label:
                return 1.0
            if any(k in label for k in ('neg', 'anger', 'sadness')):
                return -1.0
            return 0.0

        topic_sentiments = {}
        for tid in unique_topics:
            scores = [_seg_score(metadata[i])
                      for i, t in enumerate(topics) if t == tid and i < len(metadata)]
            topic_sentiments[tid] = sum(scores) / len(scores) if scores else 0

        # Build a stable colour mapping: each topic gets one Wong colour, reused everywhere
        topic_color = {tid: WONG[(i + 1) % len(WONG)] for i, tid in enumerate(unique_topics)}

        # Subplot grid — descriptive subtitles that state what to look at
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Topic landscape — how often vs. how positive",
                "Topic sentiment ranking — which topics skew positive or negative",
                "Topic frequency" if len(set(m['session'] for m in metadata)) <= 1
                                  else "Topic distribution across sessions",
                "Topic timeline — when each theme was discussed",
                "Top keywords for the most-discussed topic",
                "Per-session summary"
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'table'}],
            ],
            vertical_spacing=0.13, horizontal_spacing=0.12,
            row_heights=[0.34, 0.36, 0.30],
        )

        # === ROW 1, COL 1: Topic landscape (frequency × sentiment) ===
        try:
            max_count = max(topic_sizes.values()) if topic_sizes else 1
            for tid in unique_topics:
                label = topic_label_map.get(tid, f"Topic {tid}")
                count = topic_sizes.get(tid, 0)
                sent = topic_sentiments.get(tid, 0)
                # Bubble area encodes count; colour encodes sentiment direction (diverging)
                color = (SEMANTIC["positive"] if sent > 0.3
                         else SEMANTIC["negative"] if sent < -0.3
                         else SEMANTIC["neutral"])
                fig.add_trace(go.Scatter(
                    x=[count], y=[sent], mode='markers+text',
                    marker=dict(size=(count / max_count) * 45 + 14, color=color,
                                opacity=0.78, line=dict(color='white', width=1.5)),
                    text=[trim(label, 22)], textposition="top center",
                    textfont=dict(size=10, color="#222"),
                    hovertemplate=(f"<b>{label}</b><br>Mentions: {count}"
                                   f"<br>Avg sentiment: {sent:+.2f}<extra></extra>"),
                    showlegend=False,
                ), row=1, col=1)
            # Zero-line baseline — sentiment neutrality
            fig.add_hline(y=0, line=dict(color="#aaa", width=1, dash="dot"), row=1, col=1)
            fig.update_xaxes(title="Mention count (segments)", row=1, col=1)
            fig.update_yaxes(title="Avg sentiment  ←neg | pos→",
                             range=[-1.15, 1.15], row=1, col=1)
        except Exception as e:
            print(f"  [Error] Landscape: {e}")

        # === ROW 1, COL 2: Sentiment ranking bar chart (horizontal, ordered) ===
        try:
            # Sort topics by sentiment so the reader sees the most positive at top
            ranked = sorted(unique_topics, key=lambda t: topic_sentiments.get(t, 0))
            y_labels = [trim(topic_label_map.get(t, f"Topic {t}"), 30) for t in ranked]
            sentiments = [topic_sentiments.get(t, 0) for t in ranked]
            colors = [SEMANTIC["positive"] if s > 0.3
                      else SEMANTIC["negative"] if s < -0.3
                      else SEMANTIC["neutral"] for s in sentiments]
            fig.add_trace(go.Bar(
                x=sentiments, y=y_labels, orientation='h',
                marker=dict(color=colors), showlegend=False,
                hovertemplate="<b>%{y}</b><br>Avg sentiment: %{x:+.2f}<extra></extra>",
            ), row=1, col=2)
            fig.add_vline(x=0, line=dict(color="#666", width=1), row=1, col=2)
            fig.update_xaxes(title="Avg sentiment per topic",
                             range=[-1.05, 1.05], row=1, col=2)
            fig.update_yaxes(title="", row=1, col=2)
        except Exception as e:
            print(f"  [Error] Sentiment ranking: {e}")

        # === ROW 2, COL 1: Per-session distribution or single-session frequency ===
        try:
            session_names_unique = sorted(set(m['session'] for m in metadata))
            n_sessions = len(session_names_unique)
            if n_sessions > 1:
                for tid in unique_topics:
                    label = trim(topic_label_map.get(tid, f"Topic {tid}"), 28)
                    counts_per_session = [
                        sum(1 for i, m in enumerate(metadata)
                            if m['session'] == sess and i < len(topics) and topics[i] == tid)
                        for sess in session_names_unique
                    ]
                    fig.add_trace(go.Bar(
                        x=session_names_unique, y=counts_per_session,
                        name=label, marker_color=topic_color[tid],
                        hovertemplate=f"<b>{label}</b><br>Session %{{x}}: %{{y}} segments<extra></extra>",
                    ), row=2, col=1)
                fig.update_layout(barmode='stack')
                fig.update_xaxes(title="Session", row=2, col=1)
                fig.update_yaxes(title="Segments (count)", row=2, col=1)
            else:
                # Sort descending so the eye lands on the biggest topic first
                ranked = sorted(unique_topics, key=lambda t: -topic_sizes.get(t, 0))
                labels = [trim(topic_label_map.get(tid, f"Topic{tid}"), 28) for tid in ranked]
                counts = [topic_sizes.get(tid, 0) for tid in ranked]
                colors = [topic_color[tid] for tid in ranked]
                mean_count = sum(counts) / len(counts) if counts else 0
                fig.add_trace(go.Bar(
                    x=labels, y=counts, marker_color=colors, showlegend=False,
                    hovertemplate="<b>%{x}</b><br>Segments: %{y}<extra></extra>",
                ), row=2, col=1)
                # Reference line at mean so reader sees which topics are above/below average
                fig.add_hline(y=mean_count, line=dict(color="#888", dash="dash", width=1),
                              annotation_text=f"avg = {mean_count:.1f}",
                              annotation_position="top right", row=2, col=1)
                fig.update_xaxes(title="Topic", tickangle=-35, row=2, col=1)
                fig.update_yaxes(title="Segments (count)", row=2, col=1)
        except Exception as e:
            print(f"  [Error] Topic distribution: {e}")

        # === ROW 2, COL 2: Timeline — Y-axis uses TOPIC LABELS, not IDs ===
        try:
            # Order topics by total frequency so busiest topics sit on top
            ordered = sorted(unique_topics, key=lambda t: topic_sizes.get(t, 0))
            y_index = {tid: i for i, tid in enumerate(ordered)}
            y_tick_text = [trim(topic_label_map.get(t, f"Topic {t}"), 26) for t in ordered]

            for tid in ordered:
                label = topic_label_map.get(tid, f"Topic {tid}")
                times = [metadata[i]['timestamp']
                         for i, t in enumerate(topics) if t == tid and i < len(metadata)]
                if times:
                    fig.add_trace(go.Scatter(
                        x=times, y=[y_index[tid]] * len(times),
                        mode='markers',
                        marker=dict(size=7, opacity=0.7, color=topic_color[tid],
                                    line=dict(color='white', width=0.5)),
                        name=trim(label, 24), showlegend=False,
                        hovertemplate=f"<b>{label}</b><br>t = %{{x:.1f}}s<extra></extra>",
                    ), row=2, col=2)

            fig.update_xaxes(title="Time in video (seconds)", row=2, col=2)
            fig.update_yaxes(
                title="", tickmode='array',
                tickvals=list(range(len(ordered))), ticktext=y_tick_text,
                row=2, col=2,
            )
        except Exception as e:
            print(f"  [Error] Timeline: {e}")

        # === ROW 3, COL 1: Top words (use real weights if BERTopic provided them) ===
        try:
            if topic_sizes:
                top_topic = max(topic_sizes, key=topic_sizes.get)
                words_raw = topic_words.get(top_topic, [])[:10]
                # Words may be plain strings or (word, weight) tuples — handle both
                if words_raw and isinstance(words_raw[0], (tuple, list)):
                    word_labels = [w[0] for w in words_raw]
                    word_weights = [float(w[1]) for w in words_raw]
                else:
                    word_labels = list(words_raw)
                    # Synthetic descending rank weights — clearer than uniform bars
                    word_weights = [1.0 - i / max(len(word_labels), 1) for i in range(len(word_labels))]

                fig.add_trace(go.Bar(
                    y=word_labels[::-1], x=word_weights[::-1],
                    orientation='h', marker_color=topic_color.get(top_topic, SEMANTIC["highlight"]),
                    showlegend=False,
                    hovertemplate="<b>%{y}</b><br>weight: %{x:.2f}<extra></extra>",
                ), row=3, col=1)
                label = topic_label_map.get(top_topic, f"Topic {top_topic}")
                fig.update_xaxes(title=f"Relevance — “{trim(label, 32)}”", row=3, col=1)
                fig.update_yaxes(title="", row=3, col=1)
        except Exception as e:
            print(f"  [Error] Top words: {e}")

        # === ROW 3, COL 2: Session summary table ===
        try:
            if metadata:
                sess_names_sorted = sorted(set(m['session'] for m in metadata))
                seg_counts, primary_topics = [], []
                for sess_name in sess_names_sorted:
                    sess_indices = [i for i, m in enumerate(metadata) if m['session'] == sess_name]
                    seg_counts.append(len(sess_indices))
                    counts_by_topic = {}
                    for idx in sess_indices:
                        if idx < len(topics) and topics[idx] != -1:
                            counts_by_topic[topics[idx]] = counts_by_topic.get(topics[idx], 0) + 1
                    if counts_by_topic:
                        primary_tid = max(counts_by_topic, key=counts_by_topic.get)
                        primary_topics.append(trim(topic_label_map.get(primary_tid, f"Topic {primary_tid}"), 40))
                    else:
                        primary_topics.append("N/A")
                fig.add_trace(go.Table(
                    header=dict(values=['<b>Session</b>', '<b>Segments</b>', '<b>Primary topic</b>', '<b>Method</b>'],
                                fill_color=SEMANTIC["highlight"],
                                font=dict(color='white', size=12), align='left'),
                    cells=dict(values=[sess_names_sorted, seg_counts, primary_topics,
                                       [aggregated['method']] * len(sess_names_sorted)],
                               fill_color='#fafafa', align='left',
                               font=dict(size=11)),
                ), row=3, col=2)
        except Exception as e:
            print(f"  [Error] Summary table: {e}")

        # === Global layout with insight-stating subtitle ===
        # Pick the dominant topic to put in the subtitle as the key takeaway
        if topic_sizes and unique_topics:
            biggest = max(topic_sizes, key=topic_sizes.get)
            big_pct = topic_sizes[biggest] / max(sum(topic_sizes.values()), 1) * 100
            big_label = topic_label_map.get(biggest, f"Topic {biggest}")
            insight = (f"Most-discussed topic: <b>{trim(big_label, 40)}</b> "
                       f"({big_pct:.0f}% of segments) · {len(unique_topics)} topics · "
                       f"{len(sessions)} session{'s' if len(sessions) != 1 else ''}")
        else:
            insight = "No dominant topic detected"

        apply_clean_layout(
            fig,
            title="Topic & Sentiment Overview",
            insight=insight,
            height=1500,
        )
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.05,
                                      xanchor="center", x=0.5))
        print(f"  [Dual-View V3] Complete with {len(fig.data)} traces")
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
        
        # Calculate sentiment per topic — prefer real numeric scores
        def _seg_score_v(m):
            s = m.get('sentiment_score')
            if s is not None:
                return float(s)
            label = (m.get('sentiment') or "").lower()
            if 'pos' in label or 'joy' in label: return 1.0
            if any(k in label for k in ('neg', 'anger', 'sadness')): return -1.0
            return 0.0
        topic_sentiments = {}
        for topic_id in set(topics):
            topic_segments = [i for i, t in enumerate(topics) if t == topic_id]
            scores = [_seg_score_v(metadata[i]) for i in topic_segments if i < len(metadata)]
            topic_sentiments[topic_id] = sum(scores) / len(scores) if scores else 0
        
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
                
            # Calculate average sentiment for this topic — prefer real numeric scores
            def _ss(m):
                s = m.get('sentiment_score')
                if s is not None: return float(s)
                lab = (m.get('sentiment') or "").lower()
                if 'pos' in lab or 'joy' in lab: return 1.0
                if any(k in lab for k in ('neg', 'anger', 'sadness')): return -1.0
                return 0.0
            topic_segments = [i for i, t in enumerate(topics) if t == topic_id]
            scores_per_seg = [_ss(metadata[i]) for i in topic_segments if i < len(metadata)]
            avg_sent = sum(scores_per_seg) / len(scores_per_seg) if scores_per_seg else 0
            
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

    # ==========================================================================
    # NARRATIVE-DRIVEN REPLACEMENTS FOR BERTOPIC NATIVE VISUALISATIONS
    # ==========================================================================
    # These four methods replace BERTopic's default visualize_topics(),
    # visualize_documents(), visualize_barchart() and visualize_hierarchy()
    # with versions designed for non-technical readers:
    #   - Topic labels are always visible (not just numeric IDs)
    #   - Insight-stating subtitles tell the reader what to look at
    #   - Wong colorblind-safe palette throughout
    #   - Hover/exemplar text shows actual quotes, not just "Topic 3"
    # References: Tufte (2001) data-ink ratio, NN/g chart contrast guidelines,
    # Wong (2011) for the palette, Pinheiro & Bates (1996) for dendrograms.

    def _build_intertopic_map(self, res):
        """Topic landscape: each topic is a labelled bubble in 2D semantic space.
        Edges connect topics with high embedding similarity. Bubble size = mentions,
        colour = per-topic Wong palette (consistent with rest of the report)."""
        import plotly.graph_objects as go
        from viz_palette import WONG, apply_clean_layout
        import numpy as np

        coords = res.get('topic_coords', {}) or {}
        labels_list = res.get('topic_labels', []) or []
        sizes = res.get('topic_sizes', {}) or {}
        words_map = res.get('topic_words', {}) or {}
        if len(coords) < 2:
            return None

        # Build a label lookup that survives both list and dict forms
        def _label(tid):
            if isinstance(labels_list, dict):
                return labels_list.get(tid, f"Topic {tid}")
            if 0 <= tid < len(labels_list):
                return labels_list[tid]
            return f"Topic {tid}"

        tids = sorted(coords.keys())
        xs = [coords[t][0] for t in tids]
        ys = [coords[t][1] for t in tids]
        max_size = max((sizes.get(t, 1) for t in tids), default=1)
        bubble_sizes = [(sizes.get(t, 1) / max_size) * 60 + 22 for t in tids]
        colors = [WONG[(i + 1) % len(WONG)] for i in range(len(tids))]

        # Similarity edges — Euclidean distance in UMAP space
        coords_arr = np.array([coords[t] for t in tids])
        if len(coords_arr) > 1:
            d = np.linalg.norm(coords_arr - coords_arr.mean(axis=0), axis=1)
            scale = d.max() if d.max() > 0 else 1
        else:
            scale = 1
        edge_x, edge_y = [], []
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                dist = np.linalg.norm(coords_arr[i] - coords_arr[j]) / scale
                if dist < 0.5:  # only "close" topics get an edge
                    edge_x.extend([xs[i], xs[j], None])
                    edge_y.extend([ys[i], ys[j], None])

        # Hover text — actual top keywords, not just an ID
        hover = []
        for t in tids:
            words = words_map.get(t, [])[:6]
            words_str = ", ".join((w[0] if isinstance(w, (list, tuple)) else w) for w in words)
            hover.append(f"<b>{_label(t)}</b><br>Segments: {sizes.get(t, 0)}<br>Keywords: {words_str}")

        fig = go.Figure()
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode='lines',
                line=dict(color='#d8dde3', width=1.5),
                hoverinfo='none', showlegend=False, name='similarity',
            ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers+text',
            marker=dict(size=bubble_sizes, color=colors, opacity=0.85,
                        line=dict(color='white', width=2)),
            text=[_label(t) for t in tids],
            textposition="top center",
            textfont=dict(family='Inter, Segoe UI, sans-serif', size=12, color='#1a1a1a'),
            hovertext=hover, hoverinfo='text', showlegend=False,
        ))

        insight = (f"{len(tids)} topics laid out in semantic space — "
                   "topics that sit close together share vocabulary.")
        apply_clean_layout(fig, title="Topic landscape", insight=insight, height=560)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    def _build_document_cluster_view(self, res, metadata):
        """Document scatter where every dot is a transcript segment, coloured by
        its assigned topic, with the actual quote on hover. Sidebar callout (via
        title) names the topic each colour represents."""
        import plotly.graph_objects as go
        from viz_palette import WONG, apply_clean_layout

        coords = res.get('doc_coords_2d')
        doc_topics = res.get('doc_topics')
        docs = res.get('filtered_docs')
        labels_list = res.get('topic_labels', [])
        sizes = res.get('topic_sizes', {}) or {}
        if not coords or not doc_topics or not docs:
            return None

        def _label(tid):
            if isinstance(labels_list, dict):
                return labels_list.get(tid, f"Topic {tid}")
            if 0 <= tid < len(labels_list):
                return labels_list[tid]
            return f"Topic {tid}"

        unique_topics = sorted(set(doc_topics))
        topic_color = {tid: ("#cccccc" if tid == -1
                              else WONG[(i + 1) % len(WONG)])
                       for i, tid in enumerate(unique_topics)}

        fig = go.Figure()
        for tid in unique_topics:
            idxs = [i for i, t in enumerate(doc_topics) if t == tid]
            if not idxs:
                continue
            xs = [coords[i][0] for i in idxs]
            ys = [coords[i][1] for i in idxs]
            hover = []
            for i in idxs:
                quote = docs[i].strip().replace("\n", " ")
                if len(quote) > 160:
                    quote = quote[:160] + "…"
                hover.append(f"<b>{_label(tid)}</b><br>“{quote}”")
            name = f"{_label(tid)} ({len(idxs)})" if tid != -1 else f"Outliers ({len(idxs)})"
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(size=9, color=topic_color[tid], opacity=0.78,
                            line=dict(color='white', width=1)),
                name=name, hovertext=hover, hoverinfo='text',
                legendgroup=str(tid),
            ))

        n = len(coords)
        n_clusters = len([t for t in unique_topics if t != -1])
        insight = (f"{n} transcript segments grouped into {n_clusters} topics. "
                   "Hover any dot to read the actual quote.")
        apply_clean_layout(fig, title="Document map — every segment as a dot",
                           insight=insight, height=640)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(legend=dict(orientation="v", y=1, x=1.02,
                                      bgcolor="rgba(255,255,255,0.9)",
                                      bordercolor="#e5e7eb", borderwidth=1))
        return fig

    def _build_distinctive_words_chart(self, res):
        """Horizontal bars showing the most DISTINCTIVE words per topic.
        For each topic's top words we report the c-TF-IDF-style weight from
        BERTopic (or descending rank for NMF). Stacked in small multiples so
        researchers can compare topics at a glance."""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from viz_palette import WONG, apply_clean_layout

        words_map = res.get('topic_words', {}) or {}
        labels_list = res.get('topic_labels', []) or []
        sizes = res.get('topic_sizes', {}) or {}
        if not words_map:
            return None

        # Rank topics by size and limit to top 6 panels so the chart stays scannable
        ranked = sorted(
            (t for t in words_map.keys() if t != -1),
            key=lambda t: -sizes.get(t, 0)
        )[:6]
        if not ranked:
            return None

        def _label(tid):
            if isinstance(labels_list, dict):
                return labels_list.get(tid, f"Topic {tid}")
            if 0 <= tid < len(labels_list):
                return labels_list[tid]
            return f"Topic {tid}"

        n_cols = 2
        n_rows = (len(ranked) + n_cols - 1) // n_cols
        subplot_titles = []
        for tid in ranked:
            lbl = _label(tid)
            pct = (sizes.get(tid, 0) / max(sum(sizes.values()), 1)) * 100
            subplot_titles.append(f"{lbl}  ·  {sizes.get(tid, 0)} segs ({pct:.0f}%)")

        fig = make_subplots(rows=n_rows, cols=n_cols,
                            subplot_titles=subplot_titles,
                            vertical_spacing=0.16, horizontal_spacing=0.18)

        for i, tid in enumerate(ranked):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            entries = words_map.get(tid, [])[:8]
            words, weights = [], []
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    words.append(e[0]); weights.append(float(e[1]))
                else:
                    words.append(str(e)); weights.append(1.0 - 0.1 * len(weights))
            if not words:
                continue
            color = WONG[(i + 1) % len(WONG)]
            fig.add_trace(go.Bar(
                x=weights[::-1], y=words[::-1], orientation='h',
                marker=dict(color=color), showlegend=False,
                hovertemplate="<b>%{y}</b><br>weight: %{x:.3f}<extra></extra>",
            ), row=row, col=col)
            fig.update_xaxes(title="distinctiveness" if row == n_rows else "",
                             row=row, col=col)

        apply_clean_layout(
            fig, title="Distinctive words per topic",
            insight=("Higher bars = words that are characteristic of THIS topic "
                     "but rare elsewhere. Top 6 topics shown."),
            height=180 * n_rows + 120,
        )
        return fig

    def _build_topic_hierarchy(self, res):
        """Dendrogram of topics built from per-topic UMAP coordinates.
        Tells the researcher which topics merge into broader themes as you
        zoom out — far more actionable than a K-slider scatter."""
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
        except Exception:
            return None
        import plotly.figure_factory as ff
        import numpy as np
        from viz_palette import apply_clean_layout

        coords = res.get('topic_coords', {}) or {}
        labels_list = res.get('topic_labels', []) or []
        if len(coords) < 3:
            return None

        def _label(tid):
            if isinstance(labels_list, dict):
                return labels_list.get(tid, f"Topic {tid}")
            if 0 <= tid < len(labels_list):
                return labels_list[tid]
            return f"Topic {tid}"

        tids = sorted(coords.keys())
        X = np.array([coords[t] for t in tids])
        topic_names = [_label(t) for t in tids]

        # Build the dendrogram using Plotly's helper (Ward linkage)
        try:
            fig = ff.create_dendrogram(
                X, labels=topic_names, orientation='left',
                linkagefun=lambda d: linkage(d, method='ward'),
                colorscale=['#0072B2', '#E69F00', '#009E73', '#CC79A7',
                            '#56B4E9', '#D55E00'],
            )
        except Exception:
            return None

        apply_clean_layout(
            fig, title="Topic hierarchy",
            insight=("Topics that merge low on the tree are near-duplicates; "
                     "merges high up reveal the broader themes."),
            height=max(380, 50 * len(tids) + 120),
        )
        fig.update_layout(xaxis=dict(title="dissimilarity (ward distance)"))
        return fig

    def open_consolidated_report(self, figures):
        """Generates and opens a single HTML report with all figures.

        Layout follows research-backed dashboard principles:
          - 8-px spacing grid, generous whitespace between sections (Tufte / NN Group)
          - Typography hierarchy 24/18/14/12 with Inter font fallback
          - "How to read this report" intro panel sets reader expectations
          - F-pattern: title and overview pinned top-left, supporting visuals below
        """
        import tempfile

        # HTML Header
        html_content = ["""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>VideoHandTracker — Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>
                :root {
                    --bg: #f6f7f9;
                    --card: #ffffff;
                    --text: #1a1a1a;
                    --muted: #6b7280;
                    --rule: #e5e7eb;
                    --accent: #0072B2;        /* Wong blue */
                    --accent-soft: #e6f0f8;
                    --highlight: #E69F00;      /* Wong orange */
                    --space-xs: 4px;
                    --space-sm: 8px;
                    --space-md: 16px;
                    --space-lg: 32px;
                    --space-xl: 48px;
                }
                * { box-sizing: border-box; }
                body {
                    font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
                    margin: 0; padding: var(--space-lg) var(--space-md);
                    background: var(--bg); color: var(--text);
                    line-height: 1.55;
                }
                .container {
                    max-width: 1400px; margin: 0 auto;
                    background: var(--card); padding: var(--space-xl);
                    border-radius: 12px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 28px rgba(0,0,0,0.06);
                }
                /* Typography hierarchy */
                h1 {
                    font-size: 28px; font-weight: 700; letter-spacing: -0.01em;
                    color: var(--text); margin: 0 0 var(--space-xs) 0;
                }
                h2 {
                    font-size: 20px; font-weight: 600;
                    color: var(--text); margin: 0 0 var(--space-sm) 0;
                    border-left: 3px solid var(--accent);
                    padding-left: var(--space-md);
                }
                h3 {
                    font-size: 16px; font-weight: 600; color: var(--text);
                    margin: 0 0 var(--space-sm) 0;
                }
                p { font-size: 14px; color: var(--text); margin: 0 0 var(--space-sm) 0; }
                .subtitle {
                    color: var(--muted); font-size: 14px;
                    margin: 0 0 var(--space-lg) 0;
                }
                .timestamp { color: var(--muted); font-size: 12px; }

                /* Intro panel — F-pattern top-left anchor */
                .intro-panel {
                    background: var(--accent-soft);
                    border-left: 4px solid var(--accent);
                    padding: var(--space-md) var(--space-lg);
                    border-radius: 6px;
                    margin: var(--space-lg) 0;
                }
                .intro-panel h3 { color: var(--accent); margin-bottom: var(--space-sm); }
                .intro-panel ul { margin: var(--space-sm) 0 0 0; padding-left: 22px; }
                .intro-panel li { font-size: 13px; color: var(--text); margin-bottom: 4px; }

                /* Sections — spacing > borders for grouping */
                .chart-section {
                    margin: var(--space-xl) 0;
                    padding: var(--space-lg) 0;
                    border-top: 1px solid var(--rule);
                }
                .chart-section:first-of-type { border-top: none; padding-top: var(--space-md); }
                .section-meta {
                    font-size: 12px; color: var(--muted);
                    margin-bottom: var(--space-md);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>VideoHandTracker — Analysis Report</h1>
                <p class="subtitle">Topic, sentiment, gesture and motion insights</p>
                <p class="timestamp">Generated """ + str(np.datetime64('now')) + """</p>

                <div class="intro-panel">
                    <h3>How to read this report</h3>
                    <ul>
                        <li><b>Topic landscape & ranking</b> — bubble/bar position shows mention count vs. average sentiment. Look for outliers far from the dotted zero line.</li>
                        <li><b>Knowledge graph</b> — large coloured nodes are topics; small grey dots are keywords. Orange dotted lines connect semantically similar topics.</li>
                        <li><b>Executive summary cards</b> — star ratings reflect topic prominence; 1–2★ topics are likely noise and should be interpreted with caution.</li>
                        <li><b>Audio wheel</b> — radial timeline of the transcript; click a dot to jump to that moment in the recording.</li>
                    </ul>
                </div>
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
                          "Topic Landscape (Intertopic Map)",
                          "Where each segment lives (Document Map)",
                          "Distinctive words per topic",
                          "Topic hierarchy (how themes nest)"] and hasattr(fig, 'to_html'):
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
                    "Topic Landscape (Intertopic Map)": 700,
                    "Where each segment lives (Document Map)": 760,
                    "Distinctive words per topic": 820,
                    "Topic hierarchy (how themes nest)": 620,
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
                
            # 3d. Narrative-driven replacements for BERTopic native viz
            if hasattr(self, '_last_aggregated_result'):
                res = self._last_aggregated_result
                meta = getattr(self, '_last_metadata', [])

                # Intertopic map — custom version with labels + similarity links
                try:
                    fig_it = self._build_intertopic_map(res)
                    if fig_it is not None:
                        figures['Topic Landscape (Intertopic Map)'] = fig_it
                        print("✓ Added narrative Intertopic Map to report")
                except Exception as e:
                    print(f"⚠️ Custom intertopic map failed: {e}")

                # Document clusters — UMAP scatter with quote hover + exemplar list
                try:
                    fig_doc = self._build_document_cluster_view(res, meta)
                    if fig_doc is not None:
                        figures['Where each segment lives (Document Map)'] = fig_doc
                        print("✓ Added narrative Document Map to report")
                except Exception as e:
                    print(f"⚠️ Custom document map failed: {e}")

                # Distinctive words per topic (c-TF-IDF style)
                try:
                    fig_w = self._build_distinctive_words_chart(res)
                    if fig_w is not None:
                        figures['Distinctive words per topic'] = fig_w
                        print("✓ Added Distinctive Words chart to report")
                except Exception as e:
                    print(f"⚠️ Distinctive words chart failed: {e}")

                # Topic hierarchy dendrogram
                try:
                    fig_h = self._build_topic_hierarchy(res)
                    if fig_h is not None:
                        figures['Topic hierarchy (how themes nest)'] = fig_h
                        print("✓ Added Topic Hierarchy to report")
                except Exception as e:
                    print(f"⚠️ Topic hierarchy failed: {e}")
            
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
