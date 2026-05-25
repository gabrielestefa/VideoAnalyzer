import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL, ttk
from PIL import Image, ImageTk
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import collections

# Suppress TensorFlow/MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL_PATH = os.path.join(_BASE_DIR, "gesture_recognizer.task")


class TasksHandHeatmapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Tracker - HMI Analytics")
        self.max_canvas_width = 640
        self.max_canvas_height = 480

        self.video_path = None
        self.frames = []
        self.heatmap = None
        self.speed_heatmap = None
        self.opacity = 0.3
        self.landmarks_list = []

        tk.Button(root, text="Upload Video", command=self.upload_video).pack(pady=5)

        self.sensitivity_slider = Scale(root, from_=0.1, to=1.0, resolution=0.1,
                                        orient=HORIZONTAL,
                                        label="Detection Sensitivity (Lower = More Detections)")
        self.sensitivity_slider.set(0.5)
        self.sensitivity_slider.pack(fill=tk.X, padx=10, pady=5)

        self.analyze_btn = tk.Button(root, text="Analyze Video",
                                     command=self.start_analysis,
                                     state=tk.DISABLED)
        self.analyze_btn.pack(pady=5)

        self.status = tk.Label(root, text="", fg="blue")
        self.status.pack(pady=5)

        # Toggle Control Frame
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        self.show_heatmap_var = tk.BooleanVar(value=True)
        self.show_trace_var = tk.BooleanVar(value=False)
        self.show_skeleton_var = tk.BooleanVar(value=True)

        tk.Checkbutton(control_frame, text="Show Heatmap", variable=self.show_heatmap_var, command=lambda: self.update_frame(self.frame_slider.get())).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(control_frame, text="Show Trace", variable=self.show_trace_var, command=lambda: self.update_frame(self.frame_slider.get())).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(control_frame, text="Show Skeleton", variable=self.show_skeleton_var, command=lambda: self.update_frame(self.frame_slider.get())).pack(side=tk.LEFT, padx=5)

        # Heatmap Filter Dropdown
        tk.Label(control_frame, text="Heatmap Source:").pack(side=tk.LEFT, padx=5)
        self.heatmap_source = tk.StringVar()
        self.heatmap_combo = ttk.Combobox(control_frame, textvariable=self.heatmap_source, state="readonly")
        self.heatmap_combo['values'] = ("Whole Hand", "Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky", "Median")
        self.heatmap_combo.current(0) # Default to Whole Hand
        self.heatmap_combo.pack(side=tk.LEFT, padx=5)
        self.heatmap_combo.bind("<<ComboboxSelected>>", self.on_heatmap_source_change)

        # Heatmap Type Dropdown
        tk.Label(control_frame, text="Type:").pack(side=tk.LEFT, padx=5)
        self.heatmap_type = tk.StringVar(value="Presence")
        self.heatmap_type_combo = ttk.Combobox(control_frame, textvariable=self.heatmap_type, state="readonly", width=10)
        self.heatmap_type_combo['values'] = ("Presence", "Speed")
        self.heatmap_type_combo.pack(side=tk.LEFT, padx=5)
        self.heatmap_type_combo.bind("<<ComboboxSelected>>", self.on_heatmap_type_change)

        tk.Button(control_frame, text="Rotate 90°", command=self.rotate_view).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Show Dashboard", command=self.open_dashboard).pack(side=tk.LEFT, padx=5)

        # Filter Sliders Frame
        filter_frame = tk.Frame(root)
        filter_frame.pack(pady=5)
        
        tk.Label(filter_frame, text="Trace Filters:").pack(side=tk.LEFT, padx=5)
        
        self.min_trace_len_slider = Scale(filter_frame, from_=0, to=50, orient=HORIZONTAL, label="Min Len", command=self.on_filter_change)
        self.min_trace_len_slider.set(5)
        self.min_trace_len_slider.pack(side=tk.LEFT, padx=5)

        self.max_jump_slider = Scale(filter_frame, from_=1, to=50, orient=HORIZONTAL, label="Max Jump %", command=self.on_filter_change)
        self.max_jump_slider.set(10)
        self.max_jump_slider.pack(side=tk.LEFT, padx=5)

        self.rotation_angle = 0 # 0, 90, 180, 270

        self.canvas = tk.Canvas(root,
                                width=self.max_canvas_width,
                                height=self.max_canvas_height,
                                bg="black")
        self.canvas.pack()

        self.frame_slider = Scale(root, from_=0, to=0,
                                  orient=HORIZONTAL,
                                  label="Frame",
                                  command=self.update_frame)
        self.frame_slider.pack(fill=tk.X, padx=10)

        self.opacity_slider = Scale(root, from_=0, to=100,
                                    orient=HORIZONTAL,
                                    label="Heatmap Opacity (%)",
                                    command=self.update_opacity)
        self.opacity_slider.set(int(self.opacity * 100))
        self.opacity_slider.pack(fill=tk.X, padx=10)

        self.tk_image = None

        # GestureRecognizer (VIDEO mode)
        base_options = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)
        self.recognizer = GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

        self.dash_window = None
        self.dash_notebook = None
        self.dash_tabs = {} # Dictionary to hold tab frames

    def upload_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if path:
            self.video_path = path
            self.rotation_angle = 0  # Reset rotation for new video
            self.status.config(text="Video loaded")
            self.analyze_btn.config(state=tk.NORMAL)
            self.show_preview_frame()

    def show_preview_frame(self):
        if not self.video_path: return
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Handle Rotation
                if self.rotation_angle == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation_angle == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotation_angle == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Resize to Fit Canvas (Aspect Ratio)
                h, w = frame.shape[:2]
                ratio = min(self.max_canvas_width / w, self.max_canvas_height / h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                resized = cv2.resize(frame, (new_w, new_h))
                
                # Update canvas to actual frame size (no black bands)
                self.canvas.config(width=new_w, height=new_h)
                
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.canvas.delete("all")
                self.canvas.create_image(
                    0, 0,  # Top-left corner
                    image=imgtk,
                    anchor=tk.NW  # Northwest anchor
                )
                self.canvas.image = imgtk # Keep ref
        except Exception as e:
            print(f"Preview Error: {e}")

    def start_analysis(self):
        self.analyze_btn.config(state=tk.DISABLED)
        self.status.config(text="Processing video...")
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            cap = cv2.VideoCapture(self.video_path)
            
            # Re-initialize recognizer to reset internal state (timestamps must be monotonic)
            # Use sensitivity from slider
            sensitivity = self.sensitivity_slider.get()
            
            base_options = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)
            self.recognizer = GestureRecognizer.create_from_options(
                GestureRecognizerOptions(
                    base_options=base_options,
                    running_mode=RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=sensitivity,
                    min_hand_presence_confidence=sensitivity,
                    min_tracking_confidence=sensitivity
                )
            )

            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = self.video_width, self.video_height

            # self.heatmap will be generated at the end
            self.heatmap = np.zeros((height, width), dtype=np.float32)
            self.frames = []
            self.landmarks_list = []
            self.heatmap = np.zeros((height, width), dtype=np.float32)
            self.frames = []
            self.landmarks_list = []
            self.landmarks_list = []
            self.trace_path_left = []
            self.trace_path_right = []
            self.gesture_history_left = []
            self.gesture_history_right = []
            self.gesture_counts = collections.defaultdict(int)

            # Retrieve FPS for timestamp calculation
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0: self.fps = 30.0 # Fallback
            
            frame_index = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frames.append(frame.copy())

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
                
                # Calculate timestamp in ms
                timestamp_ms = int(frame_index * 1000 / self.fps)
                result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
                
                # Update progress percentage
                if self.total_frames > 0:
                    percent = int((frame_index / self.total_frames) * 100)
                    if frame_index % 5 == 0:  # Update every 5 frames to avoid UI lag
                        self.root.after(0, lambda p=percent: self.status.config(text=f"Processing video... ({p}%)"))

                frame_index += 1

                gesture_left, gesture_right = None, None
                frame_data = [] # List of (landmarks, label)

                if result.hand_landmarks:
                    for i, hand_landmarks in enumerate(result.hand_landmarks):
                        # Get handedness
                        label = result.handedness[i][0].category_name # "Left" or "Right"
                        
                        # Get gesture (if detected)
                        if result.gestures and len(result.gestures) > i:
                            gesture = result.gestures[i][0]  # Top gesture
                            gesture_name = gesture.category_name
                            gesture_score = gesture.score
                            
                            if label == "Left":
                                gesture_left = (gesture_name, gesture_score)
                            else:
                                gesture_right = (gesture_name, gesture_score)
                        
                        frame_data.append((hand_landmarks, label))
                
                self.gesture_history_left.append(gesture_left)
                self.gesture_history_right.append(gesture_right)

                if gesture_left: 
                    self.gesture_counts[f"Left_{gesture_left[0]}"] += 1
                if gesture_right: 
                    self.gesture_counts[f"Right_{gesture_right[0]}"] += 1

                self.landmarks_list.append(frame_data)

            cap.release()

            # Generate heatmap & trace based on initial selection
            self.regenerate_data()

            self.root.after(0, self.setup_slider)
            self.root.after(0, lambda: self.status.config(text="Analysis complete"))

        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        finally:
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))

    def on_heatmap_source_change(self, event):
        if self.landmarks_list:
            self.regenerate_data()
            # Refresh current frame
            self.update_frame(self.frame_slider.get())
            # Refresh dashboard if open
            if self.dash_window and tk.Toplevel.winfo_exists(self.dash_window):
                self.refresh_dashboard()

    def on_heatmap_type_change(self, event):
        if self.landmarks_list:
            # Just refresh the frame - heatmap already calculated
            self.update_frame(self.frame_slider.get())

    def on_filter_change(self, val):
        if self.landmarks_list:
            self.regenerate_data()
            self.update_frame(self.frame_slider.get())
            if self.dash_window and tk.Toplevel.winfo_exists(self.dash_window):
                self.refresh_dashboard()

    def on_filter_change(self, val):
        if self.landmarks_list:
            self.regenerate_data()
            self.update_frame(self.frame_slider.get())

    def filter_trace(self, raw_points):
        # raw_points is list of (x, y) or None. Coordinates are PIXEL coords.
        # We need to normalize to check jump distance % accurately? 
        # Or just use pixels: dist > max_jump_percent * diagonal (or width)
        
        if not raw_points: return []

        max_jump_pct = self.max_jump_slider.get() / 100.0
        min_len = self.min_trace_len_slider.get()
        
        # Calculate pixel threshold
        # Using diagonal for robustness or just width? Video dimensions might change?
        # Use current video dimensions self.video_width, self.video_height
        diag = (self.video_width**2 + self.video_height**2)**0.5
        jump_thresh_px = max_jump_pct * diag

        filtered = []
        segment = []

        for i, p in enumerate(raw_points):
            if p is None:
                # Break segment
                if len(segment) >= min_len:
                    filtered.extend(segment)
                else:
                    filtered.extend([None] * len(segment)) # Replace short segment with Nones or just leave gaps
                
                filtered.append(None)
                segment = []
                continue

            # check jump from previous Valid Point in Segment
            if segment:
                prev = segment[-1]
                # prev might be None if we just started? No, segment only holds valid points?
                # Actually, filtered needs to maintain synchronization with frames index.
                # So we must append None for every frame we discard.
                
                # Wait, if we discard a segment of length 3, we must put 3 Nones in its place 
                # to keep `filtered[i]` corresponding to `frames[i]`.
                
                dist = ((p[0] - prev[0])**2 + (p[1] - prev[1])**2)**0.5
                if dist > jump_thresh_px:
                    # Jump detected -> End previous segment
                    if len(segment) >= min_len:
                        filtered.extend(segment)
                    else:
                        filtered.extend([None] * len(segment))
                    
                    # Ensure we preserve the gap for the jump?
                    # Actually p is valid, just far away. It starts a new segment.
                    segment = [p]
                else:
                    segment.append(p)
            else:
                segment.append(p)

        # Flush last segment
        if len(segment) >= min_len:
            filtered.extend(segment)
        else:
            filtered.extend([None] * len(segment))

        return filtered

    def regenerate_data(self):
        source = self.heatmap_combo.get()
        indices = []
        trace_idx = 0 

        if source == "Whole Hand":
            indices = list(range(21))
            trace_idx = 0 # Wrist
        elif source == "Wrist":
            indices = [0]
            trace_idx = 0
        elif source == "Thumb":
            indices = [1, 2, 3, 4]
            trace_idx = 4 # Tip
        elif source == "Index":
            indices = [5, 6, 7, 8]
            trace_idx = 8
        elif source == "Middle":
            indices = [9, 10, 11, 12]
            trace_idx = 12
        elif source == "Ring":
            indices = [13, 14, 15, 16]
            trace_idx = 16
        elif source == "Pinky":
            indices = [17, 18, 19, 20]
            trace_idx = 20
        elif source == "Median":
            indices = list(range(21))
            trace_idx = -1 # Special flag for Median

        # Reset heat maps and traces
        self.heatmap = np.zeros((self.video_height, self.video_width), dtype=np.float32)
        self.speed_heatmap = np.zeros((self.video_height, self.video_width), dtype=np.float32)
        self.trace_path_left = []
        self.trace_path_right = []

        for frame_data in self.landmarks_list:
            # frame_data is list of (landmarks, label)
            
            trace_l, trace_r = None, None

            for hand_landmarks, label in frame_data:
                # 1. Heatmap Accumulation
                points = []
                for idx in indices:
                    lm = hand_landmarks[idx]
                    x = int(lm.x * self.video_width)
                    y = int(lm.y * self.video_height)
                    points.append((x, y))
                
                if points:
                    # Calculate bounding box
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    # Clamp
                    min_x = max(0, min_x)
                    max_x = min(self.video_width, max_x)
                    min_y = max(0, min_y)
                    max_y = min(self.video_height, max_y)

                    if max_x == min_x: max_x += 1
                    if max_y == min_y: max_y += 1
                    
                    self.heatmap[min_y:max_y, min_x:max_x] += 1

                # 2. Trace Point Extraction & Depth Proxy
                # Depth Proxy = 1 / Scale. Scale = Dist(Wrist(0), Middle MCP(9))
                wrist = hand_landmarks[0]
                mcp = hand_landmarks[9]
                
                # Calculate scale in normalized coords to be resolution independent
                scale = ((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2 + (wrist.z - mcp.z)**2)**0.5
                if scale == 0: scale = 0.001
                z_proxy = 1.0 / scale # Smaller scale (far away) -> Higher Z
                
                # Target trace point
                if trace_idx == -1:
                    # Median Calculation (Centroid)
                    mean_x = np.mean([lm.x for lm in hand_landmarks])
                    mean_y = np.mean([lm.y for lm in hand_landmarks])
                    tx = int(mean_x * self.video_width)
                    ty = int(mean_y * self.video_height)
                else:
                    t_lm = hand_landmarks[trace_idx]
                    tx = int(t_lm.x * self.video_width)
                    ty = int(t_lm.y * self.video_height)
                
                # Store (x, y, z)
                if label == "Left":
                    trace_l = (tx, ty, z_proxy)
                else:
                    trace_r = (tx, ty, z_proxy)

            self.trace_path_left.append(trace_l)
            self.trace_path_right.append(trace_r)

        # Apply Filters
        self.trace_path_left = self.filter_trace(self.trace_path_left)
        self.trace_path_right = self.filter_trace(self.trace_path_right)

        # Calculate Speed Heatmap
        # Process both hands
        for path in [self.trace_path_left, self.trace_path_right]:
            for i in range(1, len(path)):
                p_prev = path[i-1]
                p_curr = path[i]
                
                if p_prev is not None and p_curr is not None:
                    # Calculate velocity (pixels per frame)
                    dx = p_curr[0] - p_prev[0]
                    dy = p_curr[1] - p_prev[1]
                    speed = np.sqrt(dx**2 + dy**2)
                    
                    # Accumulate speed at current position
                    x, y = int(p_curr[0]), int(p_curr[1])
                    if 0 <= x < self.video_width and 0 <= y < self.video_height:
                        # Use a small area around the point for smoother visualization
                        radius = 5
                        y_min = max(0, y - radius)
                        y_max = min(self.video_height, y + radius)
                        x_min = max(0, x - radius)
                        x_max = min(self.video_width, x + radius)
                        
                        # Add speed value (weighted by presence)
                        self.speed_heatmap[y_min:y_max, x_min:x_max] += speed

        # Apply Blur to both heatmaps
        self.heatmap = cv2.GaussianBlur(self.heatmap, (51, 51), 0)
        if np.max(self.speed_heatmap) > 0:
            self.speed_heatmap = cv2.GaussianBlur(self.speed_heatmap, (51, 51), 0)

        pass 


    def calculate_metrics(self):
        metrics = {
            'left': {'dist': 0, 'max_speed': 0, 'avg_speed': 0, 'speeds': []},
            'right': {'dist': 0, 'max_speed': 0, 'avg_speed': 0, 'speeds': []}
        }
        
        # Helper for median filtering to remove outliers
        def smooth_data(data, window_size=5):
            if not data: return []
            result = []
            for i in range(len(data)):
                start = max(0, i - window_size // 2)
                end = min(len(data), i + window_size // 2 + 1)
                window = data[start:end]
                result.append(np.median(window))
            return result

        for side, path in [('left', self.trace_path_left), ('right', self.trace_path_right)]:
            total_dist = 0
            speeds = [] # Pixels per frame
            
            # Using filtered path (which has Nones for filtered segments)
            for i in range(1, len(path)):
                p1 = path[i-1]
                p2 = path[i]
                
                if p1 is None or p2 is None:
                    speeds.append(0) 
                    continue
                    
                # Dist in pixels
                d = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                total_dist += d
                speeds.append(d) 
            
            if speeds:
                # Apply smoothing to remove extreme outliers/jitter
                smoothed_speeds = smooth_data(speeds)
                
                metrics[side]['dist'] = total_dist
                metrics[side]['max_speed'] = max(smoothed_speeds) if smoothed_speeds else 0
                metrics[side]['avg_speed'] = sum(smoothed_speeds) / len(smoothed_speeds) if smoothed_speeds else 0
                metrics[side]['speeds'] = smoothed_speeds
                
        return metrics

    def calculate_hmi_metrics(self, side='left'):
        # HMI Analytics: Phase Segmentation, Tortuosity, Tremor
        
        path = self.trace_path_left if side == 'left' else self.trace_path_right
        if not path or len(path) < 2: return None
        
        # 1. Calculate Velocity & Acceleration & Jerk
        velocities = []
        accelerations = []
        jerks = []
        phases = [] # 0=Rest, 1=Reach, 2=Hover, 3=Touch
        
        # Need timestamp-aware calculation if possible, but frames are constant interval roughly
        fps = getattr(self, 'fps', 30.0)
        dt = 1.0 / fps
        
        # Fill missing data filtering Nones
        clean_path = []
        indices = []
        for i, p in enumerate(path):
            if p is not None:
                clean_path.append(np.array(p))
                indices.append(i)
        
        if len(clean_path) < 5: return None

        # Calculate derivatives
        velocities = []
        for i in range(1, len(clean_path)):
            v = np.linalg.norm(clean_path[i] - clean_path[i-1]) / dt
            velocities.append(v)
            
        accelerations = []
        for i in range(1, len(velocities)):
            a = (velocities[i] - velocities[i-1]) / dt
            accelerations.append(a)
            
        jerks = []
        for i in range(1, len(accelerations)):
            j = abs((accelerations[i] - accelerations[i-1]) / dt)
            jerks.append(j)

        # Pad arrays
        velocities = [0] + velocities
        accelerations = [0]*2 + accelerations
        jerks = [0]*3 + jerks
        
        # Phase Segmentation & Hover Extraction
        max_v = max(velocities) if velocities else 1
        hover_points = []
        phases = []
        
        for i, v in enumerate(velocities):
            # Phase Logic:
            # 0=Rest (<10% max speed)
            # 1=Reach (>40% max speed)
            # 2=Hover (In between, active but slow)
            if v < max_v * 0.1:
                p = 0
            elif v > max_v * 0.4:
                p = 1
            else:
                p = 2
                # Store hover coord
                if i < len(clean_path):
                    hover_points.append(clean_path[i])
            phases.append(p)

        # Map phases back to original video timeline
        full_phases = [-1] * len(path) # -1 = No Hand / Unknown
        for k, phase_val in enumerate(phases):
            if k < len(indices):
                original_idx = indices[k]
                full_phases[original_idx] = phase_val
        
        # Use full_phases for return, but keep original phases logic for efficiency calculation
        # Actually efficiency calc uses 'phases' (the compressed one) which is correct because it's sequential movement
        # But for visualization we want full_phases

        # Path Efficiency (Tortuosity)
        # Calculate for each "Reach" segment
        tortuosity_scores = []
        current_segment_dist = 0
        segment_start_idx = -1
        
        for i in range(1, len(phases)):
            if phases[i] == 1: # Reaching
                if phases[i-1] != 1: # Start of reach
                    segment_start_idx = i-1
                    current_segment_dist = 0
                
                # Add step distance
                step = np.linalg.norm(clean_path[i] - clean_path[i-1])
                current_segment_dist += step
                
            elif phases[i] != 1 and phases[i-1] == 1: # End of reach
                if segment_start_idx >= 0:
                    start_pt = clean_path[segment_start_idx]
                    end_pt = clean_path[i]
                    displacement = np.linalg.norm(end_pt - start_pt)
                    if displacement > 10: # Ignore tiny moves
                        t = current_segment_dist / displacement
                        tortuosity_scores.append(t)
                segment_start_idx = -1

        avg_efficiency = np.mean(tortuosity_scores) if tortuosity_scores else 1.0
        avg_jerk = np.mean(jerks) if jerks else 0

        return {
            'phases': full_phases, # Now correctly aligned with video frames
            'counts': collections.Counter(phases),
            'hover_points': hover_points,
            'efficiency': avg_efficiency,
            'avg_jerk': avg_jerk,
            'clean_path': clean_path # for plotting hover map context
        }



    def open_dashboard(self):
        if not self.frames:
            messagebox.showinfo("Info", "No analysis data yet.")
            return

        if self.dash_window and tk.Toplevel.winfo_exists(self.dash_window):
            self.dash_window.lift()
            return

        # Create Toplevel Window
        self.dash_window = tk.Toplevel(self.root)
        self.dash_window.title("Data Insights & HMI Analytics")
        self.dash_window.geometry("1100x850")
        
        # Dashboard Controls (Source Switching)
        dash_control = tk.Frame(self.dash_window)
        dash_control.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Label(dash_control, text="Data Source:").pack(side=tk.LEFT, padx=5)
        # Use same variable as main window
        dash_combo = ttk.Combobox(dash_control, textvariable=self.heatmap_source, state="readonly")
        dash_combo['values'] = self.heatmap_combo['values']
        dash_combo.pack(side=tk.LEFT, padx=5)
        dash_combo.bind("<<ComboboxSelected>>", self.on_heatmap_source_change)

        tk.Label(dash_control, text="3D Style:").pack(side=tk.LEFT, padx=(15, 5))
        self.dash_3d_style = tk.StringVar(value="Trace Lines")
        style_combo = ttk.Combobox(dash_control, textvariable=self.dash_3d_style, state="readonly", width=15)
        style_combo['values'] = ("Trace Lines", "Point Cloud", "Density Heatmap")
        style_combo.pack(side=tk.LEFT, padx=5)
        style_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_dashboard())

        # Voxel Control (Only enabled for Voxel mode theoretically, but can just be there)
        # Voxel Control (Only enabled for Voxel mode theoretically, but can just be there)
        tk.Label(dash_control, text="Density Grid:").pack(side=tk.LEFT, padx=(15, 5))
        self.dash_voxel_size = tk.IntVar(value=50)
        v_scale = tk.Scale(dash_control, from_=20, to=150, orient=tk.HORIZONTAL, variable=self.dash_voxel_size, showvalue=0)
        v_scale.pack(side=tk.LEFT, padx=5)
        # Bind release event to avoid excessive redraws
        v_scale.bind("<ButtonRelease-1>", lambda e: self.refresh_dashboard())

        self.dash_notebook = ttk.Notebook(self.dash_window)
        self.dash_notebook.pack(fill=tk.BOTH, expand=True)

        # Create Tabs Containers
        self.dash_tabs['general'] = tk.Frame(self.dash_notebook)
        self.dash_notebook.add(self.dash_tabs['general'], text="General Stats")
        
        self.dash_tabs['hmi'] = tk.Frame(self.dash_notebook)
        self.dash_notebook.add(self.dash_tabs['hmi'], text="HMI Analytics")
        
        self.dash_tabs['timeline'] = tk.Frame(self.dash_notebook)
        self.dash_notebook.add(self.dash_tabs['timeline'], text="Timeline")
        
        self.dash_tabs['3d'] = tk.Frame(self.dash_notebook)
        self.dash_notebook.add(self.dash_tabs['3d'], text="3D Visualization")

        self.refresh_dashboard()

    def refresh_dashboard(self):
        if not self.dash_window or not tk.Toplevel.winfo_exists(self.dash_window):
            return

        # Calculate all metrics fresh
        metrics = self.calculate_metrics()
        hmi_left = self.calculate_hmi_metrics('left')
        hmi_right = self.calculate_hmi_metrics('right')

        # --- Tab 1: General ---
        # Clear previous
        for widget in self.dash_tabs['general'].winfo_children():
            widget.destroy()

        fig1 = Figure(figsize=(10, 8), dpi=100)
        ax1 = fig1.add_subplot(221)
        ax2 = fig1.add_subplot(222)
        ax3 = fig1.add_subplot(223)
        ax4 = fig1.add_subplot(224)
        
        # 1. Heatmap
        ax1.imshow(self.heatmap, cmap="hot")
        ax1.set_title(f"Heatmap ({self.heatmap_source.get()})")
        ax1.axis("off")

        # 2. Velocity
        fps = getattr(self, 'fps', 30.0)
        time_axis = [i / fps for i in range(len(metrics['left']['speeds']))]
        ax2.plot(time_axis, metrics['left']['speeds'], label='L', color='blue')
        ax2.plot(time_axis, metrics['right']['speeds'], label='R', color='red')
        ax2.set_title("Velocity (px/s)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Trajectory with Z-color?
        ax3.set_title("Trajectory Map")
        if self.trace_path_left:
            valid_l = [p for p in self.trace_path_left if p]
            if valid_l:
                ax3.plot([p[0] for p in valid_l], [p[1] for p in valid_l], '.', color='blue', alpha=0.5, markersize=1)
        if self.trace_path_right:
            valid_r = [p for p in self.trace_path_right if p]
            if valid_r:
                ax3.plot([p[0] for p in valid_r], [p[1] for p in valid_r], '.', color='red', alpha=0.5, markersize=1)
        ax3.set_xlim(0, self.video_width)
        ax3.set_ylim(self.video_height, 0) 
        ax3.set_aspect('equal')

        # 4. Gesture Stats
        gestures = list(self.gesture_counts.keys())
        counts = list(self.gesture_counts.values())
        if gestures:
            ax4.bar(gestures, counts, color=['blue' if 'Left' in g else 'red' for g in gestures])
            ax4.tick_params(axis='x', rotation=45, labelsize=8)
            ax4.set_title("Detected Gestures")
            ax4.set_ylabel("Frame Count")
        else:
            ax4.text(0.5, 0.5, "No Gestures", ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Detected Gestures")

        fig1.tight_layout()
        canvas1 = FigureCanvasTkAgg(fig1, master=self.dash_tabs['general'])
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        # --- Tab 2: HMI Analytics ---
        for widget in self.dash_tabs['hmi'].winfo_children():
            widget.destroy()

        fig2 = Figure(figsize=(10, 8), dpi=100)
        # 2x2 Grid
        h_ax1 = fig2.add_subplot(221) # Pie: Phase
        h_ax2 = fig2.add_subplot(222) # Bar: Efficiency
        h_ax3 = fig2.add_subplot(223) # Scatter: Hover Map
        h_ax4 = fig2.add_subplot(224) # Text/Bar: Stability
        
        # 1. Phase Distribution (Pie)
        labels = ['Rest', 'Reach', 'Hover']
        colors = ['gray', 'green', 'orange']
        
        # Aggregate L+R counts
        total_counts = collections.Counter()
        if hmi_left: total_counts.update(hmi_left['counts'])
        if hmi_right: total_counts.update(hmi_right['counts'])
        
        sizes = [total_counts[0], total_counts[1], total_counts[2]]
        if sum(sizes) > 0:
            h_ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            h_ax1.set_title("Interaction Time Distribution")
        else:
            h_ax1.text(0.5, 0.5, "No Data", ha='center')

        # 2. Path Efficiency (Bar)
        eff_vals = []
        eff_labels = []
        if hmi_left: 
            eff_vals.append(hmi_left['efficiency'])
            eff_labels.append("Left")
        if hmi_right: 
            eff_vals.append(hmi_right['efficiency'])
            eff_labels.append("Right")
            
        if eff_vals:
            bars = h_ax2.bar(eff_labels, eff_vals, color=['blue', 'red'])
            h_ax2.set_title("Movement Efficiency (Lower is Better)")
            h_ax2.set_ylabel("Tortuosity Ratio (1.0 = Straight)")
            h_ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
            h_ax2.set_ylim(bottom=0.5) # Scale
        
        # 3. Hover Map
        h_ax3.set_title("Attention 'Hotspots' (Hover Locations)")
        h_ax3.set_xlim(0, self.video_width)
        h_ax3.set_ylim(self.video_height, 0) # Inverted Y
        h_ax3.set_aspect('equal')
        
        if hmi_left and hmi_left['hover_points']:
            pts = np.array(hmi_left['hover_points'])
            h_ax3.scatter(pts[:,0], pts[:,1], c='orange', alpha=0.3, s=10, label='L Hover')
            
        if hmi_right and hmi_right['hover_points']:
            pts = np.array(hmi_right['hover_points'])
            h_ax3.scatter(pts[:,0], pts[:,1], c='orange', alpha=0.3, s=10, label='R Hover')
        h_ax3.grid(True, alpha=0.3)
        
        # 4. Stability Summary (Text)
        h_ax4.axis('off')
        h_ax4.set_title("Motion Quality Metrics")
        
        txt = "Stability Score (Jerk Analysis):\n\n"
        if hmi_left:
            txt += f"Left Hand Avg Jerk: {hmi_left['avg_jerk']:.2f}\n"
        if hmi_right:
            txt += f"Right Hand Avg Jerk: {hmi_right['avg_jerk']:.2f}\n"
            
        txt += "\nInterpretation:\n"
        txt += "- High Hover % = Indecision or Focus\n"
        txt += "- Eff > 1.5 = Navigating obstacles or indirect path\n"
        txt += "- High Jerk = Tremor or corrective movements"
        
        h_ax4.text(0.1, 0.5, txt, fontsize=10, va='center')
        
        fig2.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, master=self.dash_tabs['hmi'])
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Tab 3: Timeline ---
        for widget in self.dash_tabs['timeline'].winfo_children():
            widget.destroy()

        fig_tl = Figure(figsize=(14, 6), dpi=100)
        ax_tl = fig_tl.add_subplot(111)
        
        # Time axis
        total_frames = len(self.frames)
        fps = getattr(self, 'fps', 30.0)
        time_axis = np.arange(total_frames) / fps
        
        # Band heights
        band_left = 0.0
        band_right = 0.2
        band_phase = 0.4
        band_speed_base = 0.6
        
        # 1. Left Hand Presence (Blue band)
        left_present = [1 if g is not None else 0 for g in self.gesture_history_left]
        ax_tl.fill_between(time_axis, band_left, band_left + 0.15, 
                           where=np.array(left_present) > 0, 
                           color='blue', alpha=0.5, label='Left Hand')
        
        # 2. Right Hand Presence (Red band)
        right_present = [1 if g is not None else 0 for g in self.gesture_history_right]
        ax_tl.fill_between(time_axis, band_right, band_right + 0.15, 
                           where=np.array(right_present) > 0, 
                           color='red', alpha=0.5, label='Right Hand')
        
        # 3. Phase Overlay (using HMI metrics phases)
        # Get phase data from HMI metrics
        phase_colors = {0: 'gray', 1: 'green', 2: 'orange'}
        
        if hmi_left and 'phases' in hmi_left:
            phases_l = hmi_left['phases']
            for phase_id, color in phase_colors.items():
                phase_mask = np.array(phases_l) == phase_id
                ax_tl.fill_between(time_axis[:len(phases_l)], band_phase, band_phase + 0.05,
                                  where=phase_mask, color=color, alpha=0.6)
        
        if hmi_right and 'phases' in hmi_right:
            phases_r = hmi_right['phases']
            for phase_id, color in phase_colors.items():
                phase_mask = np.array(phases_r) == phase_id
                ax_tl.fill_between(time_axis[:len(phases_r)], band_phase + 0.05, band_phase + 0.1,
                                  where=phase_mask, color=color, alpha=0.6)
        
        # 4. Speed Line Graph (normalized to 0.6-1.0 range)
        speeds_l = metrics['left']['speeds']
        speeds_r = metrics['right']['speeds']
        
        if speeds_l:
            max_speed = max(max(speeds_l) if speeds_l else 1, max(speeds_r) if speeds_r else 1)
            if max_speed > 0:
                norm_speeds_l = [band_speed_base + 0.35 * (s / max_speed) for s in speeds_l]
                norm_speeds_r = [band_speed_base + 0.35 * (s / max_speed) for s in speeds_r]
                
                ax_tl.plot(time_axis[:len(speeds_l)], norm_speeds_l, color='cyan', linewidth=1, alpha=0.7, label='L Speed')
                ax_tl.plot(time_axis[:len(speeds_r)], norm_speeds_r, color='magenta', linewidth=1, alpha=0.7, label='R Speed')
        
        # Styling
        ax_tl.set_xlabel('Time (seconds)', fontsize=12)
        ax_tl.set_yticks([band_left + 0.075, band_right + 0.075, band_phase + 0.05, band_speed_base + 0.175])
        ax_tl.set_yticklabels(['L Hand', 'R Hand', 'Phase', 'Speed'])
        ax_tl.set_ylim(-0.05, 1.05)
        ax_tl.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
        ax_tl.set_title('Timeline: Hand Activity, Phases, and Speed', fontsize=14, fontweight='bold')
        ax_tl.grid(True, alpha=0.3, axis='x')
        ax_tl.legend(loc='upper right', fontsize=8)
        
        # Add phase legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.6, label='Rest'),
            Patch(facecolor='green', alpha=0.6, label='Reach'),
            Patch(facecolor='orange', alpha=0.6, label='Hover')
        ]
        ax_tl.legend(handles=legend_elements, loc='upper left', fontsize=8, title='Phases')
        
        fig_tl.tight_layout()
        canvas_tl = FigureCanvasTkAgg(fig_tl, master=self.dash_tabs['timeline'])
        canvas_tl.draw()
        canvas_tl.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Make timeline clickable for video seeking
        def on_timeline_click(event):
            if event.xdata and event.inaxes == ax_tl:
                target_time = event.xdata
                target_frame = int(target_time * fps)
                if 0 <= target_frame < len(self.frames):
                    self.frame_slider.set(target_frame)
        
        canvas_tl.mpl_connect('button_press_event', on_timeline_click)

        # --- Tab 4: 3D ---
        for widget in self.dash_tabs['3d'].winfo_children():
            widget.destroy()

        fig3 = Figure(figsize=(10, 8), dpi=100)
        ax3d = fig3.add_subplot(111, projection='3d')
        
        style = self.dash_3d_style.get()
        ax3d.set_title(f"3D Visual: {style} ({self.heatmap_source.get()})")
        ax3d.set_xlabel("X (Left-Right)")
        ax3d.set_ylabel("Depth (Near-Far)")
        ax3d.set_zlabel("Height (Up-Down)")
        
        # MAPPING: X=VideoX, Y=VideoDepth(Proxy), Z=VideoHeight(Inv)
        # Rotation: If 90/270, we need to swap dimensions to match the visual "Up"
        
        angle = getattr(self, 'rotation_angle', 0)
        rotated = angle in [90, 270]

        def get_plot_data(path):
            # Returns xs, ys, zs with NaNs for gaps
            if not path: return [], [], []
            xs, ys, zs = [], [], []
            for p in path:
                if p is None:
                    xs.append(np.nan)
                    ys.append(np.nan)
                    zs.append(np.nan)
                else:
                    px, py, pz = p[0], p[1], p[2]
                    
                    # Apply visual rotation to match screen
                    if rotated:
                        # 90 deg CW standard: x' = h - y, y' = x
                        # We use this for visual plotting
                        # X_plot (width) becomes derived from Old Y
                        # Z_plot (height) becomes derived from Old X
                        
                        # Simplification: Just swap X and Y raw values logic
                        # Screen X ~ Old Y
                        # Screen Y (Inverted) ~ Old X
                        
                        # Let's map to "Plot Space":
                        # X_plot = py
                        # Y_plot = pz (Depth)
                        # Z_plot = self.video_width - px # New Height logic
                        
                        xs.append(py)
                        zs.append(self.video_width - px)
                        
                        # Update labels if first run? No, tedious.
                        # Just swapping data is enough for visual alignment.
                    else:
                        xs.append(px)
                        zs.append(self.video_height - py) # Normal Inverted Y

                    ys.append(pz * 100) # Depth scaled

            return xs, ys, zs

        # Update Axis Labels based on rotation
        if rotated:
            ax3d.set_xlabel("X (Screen Width)")
            ax3d.set_zlabel("Height (Screen Up)")
        else:
            ax3d.set_xlabel("X (Left-Right)")
            ax3d.set_zlabel("Height (Up-Down)")

        def plot_data(path, color, label):
            # For 3D Density, we need RAW coordinates to compute bins and color lookup
            # Then we apply rotation to the geometry for plotting.
            
            # Step 1: Get Raw Coords (Unrotated) for Logic
            raw_pts = []
            for p in path:
                if p: raw_pts.append(p)
            
            if not raw_pts: return
            
            # Extract Components
            raw_x = [p[0] for p in raw_pts]
            raw_y = [p[1] for p in raw_pts]
            raw_z = [p[2] for p in raw_pts] # This is Depth Proxy (not yet scaled by 100)

            # Get PLOT Coords (Rotated & Scaled) for rendering non-density plots
            xs, ys, zs = get_plot_data(path)
            # xs, ys, zs contain NaNs, filter for plotting
            clean_xs = [x for x in xs if not np.isnan(x)]
            clean_ys = [y for y in ys if not np.isnan(y)]
            clean_zs = [z for z in zs if not np.isnan(z)]
            
            if style == "Trace Lines":
                # Plot with NaNs to break lines
                ax3d.plot(xs, ys, zs, c=color, alpha=0.8, linewidth=1.5, label=label)
                # Add end dot
                if clean_xs:
                    ax3d.scatter([clean_xs[-1]], [clean_ys[-1]], [clean_zs[-1]], c=color, s=20)
                    
            elif style == "Point Cloud":
                ax3d.scatter(clean_xs, clean_ys, clean_zs, c=color, s=2, alpha=0.3, label=label)
                
            elif style == "Density Heatmap":
                # Volumetric extrusion of 2D heatmap
                bins = self.dash_voxel_size.get()
                
                # Create XY grid based on video dimensions
                x_grid = np.linspace(0, self.video_width - 1, bins)
                y_grid = np.linspace(0, self.video_height - 1, bins)
                
                # Z range for extrusion (arbitrary - just for viz)
                z_steps = 20  # Fixed number of vertical layers
                z_range = np.linspace(0, 100, z_steps)
                
                # Get max heat for normalization
                max_heat = np.max(self.heatmap) if np.max(self.heatmap) > 0 else 1
                
                try:
                    cmap = plt.get_cmap('hot')
                except:
                    cmap = cm.get_cmap('hot')
                
                grid_points = []
                colors_list = []
                
                # Sample heatmap on grid
                for gx in x_grid:
                    for gy in y_grid:
                        # Sample heatmap value
                        cx = int(np.clip(gx, 0, self.video_width - 1))
                        cy = int(np.clip(gy, 0, self.video_height - 1))
                        
                        heat_val = self.heatmap[cy, cx] / max_heat
                        
                        # Only create column if heat exists (threshold)
                        if heat_val > 0.02:  # 2% threshold
                            # Create vertical column of points
                            # Height of column could be proportional to heat_val
                            max_z_idx = int(heat_val * (z_steps - 1))
                            
                            for z_idx in range(max_z_idx + 1):
                                gz = z_range[z_idx]
                                grid_points.append((gx, gy, gz))
                                colors_list.append(heat_val)
                
                if not grid_points: return
                
                # Convert to plot coordinates
                plot_x_list = []
                plot_y_list = []
                plot_z_list = []
                
                for gx, gy, gz in grid_points:
                    if rotated:
                        plot_x_list.append(gy)
                        plot_y_list.append(gz)  # Z is now just visual depth
                        plot_z_list.append(self.video_width - gx)
                    else:
                        plot_x_list.append(gx)
                        plot_y_list.append(gz)  # Z is now just visual depth
                        plot_z_list.append(self.video_height - gy)
                
                # Map colors
                colors_mapped = cmap(colors_list)
                colors_mapped[:, 3] = 1.0
                
                # Fixed small size
                marker_size = 8
                
                ax3d.scatter(plot_x_list, plot_y_list, plot_z_list, c=colors_mapped, s=marker_size, marker='o', alpha=1.0, depthshade=True)

        if self.trace_path_left:
            plot_data(self.trace_path_left, 'blue', 'Left')

        if self.trace_path_right:
            plot_data(self.trace_path_right, 'red', 'Right')

        # Legend
        if style != "Density Heatmap":
            ax3d.legend()
            
        # View: 3/4 Isometric
        ax3d.view_init(elev=30, azim=-45)
        ax3d.set_box_aspect([1, 1, 0.6])

        canvas3 = FigureCanvasTkAgg(fig3, master=self.dash_tabs['3d'])
        canvas3.draw()
        # Add Toolbar for rotation!
        toolbar = NavigationToolbar2Tk(canvas3, self.dash_tabs['3d'])
        toolbar.update()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def old_show_dashboard(self):
        # Deprecated
        pass

    def setup_slider(self):
        self.frame_slider.configure(to=len(self.frames) - 1)
        self.update_frame(0)

    def classify_pose(self, landmarks, label):
        # Finger tip and PIP (Proximal Interphalangeal joint) indices
        # Thumb, Index, Middle, Ring, Pinky
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18] # For thumb using IP(3) vs MCP(2) or just Tip vs IP.
        # Actually for thumb: Tip(4) vs IP(3). PIP equiv is MCP(2)
        # Standard heuristic: Compare distance to wrist(0)

        wrist = landmarks[0]
        
        def dist(p1_idx, p2_idx):
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5

        def dist_obj(p_obj, p2_idx):
            p2 = landmarks[p2_idx]
            return ((p_obj.x - p2.x)**2 + (p_obj.y - p2.y)**2 + (p_obj.z - p2.z)**2)**0.5

        open_fingers = 0
        
        # Thumb: complex. check if tip is further from 'pinky mcp' (17) than 'index mcp' (5)?
        # Simple heuristic: dist(tip, wrist) > dist(ip, wrist) usually works for extended thumb?
        # A better one for 2D/3D: Tip x vs IP x relative to handedness?
        # Let's try simple distance first: Thumb Tip(4) vs Center Palm/Pinky Base?
        # Let's use distance to wrist vs IP distance to wrist.
        if dist(4, 0) > dist(3, 0): # Thumb extended
             # Verify it's not tucked. Vector checks are better but dist is okay for basic start.
             open_fingers += 1

        # Fingers: Tip(8) vs PIP(6), Tip(12) vs PIP(10)...
        for i in range(1, 5):
            if dist(tips[i], 0) > dist(pips[i], 0):
                open_fingers += 1

        if open_fingers == 0:
            return "Fist"
        elif open_fingers == 5:
            return "Open"
        elif open_fingers == 1:
            # Check if it's index
            if dist(8, 0) > dist(6, 0):
                return "Pointing"
            return "1_Finger"
        elif open_fingers == 2:
            # Check if index and middle
            if dist(8, 0) > dist(6, 0) and dist(12, 0) > dist(10, 0):
                return "Peace"
            return "2_Fingers"
        else:
            return f"{open_fingers}_Fingers"

    def rotate_view(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        if self.frames:
            self.update_frame(self.frame_slider.get())
        elif self.video_path:
            self.show_preview_frame()

    def transform_point(self, x, y):
        # x, y are normalized [0, 1]
        if self.rotation_angle == 90:
            return 1 - y, x
        elif self.rotation_angle == 180:
            return 1 - x, 1 - y
        elif self.rotation_angle == 270:
            return y, 1 - x
        return x, y

    def update_frame(self, val):
        idx = int(val)
        if idx >= len(self.frames):
            return

        frame = self.frames[idx].copy()

        # Handle rotation using CV2
        if self.rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 1. Apply Heatmap
        if self.show_heatmap_var.get():
            # Select which heatmap to show
            heatmap_type = self.heatmap_type.get()
            
            if heatmap_type == "Speed":
                hm = self.speed_heatmap.copy() if self.speed_heatmap is not None else np.zeros_like(self.heatmap)
            else:
                hm = self.heatmap.copy()
            
            # Heatmap needs to be rotated to match frame
            if self.rotation_angle == 90:
                hm = cv2.rotate(hm, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                hm = cv2.rotate(hm, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                hm = cv2.rotate(hm, cv2.ROTATE_90_COUNTERCLOCKWISE)

            heat_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Use different colormaps
            if heatmap_type == "Speed":
                # Speed: Blue (slow) -> Red (fast)
                heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
            else:
                # Presence: Hot colormap (dark -> red -> yellow)
                heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_HOT)
            overlay = cv2.addWeighted(frame, 1.0, heat_color, self.opacity, 0)
        else:
            overlay = frame.copy()

        h, w = overlay.shape[:2]

        # Helper to convert normalized point to pixel coords on the ROTATED image
        def to_pixel(nx, ny):
            rx, ry = self.transform_point(nx, ny)
            return int(rx * w), int(ry * h)

        # 2. Draw Trace
        if self.show_trace_var.get():
            # Draw Left Hand Trace (Blue)
            points_left = self.trace_path_left[:idx+1]
            for i in range(1, len(points_left)):
                if points_left[i-1] is None or points_left[i] is None: continue
                
                # These stored points are in ORIGINAL PIXEL COORDS (based on video_width/height)
                # We need to normalize them back to rotate them, then scale to new dimensions.
                # Since original video size is self.video_width, self.video_height
                
                p1_orig = points_left[i-1]
                p2_orig = points_left[i]
                
                p1_norm = (p1_orig[0] / self.video_width, p1_orig[1] / self.video_height)
                p2_norm = (p2_orig[0] / self.video_width, p2_orig[1] / self.video_height)

                p1_px = to_pixel(*p1_norm)
                p2_px = to_pixel(*p2_norm)

                cv2.line(overlay, p1_px, p2_px, (255, 0, 0), 2) # Blue

            # Draw Right Hand Trace (Red)
            points_right = self.trace_path_right[:idx+1]
            for i in range(1, len(points_right)):
                if points_right[i-1] is None or points_right[i] is None: continue

                p1_orig = points_right[i-1]
                p2_orig = points_right[i]
                
                p1_norm = (p1_orig[0] / self.video_width, p1_orig[1] / self.video_height)
                p2_norm = (p2_orig[0] / self.video_width, p2_orig[1] / self.video_height)

                p1_px = to_pixel(*p1_norm)
                p2_px = to_pixel(*p2_norm)

                cv2.line(overlay, p1_px, p2_px, (0, 0, 255), 2) # Red

        # 3. Draw Skeleton
        if self.show_skeleton_var.get() and idx < len(self.landmarks_list):
            HAND_CONNECTIONS = frozenset([
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ])

            for landmarks, label in self.landmarks_list[idx]:
                for start, end in HAND_CONNECTIONS:
                    p1 = landmarks[start]
                    p2 = landmarks[end]
                    
                    # p1.x, p1.y are normalized
                    pt1 = to_pixel(p1.x, p1.y)
                    pt2 = to_pixel(p2.x, p2.y)

                    cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)

        # 4. Draw Gesture Label
        if idx < len(self.gesture_history_left):
            gesture_l = self.gesture_history_left[idx]
            gesture_r = self.gesture_history_right[idx]
            text = ""
            if gesture_l: 
                text += f"L: {gesture_l[0]} ({gesture_l[1]:.2f}) "
            if gesture_r: 
                text += f"R: {gesture_r[0]} ({gesture_r[1]:.2f}) "
            
            if text:
                cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        # Resize to fit max canvas, preserving aspect ratio
        h, w = overlay.shape[:2]
        scale = min(self.max_canvas_width / w, self.max_canvas_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        scaled = cv2.resize(overlay, (new_w, new_h))
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(img)
        
        # Resizing Canvas to match image exactly to remove black bars
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def update_opacity(self, val):
        self.opacity = float(val) / 100
        self.update_frame(self.frame_slider.get())


if __name__ == "__main__":
    root = tk.Tk()
    app = TasksHandHeatmapApp(root)
    root.mainloop()
