"""
Interactive Audio Wheel Analyzer
Professional audio transcript visualization with topic filtering,
timeline navigation, and playback controls.
"""

import os
import json
import tempfile
from moviepy import VideoFileClip

# Professional color palette matching reference design
TOPIC_COLORS = [
    "#e74c3c",  # Red
    "#f59e0b",  # Amber/Orange
    "#10b981",  # Emerald
    "#3b82f6",  # Blue
    "#8b5cf6",  # Violet
    "#ec4899",  # Pink
    "#14b8a6",  # Teal
    "#f97316",  # Orange
    "#6366f1",  # Indigo
    "#84cc16",  # Lime
    "#06b6d4",  # Cyan
    "#a855f7",  # Purple
]


def generate_audio_wheel_html(video_path, transcript_data, output_dir=None, 
                               video_title="Audio Analysis", topics=None):
    """
    Generate interactive audio wheel analyzer HTML.
    
    Args:
        video_path: Path to the video file
        transcript_data: List of tuples (start_time, end_time, text, category/topic)
        output_dir: Optional directory for output files (uses temp if None)
        video_title: Title to display for the audio analysis
        topics: Optional dict mapping topic_id to topic_label
    
    Returns:
        Path to the generated HTML file
    """
    if not transcript_data or len(transcript_data) == 0:
        return None
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_wheel_")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract audio from video
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None
        
        # Save full audio
        audio_path = os.path.join(output_dir, "full_audio.mp3")
        clip.audio.write_audiofile(audio_path, logger=None)
        
        duration = clip.duration
        clip.close()
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None
    
    # Format duration for display
    duration_mins = int(duration // 60)
    duration_secs = int(duration % 60)
    duration_str = f"{duration_mins}h {duration_secs:02d}m" if duration_mins >= 60 else f"{duration_mins}m {duration_secs:02d}s"
    
    # Extract unique topics from transcript data
    topic_set = set()
    sentences = []
    for i, item in enumerate(transcript_data):
        start, end, text = item[0], item[1], item[2]
        topic = item[3] if len(item) > 3 and item[3] else "General"
        topic_set.add(topic)
        sentences.append({
            'id': i,
            'start': float(start),
            'end': float(end),
            'text': text,
            'topic': topic
        })
    
    # Create topic list with colors
    # Ensure General topic is always grey and last in legend if possible
    unique_topics = sorted(list(topic_set))
    if "General" in unique_topics:
        unique_topics.remove("General")
        unique_topics.append("General")  # Move to end
        
    topic_colors = {}
    color_idx = 0
    for topic in unique_topics:
        if topic == "General":
            topic_colors[topic] = "#6c757d"  # Grey
        else:
            topic_colors[topic] = TOPIC_COLORS[color_idx % len(TOPIC_COLORS)]
            color_idx += 1
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{video_title} - Audio Analysis</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: radial-gradient(ellipse at 20% 50%, #1a1a2e 0%, #0f0f1a 40%, #050510 100%);
            min-height: 100vh;
            color: #e0e0e0;
            overflow-x: hidden;
        }}
        
        /* Subtle nebula effect */
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(ellipse at 80% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 20% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }}
        
        .container {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 280px 1fr 340px;
            min-height: 100vh;
            gap: 0;
        }}
        
        @media (max-width: 1400px) {{
            .container {{
                grid-template-columns: 240px 1fr 300px;
            }}
        }}
        
        @media (max-width: 1100px) {{
            .container {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Left Panel - Info & Topics */
        .info-panel {{
            padding: 32px 24px;
            background: rgba(12, 12, 20, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }}
        
        .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }}
        
        .brand-icon {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .brand-icon svg {{
            width: 18px;
            height: 18px;
            fill: #000;
        }}
        
        .brand-text {{
            font-size: 13px;
            color: #888;
        }}
        
        .brand-text strong {{
            display: block;
            color: #fff;
            font-size: 14px;
            font-weight: 600;
        }}
        
        .video-title {{
            font-size: 22px;
            font-weight: 600;
            color: #fff;
            margin-bottom: 8px;
            line-height: 1.35;
        }}
        
        .video-meta {{
            color: #666;
            font-size: 13px;
            margin-bottom: 24px;
        }}
        
        .section-label {{
            color: #f59e0b;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 12px;
            margin-top: 24px;
        }}
        
        .topics-list {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            flex: 1;
        }}
        
        .topic-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
            border: 1px solid transparent;
        }}
        
        .topic-item:hover {{
            background: rgba(255, 255, 255, 0.06);
        }}
        
        .topic-item.hidden {{
            opacity: 0.4;
        }}
        
        .topic-item.focused {{
            border-color: rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.08);
        }}
        
        .topic-color {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        
        .topic-name {{
            flex: 1;
            font-size: 13px;
            color: #ccc;
            white-space: normal;
            line-height: 1.4;
        }}
        
        .topic-count {{
            font-size: 11px;
            color: #666;
            background: rgba(255, 255, 255, 0.05);
            padding: 2px 8px;
            border-radius: 10px;
        }}
        
        .topic-hint {{
            font-size: 11px;
            color: #555;
            margin-top: 12px;
            line-height: 1.5;
        }}
        
        /* Center Panel - Wheel */
        .wheel-panel {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
            position: relative;
        }}
        
        .wheel-container {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        #wheelCanvas {{
            cursor: grab;
        }}
        
        #wheelCanvas:active {{
            cursor: grabbing;
        }}
        
        .center-info {{
            position: absolute;
            text-align: center;
            pointer-events: none;
        }}
        
        #currentTime {{
            font-size: 20px;
            font-weight: 600;
            color: #4ecdc4;
        }}
        
        #totalDuration {{
            font-size: 11px;
            color: #555;
            margin-top: 2px;
        }}
        
        /* Player Bar */
        .player-bar {{
            margin-top: 32px;
            background: rgba(20, 20, 35, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            gap: 16px;
            width: 100%;
            max-width: 560px;
        }}
        
        .play-button {{
            width: 42px;
            height: 42px;
            background: #4ecdc4;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
            flex-shrink: 0;
        }}
        
        .play-button:hover {{
            background: #45b8b0;
            transform: scale(1.04);
        }}
        
        .play-button svg {{
            width: 16px;
            height: 16px;
            fill: #0a0a14;
        }}
        
        .progress-container {{
            flex: 1;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .time-display {{
            font-size: 12px;
            color: #666;
            font-family: 'SF Mono', 'Consolas', monospace;
            min-width: 42px;
            text-align: center;
        }}
        
        .progress-bar {{
            flex: 1;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            position: relative;
            cursor: pointer;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4ecdc4 0%, #3db9b0 100%);
            border-radius: 2px;
            width: 0%;
            transition: width 0.1s linear;
        }}
        
        .progress-handle {{
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 12px;
            height: 12px;
            background: #4ecdc4;
            border-radius: 50%;
            cursor: grab;
            opacity: 0;
            transition: opacity 0.15s;
        }}
        
        .progress-bar:hover .progress-handle {{
            opacity: 1;
        }}
        
        .playback-controls {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .mode-toggle {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #888;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s;
            white-space: nowrap;
        }}
        
        .mode-toggle:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }}
        
        .mode-toggle.active {{
            background: rgba(78, 205, 196, 0.15);
            border-color: #4ecdc4;
            color: #4ecdc4;
        }}
        
        .speed-control {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #888;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            min-width: 48px;
            text-align: center;
        }}
        
        .speed-control:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }}
        
        /* Right Panel - Sentences */
        .sentences-panel {{
            padding: 0;
            background: rgba(12, 12, 20, 0.95);
            border-left: 1px solid rgba(255, 255, 255, 0.06);
            overflow: hidden;
            max-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .search-container {{
            padding: 24px 20px 12px 20px;
            flex-shrink: 0;
            background: rgba(12, 12, 20, 0.98);
            z-index: 10;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }}
        
        .search-box {{
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 10px 14px;
            gap: 10px;
        }}
        
        .search-icon {{
            width: 14px;
            height: 14px;
            opacity: 0.4;
        }}
        
        #searchInput {{
            flex: 1;
            background: transparent;
            border: none;
            color: #fff;
            font-size: 13px;
            outline: none;
        }}
        
        #searchInput::placeholder {{
            color: #555;
        }}
        
        .search-nav {{
            display: flex;
            gap: 4px;
            align-items: center;
        }}
        
        .search-nav button {{
            width: 24px;
            height: 24px;
            background: rgba(255, 255, 255, 0.06);
            border: none;
            border-radius: 4px;
            color: #666;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .search-nav button:hover:not(:disabled) {{
            background: rgba(255, 255, 255, 0.12);
            color: #fff;
        }}
        
        .search-nav button:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}
        
        #matchCounter {{
            font-size: 11px;
            color: #555;
            margin-left: 6px;
            min-width: 30px;
        }}
        
        .sentences-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex: 1;
            overflow-y: auto;
            padding: 12px 20px 24px 20px;
        }}
        
        .sentence-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 8px;
            padding: 12px 14px;
            cursor: pointer;
            transition: all 0.15s;
            position: relative;
        }}
        
        .sentence-card:hover {{
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.08);
        }}
        
        .sentence-card.selected {{
            background: rgba(78, 205, 196, 0.08);
            border-color: rgba(78, 205, 196, 0.3);
        }}
        
        .sentence-card.match {{
            border-left: 2px solid #f59e0b;
        }}
        
        .sentence-card.hidden {{
            display: none;
        }}
        
        .sentence-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }}
        
        .sentence-topic-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        
        .sentence-time {{
            font-size: 11px;
            color: #4ecdc4;
            font-weight: 500;
        }}
        
        .sentence-text {{
            font-size: 13px;
            color: #bbb;
            line-height: 1.5;
        }}
        
        .sentence-card.playing .sentence-time::before {{
            content: 'Playing ';
            color: #4ecdc4;
        }}
        
        /* Scrollbar */
        .sentences-panel::-webkit-scrollbar,
        .info-panel::-webkit-scrollbar {{
            width: 4px;
        }}
        
        .sentences-panel::-webkit-scrollbar-track,
        .info-panel::-webkit-scrollbar-track {{
            background: transparent;
        }}
        
        .sentences-panel::-webkit-scrollbar-thumb,
        .info-panel::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 2px;
        }}
        
        /* Hidden audio */
        #audioPlayer {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Left Panel -->
        <div class="info-panel">
            <div class="brand">
                <div class="brand-icon">
                    <svg viewBox="0 0 24 24"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
                </div>
                <div class="brand-text">
                    <strong>Audio Analyzer</strong>
                    Transcript Visualization
                </div>
            </div>
            
            <h1 class="video-title">{video_title}</h1>
            <div class="video-meta">{duration_str} &middot; {len(sentences)} segments</div>
            
            <div class="section-label">Topics</div>
            <div class="topics-list" id="topicsList">
                <!-- Populated by JS -->
            </div>
            <div class="topic-hint">
                Click to hide/show. Double-click to focus.
            </div>
        </div>
        
        <!-- Center Panel -->
        <div class="wheel-panel">
            <div class="wheel-container">
                <canvas id="wheelCanvas" width="520" height="520"></canvas>
                <div class="center-info">
                    <div id="currentTime">00:00</div>
                    <div id="totalDuration">{duration_str}</div>
                </div>
            </div>
            
            <div class="player-bar">
                <button class="play-button" id="playBtn" title="Play/Pause">
                    <svg viewBox="0 0 24 24" id="playIcon">
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </button>
                <div class="progress-container">
                    <span class="time-display" id="elapsedTime">00:00</span>
                    <div class="progress-bar" id="progressBar">
                        <div class="progress-fill" id="progressFill"></div>
                        <div class="progress-handle" id="progressHandle" style="left: 0%"></div>
                    </div>
                    <span class="time-display" id="remainingTime">{int(duration//60):02d}:{int(duration%60):02d}</span>
                </div>
                <div class="playback-controls">
                    <button class="mode-toggle" id="modeToggle" title="Toggle continuous playback">Segment</button>
                    <button class="speed-control" id="speedBtn" title="Playback speed">1x</button>
                </div>
            </div>
        </div>
        
        <!-- Right Panel -->
        <div class="sentences-panel">
            <div class="search-container">
                <div class="search-box">
                    <svg class="search-icon" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                    </svg>
                    <input type="text" id="searchInput" placeholder="Search transcript..." />
                    <div class="search-nav">
                        <button id="prevBtn" title="Previous">&#8593;</button>
                        <button id="nextBtn" title="Next">&#8595;</button>
                        <span id="matchCounter"></span>
                    </div>
                </div>
            </div>
            
            <div class="sentences-list" id="sentencesList"></div>
        </div>
    </div>
    
    <audio id="audioPlayer">
        <source src="full_audio.mp3" type="audio/mpeg">
    </audio>
    
    <script>
        // Data
        const sentences = {json.dumps(sentences)};
        const topicColors = {json.dumps(topic_colors)};
        const totalDuration = {duration};
        
        // State
        let currentRotation = 0;
        let selectedSentence = null;
        let searchMatches = [];
        let currentMatchIndex = -1;
        let isDragging = false;
        let lastMouseAngle = 0;
        let isPlaying = false;
        let currentSentenceEndTime = null;
        let continuousMode = false;
        let hiddenTopics = new Set();
        let focusedTopic = null;
        
        // Elements
        const canvas = document.getElementById('wheelCanvas');
        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const outerRadius = 220;
        const innerRadius = 100;
        
        const audioPlayer = document.getElementById('audioPlayer');
        const searchInput = document.getElementById('searchInput');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const matchCounter = document.getElementById('matchCounter');
        const sentencesList = document.getElementById('sentencesList');
        const topicsList = document.getElementById('topicsList');
        const playBtn = document.getElementById('playBtn');
        const playIcon = document.getElementById('playIcon');
        const progressFill = document.getElementById('progressFill');
        const progressHandle = document.getElementById('progressHandle');
        const progressBar = document.getElementById('progressBar');
        const elapsedTime = document.getElementById('elapsedTime');
        const remainingTime = document.getElementById('remainingTime');
        const currentTimeDiv = document.getElementById('currentTime');
        const speedBtn = document.getElementById('speedBtn');
        const modeToggle = document.getElementById('modeToggle');
        
        const speeds = [1, 1.25, 1.5, 2];
        let currentSpeedIndex = 0;
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Build topics list
        function buildTopicsList() {{
            const topicCounts = {{}};
            sentences.forEach(s => {{
                topicCounts[s.topic] = (topicCounts[s.topic] || 0) + 1;
            }});
            
            topicsList.innerHTML = '';
            Object.keys(topicColors).sort().forEach(topic => {{
                const item = document.createElement('div');
                item.className = 'topic-item';
                item.dataset.topic = topic;
                item.innerHTML = `
                    <div class="topic-color" style="background: ${{topicColors[topic]}}"></div>
                    <span class="topic-name">${{topic}}</span>
                    <span class="topic-count">${{topicCounts[topic] || 0}}</span>
                `;
                
                // Single click: toggle hide
                item.addEventListener('click', (e) => {{
                    if (e.detail === 1) {{
                        setTimeout(() => {{
                            if (e.detail === 1) toggleTopic(topic);
                        }}, 200);
                    }}
                }});
                
                // Double click: focus
                item.addEventListener('dblclick', () => {{
                    focusTopic(topic);
                }});
                
                topicsList.appendChild(item);
            }});
        }}
        
        function toggleTopic(topic) {{
            if (focusedTopic) {{
                focusedTopic = null;
                hiddenTopics.clear();
            }}
            
            if (hiddenTopics.has(topic)) {{
                hiddenTopics.delete(topic);
            }} else {{
                hiddenTopics.add(topic);
            }}
            updateVisibility();
        }}
        
        function focusTopic(topic) {{
            if (focusedTopic === topic) {{
                focusedTopic = null;
                hiddenTopics.clear();
            }} else {{
                focusedTopic = topic;
                hiddenTopics.clear();
                Object.keys(topicColors).forEach(t => {{
                    if (t !== topic) hiddenTopics.add(t);
                }});
            }}
            updateVisibility();
        }}
        
        function updateVisibility() {{
            // Update topic items
            document.querySelectorAll('.topic-item').forEach(item => {{
                const topic = item.dataset.topic;
                item.classList.toggle('hidden', hiddenTopics.has(topic));
                item.classList.toggle('focused', focusedTopic === topic);
            }});
            
            // Update sentence cards
            document.querySelectorAll('.sentence-card').forEach(card => {{
                const idx = parseInt(card.dataset.index);
                const sentence = sentences[idx];
                const isHidden = hiddenTopics.has(sentence.topic);
                card.classList.toggle('hidden', isHidden);
            }});
            
            drawWheel();
        }}
        
        // Build sentence cards
        function buildSentenceCards() {{
            sentencesList.innerHTML = '';
            sentences.forEach((sentence, index) => {{
                const card = document.createElement('div');
                card.className = 'sentence-card';
                card.dataset.index = index;
                card.innerHTML = `
                    <div class="sentence-header">
                        <div class="sentence-topic-dot" style="background: ${{topicColors[sentence.topic]}}"></div>
                        <span class="sentence-time">${{formatTime(sentence.start)}}</span>
                    </div>
                    <div class="sentence-text">${{sentence.text}}</div>
                `;
                card.addEventListener('click', () => selectSentence(index));
                sentencesList.appendChild(card);
            }});
        }}
        
        function drawWheel() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(centerX, centerY);
            
            // Outer ring
            ctx.beginPath();
            ctx.arc(0, 0, outerRadius + 12, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            // Time labels
            for (let i = 0; i < 12; i++) {{
                const angle = (i / 12) * Math.PI * 2 - Math.PI / 2 + currentRotation;
                const timeValue = (i / 12) * totalDuration;
                const x = Math.cos(angle) * (outerRadius + 28);
                const y = Math.sin(angle) * (outerRadius + 28);
                
                ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(formatTime(timeValue), x, y);
            }}
            
            ctx.restore();
            
            // Sentence dots and arcs
            const isSearchActive = searchInput.value.trim().length > 0;
            
            // Sentence dots and arcs
            sentences.forEach((sentence, index) => {{
                if (hiddenTopics.has(sentence.topic)) return;
                
                const startAngle = (sentence.start / totalDuration) * Math.PI * 2 - Math.PI / 2 + currentRotation;
                const endAngle = (sentence.end / totalDuration) * Math.PI * 2 - Math.PI / 2 + currentRotation;
                const midAngle = (startAngle + endAngle) / 2;
                
                const layerOffset = (index % 4) * 12;
                const segmentRadius = outerRadius - 15 - layerOffset;
                const dotX = centerX + Math.cos(midAngle) * segmentRadius;
                const dotY = centerY + Math.sin(midAngle) * segmentRadius;
                
                const color = topicColors[sentence.topic] || '#4ecdc4';
                let dotSize = 5;
                let opacity = 0.7;
                
                if (isSearchActive) {{
                    if (searchMatches.includes(index)) {{
                        dotSize = 7;
                        opacity = 1;
                    }} else if (selectedSentence !== index) {{
                         opacity = 0.1;
                    }}
                }} else if (searchMatches.includes(index)) {{
                    dotSize = 7;
                    opacity = 1;
                }}
                
                if (selectedSentence === index) {{
                    dotSize = 9;
                    opacity = 1;
                    
                    // Glow
                    ctx.beginPath();
                    ctx.arc(dotX, dotY, dotSize + 6, 0, Math.PI * 2);
                    ctx.fillStyle = color.replace(')', ', 0.2)').replace('rgb', 'rgba');
                    ctx.fill();
                }}
                
                // Dot
                ctx.beginPath();
                ctx.arc(dotX, dotY, dotSize, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.globalAlpha = opacity;
                ctx.fill();
                ctx.globalAlpha = 1;
                
                // Arc
                ctx.beginPath();
                ctx.arc(centerX, centerY, segmentRadius, startAngle, endAngle);
                ctx.strokeStyle = selectedSentence === index ? color : `${{color}}50`;
                ctx.lineWidth = selectedSentence === index ? 2 : 1;
                ctx.stroke();
            }});
            
            // Center
            ctx.beginPath();
            ctx.arc(centerX, centerY, innerRadius - 8, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(10, 10, 20, 0.95)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(78, 205, 196, 0.2)';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            // Playhead indicator
            if (isPlaying || selectedSentence !== null) {{
                const progress = audioPlayer.currentTime / totalDuration;
                const indicatorAngle = progress * Math.PI * 2 - Math.PI / 2 + currentRotation;
                const ix = centerX + Math.cos(indicatorAngle) * (innerRadius + 15);
                const iy = centerY + Math.sin(indicatorAngle) * (innerRadius + 15);
                
                ctx.beginPath();
                ctx.arc(ix, iy, 3, 0, Math.PI * 2);
                ctx.fillStyle = '#4ecdc4';
                ctx.fill();
            }}
        }}
        
        function selectSentence(index) {{
            selectedSentence = index;
            const sentence = sentences[index];
            
            document.querySelectorAll('.sentence-card').forEach((card, i) => {{
                card.classList.toggle('selected', i === index);
                card.classList.remove('playing');
            }});
            
            const selectedCard = document.querySelector(`.sentence-card[data-index="${{index}}"]`);
            if (selectedCard) {{
                selectedCard.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
            
            if (!continuousMode) {{
                currentSentenceEndTime = sentence.end;
            }} else {{
                currentSentenceEndTime = null;
            }}
            
            audioPlayer.currentTime = sentence.start;
            audioPlayer.play();
            isPlaying = true;
            updatePlayButton();
            drawWheel();
        }}
        
        function updatePlayButton() {{
            playIcon.innerHTML = isPlaying 
                ? '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>'
                : '<path d="M8 5v14l11-7z"/>';
        }}
        
        function updateProgress() {{
            const current = audioPlayer.currentTime;
            const progress = current / totalDuration;
            
            progressFill.style.width = `${{progress * 100}}%`;
            progressHandle.style.left = `${{progress * 100}}%`;
            
            elapsedTime.textContent = formatTime(current);
            remainingTime.textContent = formatTime(totalDuration - current);
            currentTimeDiv.textContent = formatTime(current);
            
            // Update currently playing sentence
            let currentIdx = -1;
            for (let i = 0; i < sentences.length; i++) {{
                if (current >= sentences[i].start && current < sentences[i].end) {{
                    currentIdx = i;
                    break;
                }}
            }}
            
            if (currentIdx !== -1 && currentIdx !== selectedSentence) {{
                selectedSentence = currentIdx;
                document.querySelectorAll('.sentence-card').forEach((card, i) => {{
                    card.classList.toggle('selected', i === currentIdx);
                }});
            }}
            
            document.querySelectorAll('.sentence-card').forEach((card, i) => {{
                card.classList.toggle('playing', i === selectedSentence && isPlaying);
            }});
            
            drawWheel();
        }}
        
        audioPlayer.addEventListener('timeupdate', () => {{
            updateProgress();
            
            // Segment mode: stop at end
            if (!continuousMode && currentSentenceEndTime !== null && audioPlayer.currentTime >= currentSentenceEndTime) {{
                audioPlayer.pause();
                isPlaying = false;
                updatePlayButton();
                currentSentenceEndTime = null;
            }}
        }});
        
        audioPlayer.addEventListener('ended', () => {{
            isPlaying = false;
            updatePlayButton();
        }});
        
        playBtn.addEventListener('click', () => {{
            if (selectedSentence === null && sentences.length > 0) {{
                selectSentence(0);
                return;
            }}
            
            if (isPlaying) {{
                audioPlayer.pause();
                isPlaying = false;
            }} else {{
                if (!continuousMode && selectedSentence !== null) {{
                    const sentence = sentences[selectedSentence];
                    if (audioPlayer.currentTime >= sentence.end || audioPlayer.currentTime < sentence.start) {{
                        audioPlayer.currentTime = sentence.start;
                    }}
                    currentSentenceEndTime = sentence.end;
                }}
                audioPlayer.play();
                isPlaying = true;
            }}
            updatePlayButton();
        }});
        
        modeToggle.addEventListener('click', () => {{
            continuousMode = !continuousMode;
            modeToggle.textContent = continuousMode ? 'Continuous' : 'Segment';
            modeToggle.classList.toggle('active', continuousMode);
            if (continuousMode) {{
                currentSentenceEndTime = null;
            }}
        }});
        
        speedBtn.addEventListener('click', () => {{
            currentSpeedIndex = (currentSpeedIndex + 1) % speeds.length;
            const speed = speeds[currentSpeedIndex];
            audioPlayer.playbackRate = speed;
            speedBtn.textContent = speed === 1 ? '1x' : speed + 'x';
        }});
        
        progressBar.addEventListener('click', (e) => {{
            const rect = progressBar.getBoundingClientRect();
            const percentage = (e.clientX - rect.left) / rect.width;
            audioPlayer.currentTime = percentage * totalDuration;
            
            // Find sentence at this time
            for (let i = 0; i < sentences.length; i++) {{
                if (audioPlayer.currentTime >= sentences[i].start && audioPlayer.currentTime < sentences[i].end) {{
                    selectedSentence = i;
                    if (!continuousMode) {{
                        currentSentenceEndTime = sentences[i].end;
                    }}
                    break;
                }}
            }}
            updateProgress();
        }});
        
        // Canvas interactions
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const dx = x - centerX;
            const dy = y - centerY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance >= innerRadius && distance <= outerRadius + 15) {{
                const clickAngle = Math.atan2(dy, dx) + Math.PI / 2 - currentRotation;
                const normalizedAngle = ((clickAngle % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
                const clickTime = (normalizedAngle / (Math.PI * 2)) * totalDuration;
                
                let closestIdx = 0;
                let closestDist = Infinity;
                sentences.forEach((s, i) => {{
                    if (hiddenTopics.has(s.topic)) return;
                    const mid = (s.start + s.end) / 2;
                    const dist = Math.abs(clickTime - mid);
                    if (dist < closestDist) {{
                        closestDist = dist;
                        closestIdx = i;
                    }}
                }});
                selectSentence(closestIdx);
            }}
        }});
        
        canvas.addEventListener('mousedown', (e) => {{
            const rect = canvas.getBoundingClientRect();
            lastMouseAngle = Math.atan2(e.clientY - rect.top - centerY, e.clientX - rect.left - centerX);
            isDragging = true;
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            const rect = canvas.getBoundingClientRect();
            const angle = Math.atan2(e.clientY - rect.top - centerY, e.clientX - rect.left - centerX);
            currentRotation += angle - lastMouseAngle;
            lastMouseAngle = angle;
            drawWheel();
        }});
        
        canvas.addEventListener('mouseup', () => isDragging = false);
        canvas.addEventListener('mouseleave', () => isDragging = false);
        // Mouse wheel listener removed to allow page scrolling
        
        // Search
        searchInput.addEventListener('input', performSearch);
        
        function performSearch() {{
            const query = searchInput.value.toLowerCase().trim();
            searchMatches = [];
            currentMatchIndex = -1;
            
            if (query.length > 0) {{
                sentences.forEach((s, i) => {{
                    if (s.text.toLowerCase().includes(query) && !hiddenTopics.has(s.topic)) {{
                        searchMatches.push(i);
                    }}
                }});
            }}
            
            document.querySelectorAll('.sentence-card').forEach((card, i) => {{
                card.classList.toggle('match', searchMatches.includes(i));
            }});
            
            matchCounter.textContent = searchMatches.length > 0 ? searchMatches.length : '';
            prevBtn.disabled = searchMatches.length === 0;
            nextBtn.disabled = searchMatches.length === 0;
            drawWheel();
        }}
        
        prevBtn.addEventListener('click', () => {{
            if (searchMatches.length === 0) return;
            currentMatchIndex = currentMatchIndex <= 0 ? searchMatches.length - 1 : currentMatchIndex - 1;
            selectSentence(searchMatches[currentMatchIndex]);
            matchCounter.textContent = `${{currentMatchIndex + 1}}/${{searchMatches.length}}`;
        }});
        
        nextBtn.addEventListener('click', () => {{
            if (searchMatches.length === 0) return;
            currentMatchIndex = currentMatchIndex >= searchMatches.length - 1 ? 0 : currentMatchIndex + 1;
            selectSentence(searchMatches[currentMatchIndex]);
            matchCounter.textContent = `${{currentMatchIndex + 1}}/${{searchMatches.length}}`;
        }});
        
        // Initialize
        buildTopicsList();
        buildSentenceCards();
        drawWheel();
    </script>
</body>
</html>
"""
    
    # Write HTML file
    html_path = os.path.join(output_dir, "audio_wheel_analyzer.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Audio wheel analyzer generated: {html_path}")
    return html_path
