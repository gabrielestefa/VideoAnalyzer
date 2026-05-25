"""
Interactive Audio Wheel Analyzer
Generates an HTML interface with a central circular wheel showing transcript sentences
arranged by timeline position, with search functionality and audio playback.
"""

import os
import json
import tempfile
from moviepy import VideoFileClip


def generate_audio_wheel_html(video_path, transcript_data, output_dir=None):
    """
    Generate interactive audio wheel analyzer HTML.
    
    Args:
        video_path: Path to the video file
        transcript_data: List of tuples (start_time, end_time, text, category)
        output_dir: Optional directory for output files (uses temp if None)
    
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
    
    # Prepare transcript data for JSON
    sentences = []
    for i, (start, end, text, category) in enumerate(transcript_data):
        sentences.append({
            'id': i,
            'start': float(start),
            'end': float(end),
            'text': text,
            'category': category if category else 'Neutral'
        })
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Audio Analyzer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            overflow-x: hidden;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .search-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        #searchInput {{
            flex: 1;
            max-width: 500px;
            padding: 15px 20px;
            border: 2px solid #667eea;
            border-radius: 10px;
            font-size: 16px;
            outline: none;
            transition: all 0.3s ease;
        }}
        
        #searchInput:focus {{
            border-color: #764ba2;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }}
        
        .nav-buttons {{
            display: flex;
            gap: 10px;
        }}
        
        .nav-buttons button {{
            width: 45px;
            height: 45px;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            cursor: pointer;
            font-size: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .nav-buttons button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }}
        
        .nav-buttons button:active {{
            transform: translateY(0);
        }}
        
        .nav-buttons button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        #matchCounter {{
            font-weight: bold;
            color: #667eea;
            font-size: 14px;
            min-width: 100px;
            text-align: center;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        @media (max-width: 1200px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .wheel-container {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 600px;
            background: #f9f9f9;
            border-radius: 15px;
            padding: 40px;
            box-shadow: inset 0 4px 15px rgba(0,0,0,0.05);
        }}
        
        #wheelCanvas {{
            cursor: grab;
            border-radius: 50%;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
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
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        #totalDuration {{
            font-size: 16px;
            color: #666;
        }}
        
        .details-panel {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .details-panel h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        #sentenceText {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            min-height: 100px;
            margin-bottom: 20px;
            color: #333;
            line-height: 1.6;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        #timestamp {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        #audioPlayer {{
            width: 100%;
            margin-top: 15px;
            border-radius: 10px;
            outline: none;
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        .legend-label {{
            font-size: 14px;
            color: #666;
        }}
        
        .match-highlight {{
            animation: pulse 1.5s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: #667eea;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Interactive Audio Analyzer</h1>
        
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="🔍 Search transcript..." />
            <div class="nav-buttons">
                <button id="prevBtn" title="Previous Match">↑</button>
                <button id="nextBtn" title="Next Match">↓</button>
            </div>
            <span id="matchCounter"></span>
        </div>
        
        <div class="main-content">
            <div class="wheel-container">
                <canvas id="wheelCanvas" width="600" height="600"></canvas>
                <div class="center-info">
                    <div id="currentTime">00:00</div>
                    <div id="totalDuration">00:00</div>
                </div>
            </div>
            
            <div class="details-panel">
                <h3>Selected Sentence</h3>
                <div id="timestamp"></div>
                <p id="sentenceText">Click on a segment in the wheel to view the sentence and play audio.</p>
                <audio id="audioPlayer" controls>
                    <source src="full_audio.mp3" type="audio/mpeg">
                    Your browser does not support the audio element.
                </audio>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #3498db;"></div>
                <span class="legend-label">Normal</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2ecc71;"></div>
                <span class="legend-label">Selected</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f39c12;"></div>
                <span class="legend-label">Search Match</span>
            </div>
        </div>
    </div>
    
    <script>
        // Transcript data
        const sentences = {json.dumps(sentences, indent=8)};
        const totalDuration = {duration};
        
        // State
        let currentRotation = 0;
        let selectedSentence = null;
        let searchMatches = [];
        let currentMatchIndex = -1;
        let isDragging = false;
        let lastMouseAngle = 0;
        
        // Canvas setup
        const canvas = document.getElementById('wheelCanvas');
        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 250;
        const innerRadius = 180;
        
        // Audio player
        const audioPlayer = document.getElementById('audioPlayer');
        let currentSentenceEnd = null;
        
        // Monitor audio playback and pause at sentence end
        audioPlayer.addEventListener('timeupdate', () => {
            if (currentSentenceEnd !== null && audioPlayer.currentTime >= currentSentenceEnd) {
                audioPlayer.pause();
            }
        });
        
        // UI elements
        const searchInput = document.getElementById('searchInput');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const matchCounter = document.getElementById('matchCounter');
        const sentenceText = document.getElementById('sentenceText');
        const timestampDiv = document.getElementById('timestamp');
        const currentTimeDiv = document.getElementById('currentTime');
        const totalDurationDiv = document.getElementById('totalDuration');
        
        // Format time helper
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Initialize
        totalDurationDiv.textContent = formatTime(totalDuration);
        
        // Draw the wheel
        function drawWheel() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (sentences.length === 0) return;
            
            sentences.forEach((sentence, index) => {{
                const startAngle = (sentence.start / totalDuration) * Math.PI * 2 - Math.PI / 2 + currentRotation;
                const endAngle = (sentence.end / totalDuration) * Math.PI * 2 - Math.PI / 2 + currentRotation;
                
                // Determine color
                let color = '#3498db'; // Default blue
                
                if (searchMatches.includes(index)) {{
                    color = '#f39c12'; // Orange for matches
                }}
                
                if (selectedSentence === index) {{
                    color = '#2ecc71'; // Green for selected
                }}
                
                // Draw segment
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, startAngle, endAngle);
                ctx.arc(centerX, centerY, innerRadius, endAngle, startAngle, true);
                ctx.closePath();
                ctx.fillStyle = color;
                ctx.fill();
                
                // Draw border
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw text label (for larger segments)
                const segmentAngle = endAngle - startAngle;
                if (segmentAngle > 0.1) {{ // Only draw text if segment is large enough
                    const midAngle = (startAngle + endAngle) / 2;
                    const textRadius = (radius + innerRadius) / 2;
                    const textX = centerX + Math.cos(midAngle) * textRadius;
                    const textY = centerY + Math.sin(midAngle) * textRadius;
                    
                    ctx.save();
                    ctx.translate(textX, textY);
                    ctx.rotate(midAngle + Math.PI / 2);
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 12px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    
                    const label = `${{index + 1}}`;
                    ctx.fillText(label, 0, 0);
                    ctx.restore();
                }}
            }});
            
            // Draw center circle
            ctx.beginPath();
            ctx.arc(centerX, centerY, innerRadius - 10, 0, Math.PI * 2);
            ctx.fillStyle = 'white';
            ctx.fill();
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 3;
            ctx.stroke();
            
            // Update current time display
            const normalizedRotation = ((currentRotation % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
            const currentTimeSeconds = (normalizedRotation / (Math.PI * 2)) * totalDuration;
            currentTimeDiv.textContent = formatTime(currentTimeSeconds);
        }}
        
        // Handle canvas click
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const dx = x - centerX;
            const dy = y - centerY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Check if click is within the wheel
            if (distance >= innerRadius && distance <= radius) {{
                const clickAngle = Math.atan2(dy, dx) + Math.PI / 2 - currentRotation;
                const normalizedAngle = ((clickAngle % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
                const clickTime = (normalizedAngle / (Math.PI * 2)) * totalDuration;
                
                // Find the sentence at this time
                for (let i = 0; i < sentences.length; i++) {{
                    if (clickTime >= sentences[i].start && clickTime <= sentences[i].end) {{
                        selectSentence(i);
                        break;
                    }}
                }}
            }}
        }});
        
        // Select a sentence
        function selectSentence(index) {{
            selectedSentence = index;
            const sentence = sentences[index];
            
            // Update UI
            sentenceText.textContent = sentence.text;
            timestampDiv.textContent = `${{formatTime(sentence.start)}} - ${{formatTime(sentence.end)}}`;
            
            // Set sentence end time for auto-pause
            currentSentenceEnd = sentence.end;
            
            // Seek audio to sentence start and play
            audioPlayer.currentTime = sentence.start;
            audioPlayer.play();
            
            drawWheel();
        }}
        
        // Mouse drag to rotate
        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const dx = x - centerX;
            const dy = y - centerY;
            lastMouseAngle = Math.atan2(dy, dx);
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const dx = x - centerX;
            const dy = y - centerY;
            const currentMouseAngle = Math.atan2(dy, dx);
            
            const deltaAngle = currentMouseAngle - lastMouseAngle;
            currentRotation += deltaAngle;
            lastMouseAngle = currentMouseAngle;
            
            drawWheel();
        }});
        
        canvas.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        canvas.addEventListener('mouseleave', () => {{
            isDragging = false;
        }});
        
        // Touch support
        canvas.addEventListener('touchstart', (e) => {{
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            const dx = x - centerX;
            const dy = y - centerY;
            lastMouseAngle = Math.atan2(dy, dx);
            isDragging = true;
        }});
        
        canvas.addEventListener('touchmove', (e) => {{
            e.preventDefault();
            if (!isDragging) return;
            
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            const dx = x - centerX;
            const dy = y - centerY;
            const currentMouseAngle = Math.atan2(dy, dx);
            
            const deltaAngle = currentMouseAngle - lastMouseAngle;
            currentRotation += deltaAngle;
            lastMouseAngle = currentMouseAngle;
            
            drawWheel();
        }});
        
        canvas.addEventListener('touchend', () => {{
            isDragging = false;
        }});
        
        // Mouse wheel for scrolling
        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            currentRotation += e.deltaY * 0.001;
            drawWheel();
        }});
        
        // Search functionality
        searchInput.addEventListener('input', performSearch);
        
        function performSearch() {{
            const query = searchInput.value.toLowerCase().trim();
            searchMatches = [];
            currentMatchIndex = -1;
            
            if (query.length > 0) {{
                sentences.forEach((sentence, index) => {{
                    if (sentence.text.toLowerCase().includes(query)) {{
                        searchMatches.push(index);
                    }}
                }});
            }}
            
            updateMatchCounter();
            updateNavigationButtons();
            drawWheel();
        }}
        
        function updateMatchCounter() {{
            if (searchMatches.length === 0) {{
                matchCounter.textContent = searchInput.value.length > 0 ? 'No matches' : '';
            }} else {{
                const current = currentMatchIndex >= 0 ? currentMatchIndex + 1 : 0;
                matchCounter.textContent = `${{current}} of ${{searchMatches.length}} matches`;
            }}
        }}
        
        function updateNavigationButtons() {{
            prevBtn.disabled = searchMatches.length === 0;
            nextBtn.disabled = searchMatches.length === 0;
        }}
        
        // Navigation buttons
        prevBtn.addEventListener('click', () => {{
            if (searchMatches.length === 0) return;
            
            currentMatchIndex--;
            if (currentMatchIndex < 0) {{
                currentMatchIndex = searchMatches.length - 1; // Wrap around
            }}
            
            const matchIndex = searchMatches[currentMatchIndex];
            selectSentence(matchIndex);
            updateMatchCounter();
        }});
        
        nextBtn.addEventListener('click', () => {{
            if (searchMatches.length === 0) return;
            
            currentMatchIndex++;
            if (currentMatchIndex >= searchMatches.length) {{
                currentMatchIndex = 0; // Wrap around
            }}
            
            const matchIndex = searchMatches[currentMatchIndex];
            selectSentence(matchIndex);
            updateMatchCounter();
        }});
        
        // Initial draw
        drawWheel();
    </script>
</body>
</html>
"""
    
    # Write HTML file
    html_path = os.path.join(output_dir, "audio_wheel_analyzer.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Audio wheel analyzer generated: {html_path}")
    return html_path
