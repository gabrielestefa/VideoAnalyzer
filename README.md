# Hand Tracking Analysis Application

A comprehensive hand tracking and gesture recognition application with advanced analytics and comparison features.

## Features
- Real-time hand tracking and gesture recognition
- Video analysis with heatmaps, traces, and skeletons
- Audio transcription and sentiment analysis
- Interactive comparison of multiple analysis sessions
- Plotly-based interactive visualizations

## Requirements
- Python 3.11+
- Windows 10/11 (tested)

## Installation

### Quick Setup
1. Clone or download this repository
2. Run the setup script:
   ```
   setup.bat
   ```

### Manual Setup
1. Install Python 3.11+ from [python.org](https://python.org)
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the application:
```
python hand_heatmap_modern.py
```

## Key Dependencies
- **OpenCV** - Video processing
- **MediaPipe** - Hand tracking and gesture recognition
- **Plotly** - Interactive visualizations
- **CustomTkinter** - Modern UI framework
- **SpeechRecognition** - Audio transcription

## Project Structure
```
VideoHandTracker/
├── hand_heatmap_modern.py    # Main application
├── video_analysis_engine.py  # Analysis engine
├── requirements.txt          # Python dependencies
├── setup.bat                # Installation script
├── Models/                  # ML models (gesture, text)
└── Library/                # Saved analysis sessions
```

## Support
For issues or questions, please check the documentation or create an issue.
