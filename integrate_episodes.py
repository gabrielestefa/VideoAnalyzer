import sys

# Read the file
with open(r'd:\personal\OpenCV\VideoHandTracker\video_analysis_engine.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and modify process_video
modified = False
for i, line in enumerate(lines):
    if 'progress_callback(100, "Analysis complete")' in line and not modified:
        # Insert episode analysis before this line
        indent = line[:len(line) - len(line.lstrip())]
        lines.insert(i, f'{indent}progress_callback(95, "Analyzing episodes...")\n')
        lines.insert(i+1, f'{indent[:-4]}self.analyze_episodes()\n')
        lines.insert(i+2, f'{indent[:-4]}\n')
        modified = True
        break

# Find and modify load_analysis
for i, line in enumerate(lines):
    if 'self.regenerate_data()' in line and 'load_analysis' in ''.join(lines[max(0, i-20):i]):
        # Insert episode analysis after regenerate_data
        indent = line[:len(line) - len(line.lstrip())]
        lines.insert(i+1, f'\n')
        lines.insert(i+2, f'{indent}# Analyze episodes\n')
        lines.insert(i+3, f'{indent}self.analyze_episodes()\n')
        break

# Write back
with open(r'd:\personal\OpenCV\VideoHandTracker\video_analysis_engine.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Integration complete!")
