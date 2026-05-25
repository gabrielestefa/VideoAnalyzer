import sys

# Read the file
with open(r'd:\personal\OpenCV\VideoHandTracker\video_analysis_engine.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find load_analysis method and add analyze_episodes after regenerate_data
for i, line in enumerate(lines):
    if 'self.regenerate_data()' in line:
        # Check if we're in load_analysis (look back for the method definition)
        context = ''.join(lines[max(0, i-40):i])
        if 'def load_analysis' in context:
            # Check if analyze_episodes() isn't already called nearby
            future_context = ''.join(lines[i:min(len(lines), i+10)])
            if 'analyze_episodes' not in future_context:
                indent = '            '  # Same as regenerate_data
                lines.insert(i+2, f'\n')
                lines.insert(i+3, f'{indent}# Analyze episodes\n')
                lines.insert(i+4, f'{indent}self.analyze_episodes()\n')
                print(f"Added analyze_episodes() call after line {i+1}")
                break

# Write back
with open(r'd:\personal\OpenCV\VideoHandTracker\video_analysis_engine.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Integration complete!")
