import sys

# Read the file
with open(r'd:\personal\OpenCV\VideoHandTracker\hand_heatmap_modern.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with tab_semantic plotting
for i, line in enumerate(lines):
    if 'self.plot_semantic_comparison(tab_semantic, loaded_objects)' in line:
        # Insert episode tabs integration after this line
        indent = line[:len(line) - len(line.lstrip())]
        
        lines.insert(i+1, f'\n')
        lines.insert(i+2, f'{indent}# --- EPISODE ANALYSIS TABS (NEW) ---\n')
        lines.insert(i+3, f'{indent}try:\n')
        lines.insert(i+4, f'{indent}    from episode_comparison import add_episode_tabs\n')
        lines.insert(i+5, f'{indent}    add_episode_tabs(tab_view, loaded_objects, comp_win)\n')
        lines.insert(i+6, f'{indent}except Exception as e:\n')
        lines.insert(i+7, f'{indent}    print(f"Episode tabs integration failed: {{e}}")\n')
        break

# Write back
with open(r'd:\personal\OpenCV\VideoHandTracker\hand_heatmap_modern.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Comparison window integration complete!")
