"""
Episode Comparison Integration Module

This module provides ready-to-use methods to add episode-based comparison tabs
to the existing hand_heatmap_modern.py comparison window.

Usage:
    from episode_comparison import add_episode_tabs
    add_episode_tabs(tab_view, loaded_objects)
"""

import customtkinter as ctk
import tkinter as tk
import numpy as np
from typing import List, Dict
from episode_viz import (
    create_episode_timeline,
    create_episode_clustering_view,
    create_interaction_metrics_comparison,
    create_speech_motion_alignment_view
)


def add_episode_tabs(tab_view, loaded_objects, parent_window):
    """
    Add episode analysis tabs to existing comparison tab view
    
    Args:
        tab_view: CTkTabview instance
        loaded_objects: List of dicts with 'name' and 'engine' keys
        parent_window: Parent window for embedding Plotly figures
    """
    # Add new tabs
    tab_episodes = tab_view.add("Episode Timeline")
    tab_patterns = tab_view.add("Interaction Patterns")
    tab_metrics = tab_view.add("Episode Metrics")
    
    # Analyze episodes for all loaded sessions (if not already done)
    for obj in loaded_objects:
        eng = obj['engine']
        if not hasattr(eng, 'episodes_left') or not eng.episodes_left:
            try:
                eng.analyze_episodes()
            except Exception as e:
                print(f"Episode analysis failed for {obj['name']}: {e}")
    
    # Plot Episode Timeline
    if parent_window:
        plot_episode_timeline_tab(tab_episodes, loaded_objects, parent_window)
        plot_interaction_patterns_tab(tab_patterns, loaded_objects, parent_window)
        plot_episode_metrics_tab(tab_metrics, loaded_objects, parent_window)

def get_episode_figures(loaded_objects) -> Dict[str, object]:
    """Get all episode analysis figures for report generation"""
    figures = {}
    
    # Analyze if needed
    for obj in loaded_objects:
        eng = obj['engine']
        if not hasattr(eng, 'episodes_left') or not eng.episodes_left:
            try:
                eng.analyze_episodes()
            except Exception:
                pass

    figures['Episode Timeline'] = get_episode_timeline_figure(loaded_objects)
    figures['Interaction Patterns'] = get_interaction_patterns_figure(loaded_objects)
    figures['Episode Metrics'] = get_episode_metrics_figure(loaded_objects)
    
    return figures

def get_episode_timeline_figure(loaded_objects):
    """Generate episode timeline figure"""
    episodes_by_session = {}
    for obj in loaded_objects:
        eng = obj['engine']
        name = obj['name'][:15]
        episodes_by_session[name] = {
            'left': eng.episodes_left if hasattr(eng, 'episodes_left') else [],
            'right': eng.episodes_right if hasattr(eng, 'episodes_right') else []
        }
    return create_episode_timeline(episodes_by_session)

def plot_episode_timeline_tab(parent, loaded_objects, window):
    """Create and embed episode timeline visualization"""
    fig = get_episode_timeline_figure(loaded_objects)
    
    if hasattr(window, '_embed_plotly_figure'):
        window._embed_plotly_figure(fig, parent)
    else:
        _embed_plotly_figure_fallback(fig, parent)

def get_interaction_patterns_figure(loaded_objects):
    """Generate interaction patterns figure"""
    all_episodes = []
    session_labels = []
    analyzer = None
    
    for obj in loaded_objects:
        eng = obj['engine']
        name = obj['name'][:15]
        
        if hasattr(eng, 'episode_analyzer'):
            analyzer = eng.episode_analyzer
        
        if hasattr(eng, 'episodes_left'):
            all_episodes.extend(eng.episodes_left)
            session_labels.extend([f"{name} (L)"] * len(eng.episodes_left))
        
        if hasattr(eng, 'episodes_right'):
            all_episodes.extend(eng.episodes_right)
            session_labels.extend([f"{name} (R)"] * len(eng.episodes_right))
            
    if not all_episodes or not analyzer:
        return None
        
    return create_episode_clustering_view(all_episodes, session_labels, analyzer)

def plot_interaction_patterns_tab(parent, loaded_objects, window):
    """Create and embed UMAP clustering visualization"""
    fig = get_interaction_patterns_figure(loaded_objects)
    
    if fig is None:
        ctk.CTkLabel(parent, text="No episodes available for pattern analysis",
                    font=("Arial", 14)).pack(pady=50)
        return

    if hasattr(window, '_embed_plotly_figure'):
        window._embed_plotly_figure(fig, parent)
    else:
        _embed_plotly_figure_fallback(fig, parent)

def get_episode_metrics_figure(loaded_objects):
    """Generate episode metrics figure"""
    metrics_by_session = {}
    
    for obj in loaded_objects:
        eng = obj['engine']
        name = obj['name'][:15]
        
        if hasattr(eng, 'get_episode_metrics'):
            metrics_by_session[name] = eng.get_episode_metrics()
        else:
            # Fallback
            metrics_by_session[name] = {
                'left': {'count': 0, 'avg_duration': 0, 'avg_confidence': 0, 'types': {}},
                'right': {'count': 0, 'avg_duration': 0, 'avg_confidence': 0, 'types': {}}
            }
            if hasattr(eng, 'episodes_left') and eng.episodes_left:
                eps = eng.episodes_left
                metrics_by_session[name]['left'] = {
                    'count': len(eps),
                    'avg_duration': np.mean([e.duration for e in eps]),
                    'avg_confidence': np.mean([e.confidence for e in eps]),
                    'types': {}
                }
            if hasattr(eng, 'episodes_right') and eng.episodes_right:
                eps = eng.episodes_right
                metrics_by_session[name]['right'] = {
                    'count': len(eps),
                    'avg_duration': np.mean([e.duration for e in eps]),
                    'avg_confidence': np.mean([e.confidence for e in eps]),
                    'types': {}
                }
                
    return create_interaction_metrics_comparison(metrics_by_session)

def plot_episode_metrics_tab(parent, loaded_objects, window):
    """Create and embed episode metrics comparison"""
    fig = get_episode_metrics_figure(loaded_objects)
    
    if hasattr(window, '_embed_plotly_figure'):
        window._embed_plotly_figure(fig, parent)
    else:
        _embed_plotly_figure_fallback(fig, parent)


def _embed_plotly_figure_fallback(fig, parent):
    """Fallback method to display Plotly figure in browser"""
    import tempfile
    import webbrowser
    
    html_str = fig.to_html(include_plotlyjs='cdn', config={
        'responsive': True,
        'displayModeBar': True,
        'displaylogo': False
    })
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
        f.write(html_str)
        temp_path = f.name
    
    webbrowser.open('file:///' + temp_path.replace('\\', '/'))
    
    # Show confirmation UI
    frame = ctk.CTkFrame(parent)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    ctk.CTkLabel(frame, text="✓ Interactive Chart Opened in Browser", 
                font=("Arial", 18, "bold"), text_color="#00aa00").pack(pady=(40, 20))
    
    ctk.CTkButton(frame, text="🔄 Reopen Chart", 
                 command=lambda: webbrowser.open('file:///' + temp_path.replace('\\', '/')),
                 height=45, font=("Arial", 13, "bold")).pack(pady=20)


# Standalone function to patch existing comparison window
def patch_comparison_window(comparison_method):
    """
    Decorator to enhance existing compare_selected method
    
    Usage:
        @patch_comparison_window
        def compare_selected(self, lib_window=None):
            # existing code...
    """
    def wrapper(self, *args, **kwargs):
        # Call original method
        result = comparison_method(self, *args, **kwargs)
        
        # Find the comparison window and tab_view
        # This assumes the window was just created
        # You may need to store references or modify the original method
        
        return result
    
    return wrapper
