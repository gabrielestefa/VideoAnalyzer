"""
Episode Visualization Components for Hand Tracking Analysis

Provides reusable Plotly-based visualizations for interaction episodes.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict
from episode_analyzer import Episode


def create_episode_timeline(episodes_by_session: Dict[str, Dict[str, List[Episode]]], 
                            fps: float = 30.0) -> go.Figure:
    """
    Create Gantt chart showing episodes across multiple sessions
    
    Args:
        episodes_by_session: Dict of {session_name: {'left': [...], 'right': [...]}}
        fps: Frames per second for display
        
    Returns:
        Plotly Figure with episode timeline
    """
    fig = go.Figure()
    
    # Color mapping for episode types
    type_colors = {
        'rest': '#d3d3d3',  # Light gray
        'reach': '#77dd77',  # Green
        'direct_action': '#6a5acd',  # Blue
        'exploratory': '#ffb347',  # Orange
        'manipulation': '#ff6961',  # Red
        'unknown': '#cccccc'  # Gray
    }
    
    y_position = 0
    y_labels = []
    
    for session_name, hands_data in episodes_by_session.items():
        for hand in ['left', 'right']:
            episodes = hands_data.get(hand, [])
            if not episodes:
                continue
                
            label = f"{session_name[:15]} - {hand.capitalize()}"
            y_labels.append(label)
            
            for ep in episodes:
                # Create bar for this episode
                fig.add_trace(go.Bar(
                    x=[ep.duration],
                    y=[y_position],
                    base=ep.start_time,
                    orientation='h',
                    marker=dict(
                        color=type_colors.get(ep.episode_type, '#cccccc'),
                        line=dict(width=1, color='white')
                    ),
                    name=ep.episode_type,
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{ep.episode_type.capitalize()}</b><br>'
                        f'Time: {ep.start_time:.2f}s - {ep.end_time:.2f}s<br>'
                        f'Duration: {ep.duration:.2f}s<br>'
                        f'Confidence: {ep.confidence:.2f}<br>'
                        f'Avg Velocity: {ep.kinematic_features.get("mean_velocity", 0):.3f}<br>'
                        f'Path Efficiency: {ep.kinematic_features.get("path_efficiency", 0):.2f}<br>'
                        f'Hesitation: {ep.intent_features.get("hesitation_index", 0):.3f}<br>'
                        '<extra></extra>'
                    )
                ))
            
            y_position += 1
    
    # Add legend manually
    for ep_type, color in type_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=ep_type.replace('_', ' ').title(),
            showlegend=True
        ))
    
    fig.update_layout(
        title="Interaction Episode Timeline<br><sup>Color code indicates action type; Hover for details</sup>",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels
        ),
        barmode='overlay',
        height=max(400, len(y_labels) * 40),
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def create_episode_clustering_view(all_episodes: List[Episode], 
                                   session_labels: List[str],
                                   analyzer) -> go.Figure:
    """
    Create UMAP projection of episodes colored by session
    
    Args:
        all_episodes: List of all Episode objects from all sessions
        session_labels: List of session names (same length as all_episodes)
        analyzer: EpisodeAnalyzer instance for embedding computation
        
    Returns:
        Plotly Figure with 2D scatter plot
    """
    if len(all_episodes) < 5:
        # Not enough data for UMAP
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough episodes for clustering visualization (minimum 5 required)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Compute embeddings
    embeddings = []
    for ep in all_episodes:
        emb = analyzer.compute_episode_embedding(ep, embed_dim=32)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    # Try UMAP, fall back to PCA if not available
    try:
        from umap import UMAP
        reducer = UMAP(n_neighbors=min(15, len(all_episodes) - 1), n_components=2, random_state=42)
        projection = reducer.fit_transform(embeddings)
    except ImportError:
        try:
            # Fallback to PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            projection = reducer.fit_transform(embeddings)
        except ImportError as e:
            # No sklearn available
            print(f"Sklearn import failed: {e}")
            fig = go.Figure()
            fig.update_layout(
                title="Clustering Unavailable (scikit-learn not installed)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(
                    text="Please install scikit-learn to view episode clustering:<br>pip install scikit-learn",
                    xref="paper", yref="paper",
                    showarrow=False, font=dict(size=14)
                )]
            )
            return fig
    
    # Create scatter plot
    fig = go.Figure()
    
    # Group by session for coloring
    unique_sessions = list(set(session_labels))
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    
    for i, session in enumerate(unique_sessions):
        mask = np.array([s == session for s in session_labels])
        session_episodes = [ep for ep, m in zip(all_episodes, mask) if m]
        
        # Extract info for hover
        hover_texts = []
        for ep in session_episodes:
            text = (
                f"Type: {ep.episode_type}<br>"
                f"Duration: {ep.duration:.2f}s<br>"
                f"Confidence: {ep.confidence:.2f}<br>"
                f"Velocity: {ep.kinematic_features.get('mean_velocity', 0):.3f}"
            )
            hover_texts.append(text)
        
        fig.add_trace(go.Scatter(
            x=projection[mask, 0],
            y=projection[mask, 1],
            mode='markers',
            name=session[:15],
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=hover_texts,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    fig.update_layout(
        title="Interaction Style Clustering (UMAP Projection)",
        xaxis_title="Dimension 1 (Kinematic Similarity)",
        yaxis_title="Dimension 2 (Kinematic Similarity)",
        height=600,
        hovermode='closest',
        template='plotly_white',
        legend=dict(title="Session"),
        annotations=[dict(
            text="Proximity indicates kinematic similarity",
            xref="paper", yref="paper",
            x=0, y=1, showarrow=False,
            font=dict(size=10, color="gray")
        )]
    )
    
    return fig


def create_interaction_metrics_comparison(episode_metrics_by_session: Dict[str, Dict]) -> go.Figure:
    """
    Create comparison of episode-level metrics across sessions
    
    Args:
        episode_metrics_by_session: Dict of {session_name: metrics_dict}
        
    Returns:
        Plotly Figure with metric comparisons
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Interaction Frequency', 'Mean Episode Duration (s)', 
                       'Mean Intent Confidence Score (0-1)', 'Episode Type Distribution'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    session_names = list(episode_metrics_by_session.keys())
    short_names = [name[:10] for name in session_names]
    
    # Episode counts
    left_counts = [episode_metrics_by_session[s]['left']['count'] for s in session_names]
    right_counts = [episode_metrics_by_session[s]['right']['count'] for s in session_names]
    
    fig.add_trace(go.Bar(
        x=short_names, y=left_counts,
        name='Left', marker_color='#636EFA'
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=short_names, y=right_counts,
        name='Right', marker_color='#EF553B'
    ), row=1, col=1)
    
    # Average duration
    left_dur = [episode_metrics_by_session[s]['left']['avg_duration'] for s in session_names]
    right_dur = [episode_metrics_by_session[s]['right']['avg_duration'] for s in session_names]
    
    fig.add_trace(go.Bar(
        x=short_names, y=left_dur,
        name='Left', marker_color='#636EFA', showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=short_names, y=right_dur,
        name='Right', marker_color='#EF553B', showlegend=False
    ), row=1, col=2)
    
    # Average confidence
    left_conf = [episode_metrics_by_session[s]['left']['avg_confidence'] for s in session_names]
    right_conf = [episode_metrics_by_session[s]['right']['avg_confidence'] for s in session_names]
    
    fig.add_trace(go.Bar(
        x=short_names, y=left_conf,
        name='Left', marker_color='#636EFA', showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=short_names, y=right_conf,
        name='Right', marker_color='#EF553B', showlegend=False
    ), row=2, col=2)
    
    # Episode type distribution (aggregate all sessions)
    all_types = {}
    for metrics in episode_metrics_by_session.values():
        for hand in ['left', 'right']:
            for ep_type, count in metrics[hand].get('types', {}).items():
                all_types[ep_type] = all_types.get(ep_type, 0) + count
    
    if all_types:
        fig.add_trace(go.Bar(
            x=list(all_types.keys()),
            y=list(all_types.values()),
            marker_color='#00CC96',
            showlegend=False
        ), row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        barmode='group',
        template='plotly_white'
    )
    
    return fig


def create_speech_motion_alignment_view(episodes: List[Episode]) -> go.Figure:
    """
    Visualize speech-motion alignment for episodes
    
    Args:
        episodes: List of Episode objects with aligned speech
        
    Returns:
        Plotly Figure showing alignment
    """
    fig = go.Figure()
    
    episodes_with_speech = [ep for ep in episodes if ep.aligned_speech]
    
    if not episodes_with_speech:
        fig.add_annotation(
            text="No episodes with aligned speech found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Create timeline view
    for i, ep in enumerate(episodes_with_speech[:20]):  # Limit to first 20 for clarity
        # Episode bar
        fig.add_trace(go.Bar(
            x=[ep.duration],
            y=[i * 2],
            base=ep.start_time,
            orientation='h',
            marker=dict(color='#636EFA', opacity=0.5),
            name='Motion',
            showlegend=(i == 0),
            hovertemplate=f'Episode: {ep.start_time:.1f}s - {ep.end_time:.1f}s<extra></extra>'
        ))
        
        # Speech bars
        for start, end, text in ep.aligned_speech:
            fig.add_trace(go.Bar(
                x=[end - start],
                y=[i * 2 + 1],
                base=start,
                orientation='h',
                marker=dict(color='#00CC96', opacity=0.7),
                name='Speech',
                showlegend=(i == 0),
                hovertemplate=f'Speech: "{text}"<br>{start:.1f}s - {end:.1f}s<extra></extra>'
            ))
    
    fig.update_layout(
        title="Speech-Motion Alignment",
        xaxis_title="Time (seconds)",
        yaxis=dict(visible=False),
        height=max(400, len(episodes_with_speech[:20]) * 50),
        barmode='overlay',
        template='plotly_white'
    )
    
    return fig
