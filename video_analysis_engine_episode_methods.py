    def get_sentiment_stats(self):
        """Returns sentiment distribution dictionary."""
        stats = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Total': 0}
        
        for entry in self.transcript:
            # entry: (start, end, text, category_str)
            # category_str example: "joy (0.9)" or "negative (0.8)"
            cat_str = entry[3].lower() if len(entry) > 3 else "neutral"
            
            # Simple keyword matching for common sentiment labels
            if "joy" in cat_str or "positive" in cat_str:
                stats['Positive'] += 1
            elif "sadness" in cat_str or "anger" in cat_str or "negative" in cat_str:
                stats['Negative'] += 1
            else:
                stats['Neutral'] += 1
            stats['Total'] += 1
            
        return stats
    
    # --- Episode Analysis Methods ---
    
    def analyze_episodes(self):
        """Analyze interaction episodes for both hands"""
        # Generate timestamps array
        timestamps = np.array([i / self.fps for i in range(max(len(self.trace_path_left), len(self.trace_path_right)))])
        
        # Prepare transcript for alignment
        transcript_tuples = [(t[0], t[1], t[2]) for t in self.transcript] if self.transcript else None
        
        # Analyze left hand
        if self.trace_path_left and len(self.trace_path_left) > 0:
            try:
                self.episodes_left = self.episode_analyzer.analyze_trace(
                    self.trace_path_left,
                    timestamps[:len(self.trace_path_left)],
                    hand='left',
                    transcript=transcript_tuples
                )
            except Exception as e:
                print(f"Left hand episode analysis failed: {e}")
                self.episodes_left = []
        
        # Analyze right hand
        if self.trace_path_right and len(self.trace_path_right) > 0:
            try:
                self.episodes_right = self.episode_analyzer.analyze_trace(
                    self.trace_path_right,
                    timestamps[:len(self.trace_path_right)],
                    hand='right',
                    transcript=transcript_tuples
                )
            except Exception as e:
                print(f"Right hand episode analysis failed: {e}")
                self.episodes_right = []
    
    def get_episode_metrics(self):
        """Calculate aggregate episode-level metrics"""
        metrics = {
            'left': {'count': 0, 'avg_duration': 0, 'avg_confidence': 0, 'types': {}},
            'right': {'count': 0, 'avg_duration': 0, 'avg_confidence': 0, 'types': {}}
        }
        
        for side, episodes in [('left', self.episodes_left), ('right', self.episodes_right)]:
            if not episodes:
                continue
                
            metrics[side]['count'] = len(episodes)
            metrics[side]['avg_duration'] = np.mean([ep.duration for ep in episodes])
            metrics[side]['avg_confidence'] = np.mean([ep.confidence for ep in episodes])
            
            # Count episode types
            type_counts = collections.Counter([ep.episode_type for ep in episodes])
            metrics[side]['types'] = dict(type_counts)
        
        return metrics
    
    def get_episode_at(self, time_seconds, hand='left'):
        """Get episode information at specific timestamp"""
        episodes = self.episodes_left if hand == 'left' else self.episodes_right
        
        for ep in episodes:
            if ep.start_time <= time_seconds <= ep.end_time:
                return ep
        return None
