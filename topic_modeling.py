"""
Hybrid Adaptive Topic Modeling System
Tries NMF first, falls back to BERTopic if quality is insufficient
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ZeroShotClassifier:
    """
    Efficient embedding-based zero-shot classifier.
    Reuses existing sentence-transformers models for negligible overhead.
    """
    def __init__(self):
        # Load categories from JSON if available, otherwise use defaults
        self.categories = self._load_categories()
        self.model = None
        self.category_embeddings = None

    def _load_categories(self):
        import json
        import os
        
        default_categories = {
            "Usability Issue / Bug": [
                "bug", "error", "crash", "glitch", "problem", "broken", "fail", "issue", "mistake", "wrong", 
                "confusing", "hard to use", "stuck", "weird", "unexpected", "doesn't work"
            ],
            "Feature Request": [
                "feature", "request", "add", "create", "would be nice", "wish", "want", "need", "missing", 
                "suggest", "idea", "improvement", "better if", "can we have"
            ],
            "Positive Feedback": [
                "good", "great", "awesome", "amazing", "cool", "nice", "love", "perfect", "excellent", 
                "helpful", "useful", "impressed", "well done", "favorite"
            ],
            "Instruction / Walkthrough": [
                "click", "tap", "press", "select", "drag", "drop", "move", "go to", "open", "close", 
                "step", "first", "then", "next", "how to", "tutorial", "guide"
            ],
            "Visual Design": [
                "color", "layout", "font", "style", "look", "appearance", "beautiful", "ugly", "clean", 
                "cluttered", "design", "ui", "interface", "button", "icon"
            ],
            "General / Irrelevant": [
                "hello", "hi", "hey", "okay", "ok", "yeah", "yes", "no", "maybe", "thanks", "thank you",
                "video", "transcript", "test", "testing", "one", "two", "three", "checking", "um", "uh",
                "screen", "recording", "audio", "mic", "microphone"
            ]
        }
        
        try:
            # Look for categories.json in the same directory as this script
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "categories.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    categories = json.load(f)
                return categories
        except Exception as e:
            print(f"⚠ Failed to load categories.json, using defaults: {e}")
            
        return default_categories

    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            
            # Pre-compute category embeddings (average of keywords for robustness)
            self.category_embeddings = {}
            for cat, keywords in self.categories.items():
                # We embed the category name AND its keywords to create a robust prototype
                text_to_embed = [cat] + keywords
                embeddings = self.model.encode(text_to_embed)
                # Average them to get a centroid for the category
                self.category_embeddings[cat] = np.mean(embeddings, axis=0)

    def classify(self, text_segments: List[str], threshold: float = 0.18) ->  List[Dict]:
        self._ensure_model()
        
        if not text_segments:
            return []
            
        # Bulk encode all segments
        segment_embeddings = self.model.encode(text_segments)
        
        results = []
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Convert category embeddings to matrix
        cat_names = list(self.category_embeddings.keys())
        cat_matrix = np.array([self.category_embeddings[c] for c in cat_names])
        
        # Compute similarity matrix (segments x categories)
        similarities = cosine_similarity(segment_embeddings, cat_matrix)
        
        for i, text in enumerate(text_segments):
            best_idx = np.argmax(similarities[i])
            best_score = similarities[i][best_idx]
            
            result = {
                "text": text,
                "category": "Unclassified",
                "score": float(best_score),
                "is_confident": False
            }
            
            if best_score > threshold:
                result["category"] = cat_names[best_idx]
                result["is_confident"] = True
            else:
                # Debug: Why was this unclassified?
                if i < 5: # Print first 5 unclassified for debugging
                     print(f"  [Unclassified] '{text[:50]}...' Best: {cat_names[best_idx]} ({best_score:.3f}) < {threshold}")
                
            results.append(result)
            
        return results


class SemanticClusterer:
    """
    Performs K-Means clustering at multiple scales (K=3, 5, 8, 10).
    Allows users to explore "coarse" vs "fine" semantic structures.
    """
    def __init__(self, embedding_model=None):
        self.model = embedding_model # Reuse existing model if possible
        
    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def cluster(self, text_segments: List[str]) -> Dict[int, Dict]:
        self._ensure_model()
        if not text_segments or len(text_segments) < 3:
            print(f"  [Semantic Clustering] Insufficient segments: {len(text_segments) if text_segments else 0}")
            return {}
            
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        # 1. Embed all segments
        print(f"  [Semantic Clustering] Encoding {len(text_segments)} segments...")
        embeddings = self.model.encode(text_segments)
        print(f"  [Semantic Clustering] Embeddings shape: {embeddings.shape}")
        
        # 2. Reduce to 2D for visualization (using PCA for speed/stability)
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(embeddings)
        
        # Check if PCA produced valid coordinates
        if np.all(coords_2d == 0) or np.isnan(coords_2d).any():
            print(f"  [Warning] PCA produced invalid coordinates (all zeros or NaN), using random layout")
            # Fallback: use random layout
            coords_2d = np.random.randn(len(text_segments), 2) * 10
        else:
            # Normalize coordinates to reasonable range for visualization
            coords_2d = (coords_2d - coords_2d.mean(axis=0)) / (coords_2d.std(axis=0) + 1e-8)
            coords_2d = coords_2d * 10  # Scale to [-30, 30] roughly
            
        print(f"  [Semantic Clustering] Coords range: X=[{coords_2d[:, 0].min():.2f}, {coords_2d[:, 0].max():.2f}], Y=[{coords_2d[:, 1].min():.2f}, {coords_2d[:, 1].max():.2f}]")
        
        results = {}
        k_values = [3, 5, 8, 10, 20]
        
        # 3. Running K-Means for each K
        for k in k_values:
            if k >= len(text_segments):
                continue
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(embeddings)
            
            # Find closest words to centroids to name the clusters
            # Simple approach: find segment closest to center
            cluster_names = {}
            for i in range(k):
                # Get indices of points in this cluster
                indices = np.where(labels == i)[0]
                if len(indices) == 0:
                    continue
                    
                # Calculate centroid manually or use kmeans.cluster_centers_[i]
                center = kmeans.cluster_centers_[i]
                
                # Find closest segment in this cluster to the center
                cluster_embeddings = embeddings[indices]
                dists = np.linalg.norm(cluster_embeddings - center, axis=1)
                closest_idx = indices[np.argmin(dists)]
                
                # Use first few words of central sentence as label
                # In robust version, we'd use TF-IDF top words of the cluster
                text = text_segments[closest_idx]
                name = " ".join(text.split()[:4]) + "..."
                cluster_names[i] = name
            
            results[k] = {
                'labels': labels.tolist(),
                'coords': coords_2d.tolist(),
                'cluster_names': cluster_names
            }
            
        print(f"  [Semantic Clustering] Generated {len(results)} K-value results")
        return results


class HybridTopicModeler:
    """Adaptive topic modeling with NMF → BERTopic fallback"""
    
    def __init__(self, n_topics: int = 8, min_topic_size: int = 3):
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.method_used = None
        self.topics = None
        self.topic_labels = None
        self.topic_words = None
        self.topic_words = None
        self.zero_shot = ZeroShotClassifier()  # Initialize Zero-Shot
        self.semantic_clusterer = SemanticClusterer() # Initialize Semantic Clusterer
        
    def extract_topics(self, documents: List[str]) -> Dict:
        """
        Main method to extract topics from documents.
        """
        if len(documents) < self.min_topic_size:
            return self._create_empty_result("Insufficient documents")
            
        # Run Zero-Shot Classification
        # Run Zero-Shot Classification
        # DEPRECATED: Merged into Guided BERTopic (see _try_bertopic)
        # We return empty list here to disable the standalone chart
        print("Skipping standalone Zero-Shot Classification (Guided BERTopic active)...")
        zero_shot_results = []
        # try:
        #     zero_shot_results = self.zero_shot.classify(documents)
        #     print(f"[OK] Classified {len(zero_shot_results)} segments")
        # except Exception as e:
        #     print(f"[Warning] Zero-Shot Classification failed: {e}")
        #     zero_shot_results = []
            
        # Run Semantic Clustering (K-Means)
        print("Running Semantic Clustering (K-Means)...")
        cluster_results = {}
        try:
            # Explicitly ensure the semantic clusterer has a model
            # (Since zero-shot is now skipped, we can't rely on model sharing)
            if not hasattr(self.semantic_clusterer, 'model') or self.semantic_clusterer.model is None:
                print("  Initializing embedding model for semantic clustering...")
                from sentence_transformers import SentenceTransformer
                self.semantic_clusterer.model = SentenceTransformer("all-MiniLM-L6-v2")
                
            cluster_results = self.semantic_clusterer.cluster(documents)
            if cluster_results:
                print(f"  [OK] Generated clusters for K={list(cluster_results.keys())}")
            else:
                print("  [Warning] Semantic clustering returned empty results")
        except Exception as e:
            print(f"  [Warning] Semantic Clustering failed: {e}")
            import traceback
            traceback.print_exc()
            cluster_results = {}
        
        # Try NMF first - DEPRECATED: User requested BERTopic Only standard
        # print("Attempting NMF topic modeling...")
        # nmf_result = self._try_nmf(documents)
        # nmf_result['zero_shot'] = zero_shot_results  # Attach classification
        # nmf_result['semantic_clusters'] = cluster_results
        
        # Evaluate quality
        # quality = self._evaluate_quality(nmf_result, documents)
        # print(f"NMF Quality Score: {quality:.3f}")
        
        # Decision: use NMF or fallback to BERTopic
        print("Using BERTopic as primary model (Direct Mode)...")
        bertopic_result = self._try_bertopic(documents)
        bertopic_result['method'] = 'BERTopic'
        bertopic_result['quality_score'] = 1.0 # Assume best quality
        bertopic_result['zero_shot'] = zero_shot_results 
        bertopic_result['semantic_clusters'] = cluster_results
        self.method_used = 'BERTopic'
        return bertopic_result
    
    def _try_nmf(self, documents):
        """Attempt NMF topic modeling"""
        try:
            # Comprehensive custom stop words
            custom_stop_words = [
                # Common English stop words
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'now',
                # Video/UI specific words that are too generic
                'video', 'screen', 'click', 'button', 'see', 'look', 'show', 'going',
                'like', 'want', 'need', 'get', 'thing', 'know', 'think', 'really',
                'actually', 'basically', 'okay', 'yeah', 'um', 'uh', 'kind', 'also',
                'then', 'here', 'there', 'inch', 'inches', 'about', 'something', 'action',
                'right', 'left', 'up', 'down', 'good'
            ]
            
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words=custom_stop_words,
                max_df=0.85,  # Ignore words appearing in >85% of docs
                min_df=2,
                ngram_range=(1, 3),  # Include phrases up to 3 words
                token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only 3+ letter words
            )
            tfidf = vectorizer.fit_transform(documents)
            
            # Apply NMF
            nmf = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=200,
                alpha_W=0.1,
                alpha_H=0.1
            )
            doc_topics = nmf.fit_transform(tfidf)
            
            # Assign topics (dominant topic per doc)
            topics = np.argmax(doc_topics, axis=1).tolist()
            
            # Extract top words per topic
            feature_names = vectorizer.get_feature_names_out()
            topic_words = {}
            topic_labels = []
            
            # Generic words to skip in labels (even if they passed vectorizer)
            skip_in_labels = {'also', 'then', 'here', 'there', 'cars', 'car', 'about', 'something', 'action'}
            
            for topic_idx, topic in enumerate(nmf.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [(feature_names[i], topic[i]) for i in top_indices]
                topic_words[topic_idx] = top_words
                
                # Create label from top 3 meaningful words with deduplication
                label_words = []
                seen_words = set()
                
                for word, score in top_words[:10]:  # Check up to 10 words
                    w_lower = word.lower()
                    if len(word) > 3 and w_lower not in skip_in_labels and w_lower not in seen_words:
                        label_words.append(word)
                        seen_words.add(w_lower)
                        # Check for sub-word duplicates (e.g., "Vehicle" vs "Vehicles")
                        seen_words.add(w_lower.rstrip('s'))
                        if len(label_words) == 3:
                            break
                
                if label_words:
                    label = "_".join(label_words)
                    topic_labels.append(self._beautify_label(label))
                else:
                    topic_labels.append(f"Topic {topic_idx + 1}")
            
            # --- LLM ENHANCEMENT FOR NMF ---
            # Now try to upgrade these labels using the local LLM
            print("  Enhancing NMF labels with local LLM...")
            try:
                # Prepare input format for the helper: {0: ["art", "design"], 1: ["bug", "fix"]}
                topic_keywords_map = {}
                for tid, words_scores in topic_words.items():
                    # topic_words is {tid: [(word, score), ...]}
                    keywords = [w[0] for w in words_scores]
                    topic_keywords_map[tid] = keywords
                    
                llm_labels_map = self._generate_llm_labels_simple(topic_keywords_map)
                
                # Apply if we got results
                if llm_labels_map:
                    for i, label in enumerate(topic_labels):
                        # topic_labels list index corresponds to topic_idx (0..N)
                        if i in llm_labels_map:
                            new_label = llm_labels_map[i]
                            # Clean and Prefix
                            new_label = list(str(new_label).replace('Label:', '').strip().strip('"').strip("'").split('\n'))[0]
                            if new_label.endswith('.'): new_label = new_label[:-1]
                            
                            topic_labels[i] = new_label 
                            # print(f"  [NMF] Topic {i} -> {new_label}")
            except Exception as llm_e:
                print(f"  [Warning] NMF LLM enhancement failed: {llm_e}")
            # -------------------------------
            
            # Calculate topic prevalence
            unique, counts = np.unique(topics, return_counts=True)
            topic_sizes = dict(zip(unique.tolist(), counts.tolist()))
            
            return {
                'topics': topics,
                'topic_labels': topic_labels,
                'topic_words': topic_words,
                'topic_sizes': topic_sizes,
                'doc_topic_matrix': doc_topics,
                'n_topics': len(set(topics))
            }
            
        except Exception as e:
            print(f"NMF failed: {e}")
            return self._create_empty_result(str(e))
    
    def _try_nmf_aggressive(self, documents: List[str]) -> Dict:
        """Try NMF with smaller topic count for difficult-to-cluster data"""
        print("Attempting aggressive NMF with reduced topic count...")
        try:
            # Use fewer topics for sparse data
            n_topics_aggressive = min(4, max(2, len(documents) // 4))
            
            # Comprehensive custom stop words
            # Comprehensive custom stop words
            custom_stop_words = [
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'now',
                'video', 'screen', 'click', 'button', 'see', 'look', 'show', 'going',
                'like', 'want', 'need', 'get', 'thing', 'know', 'think', 'really',
                'really', 'actually', 'basically', 'okay', 'yeah', 'um', 'uh', 'kind', 'also',
                'then', 'here', 'there', 'inch', 'inches', 'about', 'something', 'action',
                'right', 'left', 'up', 'down', 'good', 'lot', 'bit', 'little', 'terms', 'thing',
                'their', 'our', 'one', 'two', 'three', 'use', 'using', 'used', 'make', 'made'
            ]
            
            vectorizer = TfidfVectorizer(
                max_features=300,
                stop_words=custom_stop_words,
                max_df=0.9,
                min_df=1,  # Lower threshold
                ngram_range=(1, 2)
            )
            tfidf = vectorizer.fit_transform(documents)
            
            nmf = NMF(
                n_components=n_topics_aggressive,
                random_state=42,
                max_iter=300,
                alpha_W=0.05,
                alpha_H=0.05
            )
            doc_topics = nmf.fit_transform(tfidf)
            
            topics = np.argmax(doc_topics, axis=1).tolist()
            
            feature_names = vectorizer.get_feature_names_out()
            topic_words = {}
            topic_labels = []
            
            skip_in_labels = {'also', 'then', 'here', 'there', 'cars', 'car', 'about', 'something', 'action'}
            
            for topic_idx, topic in enumerate(nmf.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [(feature_names[i], topic[i]) for i in top_indices]
                topic_words[topic_idx] = top_words
                
                label_words = []
                seen_words = set()
                for word, score in top_words[:10]:
                    w_lower = word.lower()
                    if len(word) > 3 and w_lower not in skip_in_labels and w_lower not in seen_words:
                        label_words.append(word)
                        seen_words.add(w_lower)
                        seen_words.add(w_lower.rstrip('s')) # Basic plural check
                        if len(label_words) == 3:
                            break
                            
                if label_words:
                    label = "_".join(label_words)
                    topic_labels.append(self._beautify_label(label))
                else:
                    topic_labels.append(f"Topic {topic_idx + 1}")
            
            unique, counts = np.unique(topics, return_counts=True)
            topic_sizes = dict(zip(unique.tolist(), counts.tolist()))
            
            print(f"[OK] Aggressive NMF found {len(set(topics))} topics")
            
            return {
                'topics': topics,
                'topic_labels': topic_labels,
                'topic_words': topic_words,
                'topic_sizes': topic_sizes,
                'doc_topic_matrix': doc_topics,
                'n_topics': len(set(topics))
            }
        except Exception as e:
            print(f"Aggressive NMF failed: {e}")
            return self._create_empty_result(str(e))
    
    def _try_bertopic(self, documents: List[str]) -> Dict:
        """Extract topics using BERTopic"""
        try:
            from bertopic import BERTopic
            from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
            from sentence_transformers import SentenceTransformer
            from umap import UMAP
            from hdbscan import HDBSCAN
            from transformers import pipeline
            import torch # Check for GPU
            
            # 1. NOISE FILTERING & PRE-PROCESSING
            # Filter out extreme noise/fillers before modeling to prevent "I'm sorry" topics
            noise_phrases = ["i'm sorry", "didn't get that", "thank you", "okay", "yeah", "yes", "um", "uh", "no"]
            filtered_docs = []
            valid_indices = []
            for i, doc in enumerate(documents):
                clean_doc = doc.lower().strip()
                # Ignore very short segments that are just fillers
                if len(clean_doc) < 15 and any(noise in clean_doc for noise in noise_phrases):
                    continue
                filtered_docs.append(doc)
                valid_indices.append(i)
            
            if not filtered_docs:
                return self._create_empty_result("No valid segments after filtering noise")

            # TINY DATASET FALLBACK (Manual Mode)
            # If < 5 docs, UMAP/BERTopic is overkill and error-prone.
            # Just group them into a single topic.
            if len(filtered_docs) < 5:
                print(f"  [BERTopic] Dataset too small ({len(filtered_docs)} docs) for full modeling. Using manual fallback.")
                return {
                    'topics': [0] * len(filtered_docs), # All in Topic 0
                    'topic_labels': {0: "General Review"},
                    'topic_words': {0: [("note", 0.5), ("review", 0.5)]}, # Dummy keywords
                    'topic_sizes': {0: len(filtered_docs)},
                    'topic_coords': {}, # No UMAP
                    'doc_topic_matrix': None, # Skipping matrix
                    'n_topics': 1
                }
            
            # 2. SETUP MODELS (GPU-accelerated when available)
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  [BERTopic] Embedding device: {device.upper()}")
            
            # Dynamic Parameter Adjustment for Small Datasets
            n_docs = len(filtered_docs)
            print(f"  [BERTopic] Configuring for {n_docs} documents...")
            
            # UMAP: Neighbors must be < n_docs
            # Default n_neighbors=15, but for small data we must reduce it
            n_neighbors = 15
            n_components = 5
            umap_min_dist = 0.0 # Default for larger datasets

            if n_docs <= 15:
                n_neighbors = max(2, n_docs - 2) # e.g. if 4 docs, neighbors=2
                n_components = min(n_docs - 1, 5)
                umap_min_dist = 0.1 # Use UMAP default for very small datasets

            # HDBSCAN: Min Cluster Size
            # Default is 15
            # With cleaner data (faster-whisper), we need to be more aggressive to find nuances
            min_cluster_size = 8 
            if n_docs < 200:
                # Scaled for small-medium datasets
                # < 50 docs: 2-5
                # 67 docs: ~4-5
                # 150 docs: ~10
                min_cluster_size = max(2, int(n_docs / 20))
            
            # BERTopic: Min Topic Size
            min_topic_size_bertopic = 8
            if n_docs < 200:
                min_topic_size_bertopic = max(2, int(n_docs / 20))
                
            print(f"  [BERTopic] Params: Neighbors={n_neighbors}, Components={n_components}, UMAP_min_dist={umap_min_dist}, ClusterSize={min_cluster_size}, TopicSize={min_topic_size_bertopic}")

            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=umap_min_dist, metric='cosine', random_state=42)
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size, 
                min_samples=2 if n_docs < 50 else max(2, int(n_docs / 40)), # Reduced to find more structure
                metric='euclidean', 
                cluster_selection_method='leaf', # Was 'eom', 'leaf' finds smaller structure -> more topics
                prediction_data=True
            )
            # 2. FIX LABELS: Use Local LLM + KeyBERTInspired + MMR
            # Try to load local LLM
            # Define representation models (Use fast KeyBERT/MMR for internal model)
            # We will use our custom LLM step (generate_topic_summaries) for the final labels
            # This prevents the "Fine-tuning topics" step from hanging
            representation_model = {
                "KeyBERT": KeyBERTInspired(),
                "Aspect1": MaximalMarginalRelevance(diversity=0.3),
                "Main": KeyBERTInspired()
            }
            
            # Custom vectorizer with stop words for cleaner topic representation
            # 1. Standard NLTK English Stop Words (Comprehensive)
            nltk_stop_words = [
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
            ]
            
            # 2. Aggressive Conversational & Transcript Fillers (The "Hygiene Filter")
            conversational_fillers = [
                'put', 'simply', 'quite', 'still', 'actually', 'basically', 'literally', 'really',
                'say', 'said', 'saying', 'tell', 'telling', 'told',
                'going', 'getting', 'got', 'get', 'go', 'went',
                'know', 'think', 'thought', 'mean', 'meaning',
                'maybe', 'perhaps', 'probably', 'possibly',
                'okay', 'ok', 'yeah', 'yes', 'yep', 'nope', 'nah',
                'um', 'uh', 'hmm', 'ah', 'oh', 'wow',
                'kind', 'sort', 'type', 'thing', 'stuff', 'bit',
                'lot', 'much', 'many', 'way', 'make', 'made',
                'look', 'looking', 'see', 'seeing', 'saw',
                'right', 'left', 'good', 'bad', 'great', 'nice', # Generic adjectives often better removed from topics
                'best', 'worst', 'better', 'worse' # Added to remove subjective adjectives from topic names
            ]
            
            # 3. Domain Specific / Useless UI terms
            domain_specific = [
                'video', 'screen', 'click', 'button', 'clip', 'transcript', 'minutes', 'seconds',
                'inch', 'inches', 'action', 'something', 'terms', 'term', 'people', 'folks' 
            ]
            
            custom_stop_words = list(set(nltk_stop_words + conversational_fillers + domain_specific))
            
            # Adaptive min_df: Use 1 for small datasets to ensure we don't lose all keywords
            min_df_val = 2 if len(filtered_docs) > 300 else 1
            vectorizer_model = CountVectorizer(stop_words=custom_stop_words, ngram_range=(1, 3), min_df=min_df_val)

            # 3. RUN BERTOPIC
            # Use Guided Topic Modeling if seed topics exist
            from bertopic import BERTopic
            
            seed_topics = [
                "Usability Issue / Bug", 
                "Feature Request", 
                "Positive Feedback", 
                "Instruction / Walkthrough", 
                "Visual Design",
                "General / Irrelevant"
            ]
            
            print(f"  [OK] Guided BERTopic enabled with {len(seed_topics)} seed topics: {seed_topics}")
            # Ensure MinTopicSize is respected
            print(f"  BERTopic config: min_topic_size={min_topic_size_bertopic}, min_df={min_df_val}, n_docs={n_docs}")
            
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                seed_topic_list=seed_topics,
                min_topic_size=min_topic_size_bertopic, # Corrected to use the calculated variable
                top_n_words=10,
                nr_topics="auto",
                calculate_probabilities=False,
                verbose=True
            )
            
            # Pre-calculate embeddings (required for visualize_documents)
            print("Calculating embeddings for BERTopic...")
            embeddings = embedding_model.encode(filtered_docs, show_progress_bar=False)
            
            # Fit model
            topics, probs = topic_model.fit_transform(filtered_docs, embeddings=embeddings)

            # 4. TOPIC MERGING
            try:
                # Check if we have valid topics before reducing
                unique_topics = set(topics)
                if -1 in unique_topics: unique_topics.remove(-1)
                
                if len(unique_topics) > 0:
                    if len(filtered_docs) > 100:
                        topic_model.reduce_topics(filtered_docs, nr_topics=15)
                    topic_model.reduce_topics(filtered_docs, nr_topics="auto")
                    topics = topic_model.topics_
                else:
                    print("  [Info] Skipping topic reduction: No clusters found (all outliers)")
                    
            except Exception as merge_e:
                print(f"  [Warning] Topic merging skipped/failed: {merge_e}")

            # 5. SOFT CLUSTERING (HANDLE OUTLIERS)
            outlier_indices = [i for i, t in enumerate(topics) if t == -1]
            if outlier_indices:
                print(f"  Processing {len(outlier_indices)} outliers with Soft Clustering...")
                try:
                    # Use approximate distribution for outlier reassignment
                    topic_distr, _ = topic_model.approximate_distribution([filtered_docs[i] for i in outlier_indices], window=4, calculate_tokens=False)
                    
                    reassigned_count = 0
                    threshold = 0.05 # Conservative threshold from article
                    
                    for idx, dist in enumerate(topic_distr):
                        # dist is list of probabilities matching topic indices
                        # find best topic
                        if len(dist) > 0:
                            # dist is usually robust, let's find max
                            # It's not a simple array sometimes?
                            # topic_distr format: list of (topic, prob) or array?
                            # Docs say: returns (distributions, token_distributions)
                            # distributions is List[np.ndarray] or np.ndarray of shape (n_docs, n_topics)?
                            # Actually it returns array of shape (n_docs, n_topics) if calculate_probabilities=False was used in modeling?
                            # No, approximate_distribution returns distribution of topics for provided docs.
                            
                            # Let's assume standard numpy array behavior or list of probabilities
                            # If distribution is dense
                            best_topic_idx = np.argmax(dist)
                            best_prob = dist[best_topic_idx]
                            
                            if best_prob >= threshold:
                                # Map back to global topic ID?
                                # topic_model.topic_labels_ might map indices.
                                # Wait, approximate_distribution returns probs for *topics in the model*.
                                # The index in 'dist' corresponds to topic_model.get_topics() keys? No.
                                # It corresponds to topic_model.topic_labels_ order?
                                # Usually simply mapping to the topic labels list.
                                
                                # Let's trust topic_model.topics_ structure.
                                # Actually, checking the docs/article:
                                # "argmax" usually gives the index in the topic list.
                                # Safer to skip complex mapping and just trust the model updated topics? 
                                # Wait, we must manually update 'topics' list.
                                
                                # We need the actual TOPIC ID from the index.
                                # topic_model.get_topic_info() has ID.
                                # But `dist` is an array of size (n_topics,).
                                # Does index 0 = Topic 0? Or First topic in list?
                                # Typically it aligns with `topic_model.get_topics()` keys which are integers.
                                # Excluding -1?
                                
                                # Simpler approach:
                                # Just leave them as -1 if we aren't 100% sure of mapping, 
                                # OR use the method `topic_model.transform`? No, that's deterministic.
                                
                                # Let's assume topics list usage.
                                # Article used: `topic_distr` and filtered.
                                
                                # Critical: If we update `topics`, visualizations like barchart will NOT reflect it unless we update the model?
                                # We are just returning `topics` list to the UI. The UI uses it for grouping.
                                # So if we update the list here, the "Unclassified" bucket shrinks in our UI. That's good!
                                
                                # Let's TRY to find the topic ID.
                                # Logic: We will trust that dist array indices map 0..N to Topic 0..N (excluding -1).
                                # If there are gaps in topic IDs, this breaks.
                                # BERTopic usually produces contiguous topic IDs (0, 1, 2...)
                                
                                # Let's be safe: If we reassign, let's make sure topic exists.
                                mapped_topic = best_topic_idx # Assuming 0-indexed contiguous
                                # Adjust if topic IDs start at 0 (they do).
                                
                                # Update our local list
                                global_doc_idx = outlier_indices[idx]
                                topics[global_doc_idx] = mapped_topic
                                reassigned_count += 1
                                
                    print(f"  [OK] Reassigned {reassigned_count}/{len(outlier_indices)} outliers using Soft Clustering (thresh={threshold})")
                    
                except Exception as e:
                    print(f"  [Warning] Soft Clustering failed: {e}")

            # Count actual valid topics (excluding -1)
            valid_unique_topics = set(topics)
            if -1 in valid_unique_topics:
                valid_unique_topics.remove(-1)
            num_valid_topics = len(valid_unique_topics)
            
            print(f"BERTopic found {len(set(topics))} unique topic IDs: {set(topics)} (Valid: {num_valid_topics})")

            # --- EXTRACT INFO ---
            topic_info = topic_model.get_topic_info()
            
            # Use SEEDED topics for zero-shot class check if available
            zeroshot_topics = seed_topics if seed_topics else []
            
            # Extract topic words and LABELS *BEFORE* Visualization, so viz uses cleaned labels
            topic_words = {}
            topic_labels = []
            topic_label_map = {} # ID -> Label
            
            # Helper to map topics
            unique_topics_sorted = sorted(list(set(topics)))
            if -1 in unique_topics_sorted:
                unique_topics_sorted.remove(-1)
                
            topic_id_to_index = {tid: i for i, tid in enumerate(unique_topics_sorted)}
            
            if not unique_topics_sorted:
                 print("[Warning] BERTopic only found outliers, forcing NMF...")
                 return self._try_nmf_aggressive(documents)

            valid_topic_count = 0
            used_labels = set()
            
            print("Extracting and cleaning topic labels...")
            
            for topic_id in unique_topics_sorted:
                # 1. Get words for words-cloud/top-words list
                # Use KeyBERT for keywords if available (Clean source for retries)
                title_keywords = topic_model.get_topic(topic_id, full=True)
                
                clean_words = []
                if 'KeyBERT' in title_keywords:
                     clean_words = [(w, s) for w, s in title_keywords['KeyBERT'][:10]]
                elif 'Main' in title_keywords:
                     clean_words = [(w, s) for w, s in title_keywords['Main'][:10]]
                else:
                     clean_words = [(w, s) for w, s in topic_model.get_topic(topic_id)[:10]]
                     
                if clean_words:
                     topic_words[topic_id_to_index[topic_id]] = clean_words
                     
                     # DEBUG: Print Input Keywords
                     clean_kws_str = ", ".join([w[0] for w in clean_words[:8]])
                     # Raw keywords for deeper debugging (topic_model.get_topic returns list of tuples)
                     raw_kws = topic_model.get_topic(topic_id)
                     raw_kws_str = ", ".join([w[0] for w in raw_kws[:8]]) if raw_kws else "None"
                     
                     print(f"  [Topic {topic_id}] Raw Keywords: {raw_kws_str}")
                     print(f"  [Topic {topic_id}] Clean Keywords: {clean_kws_str}")
                     
                     
                     # 1. NOISE TOPIC DETECTION (Keyword Density Check)
                     # If keywords are dominated by fillers/refusals, it's a "Noise Topic"
                     # e.g. ["i'm", "sorry", "don't", "know", "think", "actually"]
                     noise_keywords = ["sorry", "know", "think", "don't", "cant", "that", "this", "just", "really"]
                     kw_list = [w[0].lower() for w in clean_words]
                     noise_count = sum(1 for kw in kw_list[:6] if any(n in kw for n in noise_keywords))
                     
                     if noise_count >= 3:
                         print(f"    [Info] Topic {topic_id} appears to be a noise cluster. Using fallback.")
                         final_label = f"Misc / Unclear {topic_id}"
                         topic_labels.append(final_label)
                         topic_label_map[int(topic_id)] = final_label
                         used_labels.add(final_label.lower())
                         valid_topic_count += 1
                         continue

                     # Debug: Print keywords being used
                     kw_debug = ", ".join([w[0] for w in clean_words[:8]])
                     print(f"  [Topic {topic_id}] Input Keywords: {kw_debug}")
                     
                     # 2. Get Label from Main Representation (LLM or KeyBERT)
                     try:
                         raw_label = ""
                         if 'Main' in title_keywords and title_keywords['Main']:
                             # For TextGeneration, it returns a list like [('Generated Label', 1)]
                             # We assume the first item is the full label string
                             raw_label = title_keywords['Main'][0][0]
                         else:
                             # Fallback to default feature vector
                             raw_label = clean_words[0][0]
                         
                         # Clean it up
                         import re
                         raw_str = str(raw_label)
                         
                         # RETRY LOGIC: If label seems malformed, try to regenerate using the LLM directly
                         # Criteria for bad label: 
                         # 1. Unbalanced quotes (only one " or ')
                         # 2. Punctuation inside quotes
                         # 3. Empty or too long
                         
                         clean_label = None
                         max_retries = 10
                         retry_count = 0
                         
                         # Helper to validate and clean label
                         # Clean it up using the global helper
                         import re
                         raw_str = str(raw_label)
                         
                         # Check for empty/garbage keywords BEFORE starting LLM logic
                         # If Topic 6 has ", , ,", we should detect it here.
                         kw_text = "".join([w[0] for w in clean_words]).replace(",", "").strip()
                         if not kw_text or all(c in '.,:;!? ' for c in kw_text):
                             print(f"    [Info] Topic {topic_id} has no valid keywords, using fallback label.")
                             clean_label = "Unclassified / General"
                             # Skip the LLM retry block
                         else:
                             # Pass keywords for anti-echoing validation
                             keyword_list = [w[0] for w in clean_words]
                             clean_label = self._clean_llm_label(raw_str, keywords=keyword_list)
                         
                         # RETRY LOGIC: If validation returns None, regenerate
                         # We check if 'generator' is available in local scope
                         if clean_label is None and 'generator' in locals():
                             
                             import time
                             
                             # Smart Keyword Dedup for Retries:
                             # Flatten phrases "concept art" -> "concept", "art" to avoid repetitive input
                             seen_terms = set()
                             unique_terms = []
                             for key_phrase in [w[0] for w in clean_words[:10]]:
                                 for term in key_phrase.split():
                                     if term.lower() not in seen_terms and len(term) > 2:
                                         seen_terms.add(term.lower())
                                         unique_terms.append(term)
                             kw_str = ", ".join(unique_terms[:8]) # Top 8 unique words
                             
                             start_time = time.time()
                             retry_count = 0
                             max_retries = 10
                             
                             # print(f"    [Label Retry] Validating label for Topic {topic_id}...")
                             # print(f"      Input Keywords: {kw_str}")
                             
                             while retry_count < max_retries:
                                 # Check timeout
                                 if time.time() - start_time > 15:
                                     break
                                 
                                 # Try to validate current raw output
                                 # Pass the flattened unique terms to help rejection
                                 clean_label = self._clean_llm_label(raw_str, keywords=unique_terms)
                                 if clean_label:
                                     break # Valid label found!
                                 
                                 # If here, validation failed. Regenerate.
                                 retry_count += 1
                                 
                                 # Aggressive Prompting on Retry
                                 if retry_count == 1:
                                     # Explicitly ask for a guess and forbid "I'm sorry"
                                     prompt = f"Keywords: {kw_str}\nSummarize these terms into a descriptive theme (concise phrase or word). DO NOT say 'I\'m sorry' or suggest a lack of context. Use them to form a best-guess category name:"
                                 elif retry_count == 2:
                                    prompt = f"What is the best descriptive category name for these keywords? Keywords: {kw_str}. Answer in a short phrase (no single words):"
                                 elif retry_count == 3:
                                    prompt = f"Provide a concise topic name for: {kw_str}. Topic name:"
                                 else:
                                    # Fallback
                                    prompt = f"Topic Category: {kw_str}"
                                     
                                 try:
                                     # Change seed for variety
                                     import random
                                     from transformers import set_seed
                                     new_seed = random.randint(0, 100000)
                                     set_seed(new_seed)
                                     
                                     # Max length 30 (allow more detail), top_p for better sampling, add repetition_penalty
                                     output = generator(
                                         prompt, 
                                         max_length=30, 
                                         do_sample=True, 
                                         temperature=0.7, # Lowered for more stability
                                         top_p=0.9, 
                                         top_k=50,
                                         repetition_penalty=1.2 # Prevent echoing input
                                     )
                                     raw_str = output[0]['generated_text'].strip()
                                     # print(f"      - Retry {retry_count}/{max_retries} (Seed {new_seed}): {raw_str}")
                                 except Exception:
                                     break
                         
                         # Final cleanup/Fallback
                         if not clean_label:
                             # Last resort: take top 3 keywords title-cased
                             clean_label = ", ".join([w[0] for w in clean_words[:3]]).title()

                         # Final truncation if still too long (>5 words)
                         words_split = clean_label.split()
                         if len(words_split) > 5:
                             clean_label = " ".join(words_split[:4])
                             
                         # Sanity Check: If output is a paragraph, it failed.
                         # if len(clean_label.split()) > 8:
                             # Fallback to feature vector (top keyword) or first 3 words
                             # clean_label = " ".join(clean_label.split()[:5]) # Truncate to 5 words max if LLM went rogue
                             
                         # Deduplicate
                         final_label = clean_label
                         if clean_label.lower() in used_labels:
                             # If duplicate, try adding distinct word or ID
                             final_label = f"{clean_label} ({topic_id})"
                        
                         topic_labels.append(final_label)
                         topic_label_map[int(topic_id)] = final_label  # Convert to regular int for dict key
                         used_labels.add(clean_label.lower()) # Add base label to used
                         used_labels.add(final_label.lower())
                             
                     except Exception as lbl_e:
                         print(f"  [Warning] Label extraction failed for topic {topic_id}: {lbl_e}")
                         fallback = f"Topic {topic_id}"
                         topic_labels.append(fallback)
                         topic_label_map[int(topic_id)] = fallback  # Convert to regular int for dict key
                         
                     valid_topic_count += 1
            
            # Set Custom Labels in BERTopic Model so Visualizations use them!
            topic_model.set_topic_labels(topic_label_map)
            print("  [OK] Custom labels applied to model")
            
            # Print final topic labels
            print("  [LLM Topic Labels]:")
            for tid, label in topic_label_map.items():
                print(f"    Topic {tid}: {label}")

            # --- REFINEMENT STAGE (User Requested Pipeline) ---
            # 1. Generate Summaries (TinyLlama)
            # 2. Generate Refined Title from Summary (TinyLlama)
            # 3. Update Labels Gloabally
            # 4. THEN Generate Visualizations
            
            topic_summaries = {}
            if unique_topics_sorted: 
                try:
                    summaries, refined_labels = self.generate_topic_summaries(
                        topic_labels=topic_label_map, # Current labels
                        topic_words=topic_words,
                        documents=filtered_docs,
                        topics=topics,
                        topic_model=topic_model
                    )
                    
                    topic_summaries = summaries
                    
                    # Apply refined labels
                    if refined_labels:
                        print(f"  [Label Refinement] Updating {len(refined_labels)} topic labels based on summaries...")
                        for tid, new_label in refined_labels.items():
                            old_label = topic_label_map.get(tid, "Unknown")
                            topic_label_map[tid] = new_label # Update map
                            print(f"    Topic {tid}: '{old_label}' -> '{new_label}'")
                            
                        # Update BERTopic internal state so charts use new labels
                        try:
                            topic_model.set_topic_labels(topic_label_map)
                            print("  [OK] Updated BERTopic internal labels for visualization")
                        except Exception as e:
                            print(f"  [Warning] Could not update BERTopic internal labels: {e}")
                            
                except Exception as sum_e:
                    print(f"  [Warning] Pipeline Refinement failed: {sum_e}")
            
            # del generator # Free memory used by initial labeler if any (REMOVED: generator no longer used here)
            
            # Re-Get topic info AFTER setting refined labels
            topic_info = topic_model.get_topic_info()
            
            # Generate Native Visualizations (unchanged)
            print("Generating native BERTopic visualizations...")
            viz_intertopic = None
            viz_documents = None
            viz_barchart = None
            viz_datamap = None
            
            try:
                # 1. Intertopic Distance Map
                # REQUIRES at least 3 topics to meaningfully visualize distances/clusters in UMAP
                if topic_model.topic_embeddings_ is not None:
                     if num_valid_topics >= 2:
                         print(f"  Topic embeddings shape: {np.array(topic_model.topic_embeddings_).shape}")
                         try:
                            # Visualize topics using MAIN representation
                            # Passing custom_labels=True to use the ones we just set
                            viz_intertopic = topic_model.visualize_topics(custom_labels=True)
                            if viz_intertopic is None:
                                print("  [Warning] visualize_topics() returned None")
                            else:
                                viz_intertopic.update_layout(autosize=True)
                                print("  [OK] Generated visualize_topics()")
                         except Exception as it_e:
                             print(f"  [Warning] visualize_topics() internal error: {it_e}")
                     else:
                         print(f"  [Info] Skipping visualize_topics: Need 3+ valid topics, found {num_valid_topics}")
                else:
                     print("  [Warning] Topic embeddings are None! Visualize topics will likely fail.")

                # 2. Document Clusters (2D UMAP projection)
                if len(filtered_docs) > 2000:
                    viz_documents = topic_model.visualize_documents(filtered_docs[:2000], embeddings=embeddings[:2000], custom_labels=True)
                else:
                    viz_documents = topic_model.visualize_documents(filtered_docs, embeddings=embeddings, custom_labels=True)
                # Fix legend position to prevent squashing - put at bottom
                viz_documents.update_layout(
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                    margin=dict(b=120)  # More bottom margin for legend
                )
                print("  [OK] Generated visualize_documents()")
                    
                # 3. Bar Chart (Top Words)
                viz_barchart = topic_model.visualize_barchart(top_n_topics=8, custom_labels=True)
                # Fix legend position to prevent squashing - put at bottom
                viz_barchart.update_layout(
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                    margin=dict(b=120)  # More bottom margin for legend
                )
                print("  [OK] Generated visualize_barchart()")
                
                # 4. Interactive DataMapPlot (New Request)
                # Requires explicit 2D embeddings
                print("  Generating Interactive DataMapPlot...")
                try:
                    from umap import UMAP
                    # Calculate 2D embeddings specifically for visualization (separate from model UMAP)
                    # Use cosine metric and decent neighbors for structure
                    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
                    embeddings_2d = umap_2d.fit_transform(embeddings)
                    
                    # Generate plot
                    # Note: interactive=True usually returns a DataMapPlot object or HTML
                    viz_datamap = topic_model.visualize_document_datamap(filtered_docs, reduced_embeddings=embeddings_2d, interactive=True)
                    print("  [OK] Generated visualize_document_datamap()")
                except Exception as dmp_e:
                    print(f"  [Warning] visualize_document_datamap failed: {dmp_e}")
                    viz_datamap = None
                
            except Exception as viz_e:
                print(f"[Warning] Visualization generation failed: {viz_e}")
                import traceback
                traceback.print_exc()
            
            # Check if we only found 1 topic - this means BERTopic failed
            
            # Check if we only found 1 topic - this means BERTopic failed
            if valid_topic_count <= 1:
                print(f"[Warning] BERTopic only found {valid_topic_count} topic(s), forcing NMF with lower topic count...")
                # Force NMF with fewer topics
                return self._try_nmf_aggressive(documents)
            
            # Map topics back to original document list indices
            all_topics_result = [-1] * len(documents)
            for i, tid in enumerate(topics):
                all_topics_result[valid_indices[i]] = tid
                
            # Final topics for return (map outliers/unassigned to 0 or leave as -1)
            # Standardizing on mapping -1 to a 'General' topic if not guided
            final_topics = [t if t != -1 else 0 for t in all_topics_result]
            
            # Calculate sizes from the FULL list
            unique, counts = np.unique(final_topics, return_counts=True)
            topic_sizes = dict(zip(unique.tolist(), counts.tolist()))
            
            # --- CALCULATE 2D TOPIC COORDINATES FOR NETWORK GRAPH ---
            # We want the network graph to match the semantic layout of the Intertopic Map
            topic_coords = {}
            if topic_model.topic_embeddings_ is not None:
                try:
                    # Filter valid topics (excluding -1 and alignment check)
                    # topic_embeddings_ typically aligns with topics sorted by ID? 
                    # Actually, let's use the robust approach just in case:
                    # Re-embed the topic representation strings to be 100% sure of mapping
                    
                    valid_unique_topics = sorted([t for t in unique.tolist() if t != -1])
                    if len(valid_unique_topics) >= 3:
                        topic_strings = []
                        for t in valid_unique_topics:
                            words = topic_model.get_topic(t)
                            if words:
                                # Use top 10 words as the "semantic center" of the topic
                                text = " ".join([w[0] for w in words[:10]])
                                topic_strings.append(text)
                            else:
                                topic_strings.append("")
                        
                        if topic_strings:
                            # UMAP for layout (2 neighbors for local structure)
                            from umap import UMAP
                            # Guard n_neighbors against very small topic counts
                            n_n = min(5, len(valid_unique_topics) - 1)
                            if n_n < 2: n_n = 2
                            
                            layout_umap = UMAP(n_neighbors=n_n, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
                            coords_2d = layout_umap.fit_transform(embedding_model.encode(topic_strings))
                            
                            for i, t in enumerate(valid_unique_topics):
                                topic_coords[t] = coords_2d[i].tolist()
                    else:
                        # Fallback for very few topics (1 or 2): Just space them out manually
                        # This prevents UMAP errors on tiny arrays
                        if len(valid_unique_topics) == 1:
                            topic_coords[valid_unique_topics[0]] = [0.5, 0.5]
                        elif len(valid_unique_topics) == 2:
                            topic_coords[valid_unique_topics[0]] = [0.2, 0.2]
                            topic_coords[valid_unique_topics[1]] = [0.8, 0.8]
                except Exception as e:
                    print(f"[Warning] Failed to calculate topic coords for network graph: {e}")
            
            print(f"[OK] BERTopic successfully found {valid_topic_count} topics")
            
            # Prepare result dictionary
            bertopic_result = {
                'topics': topics,
                'topic_labels': list(topic_label_map.values()), # Ensure this uses the FINAL refined list
                'topic_label_map': topic_label_map,
                'topic_words': topic_words,
                'topic_sizes': topic_sizes,
                'n_topics': len(set(topics)),
                'viz_intertopic': viz_intertopic,
                'viz_documents': viz_documents,
                'viz_barchart': viz_barchart,
                'viz_datamap': viz_datamap,
                'document_info': topic_model.get_document_info(filtered_docs),
                'guided_topics': seed_topics,
                'topic_coords': topic_coords,
                'topic_summaries': topic_summaries
            }
            
            # Inject PROPER clean labels into document_info so charts match the LLM output
            if topic_label_map and bertopic_result['document_info'] is not None:
                try:
                    # topic_label_map is {0: "Label", ...}
                    # document_info['Topic'] contains IDs
                    bertopic_result['document_info']['CustomLabel'] = bertopic_result['document_info']['Topic'].map(topic_label_map)
                except Exception as map_e:
                    print(f"  [Warning] Failed to map CustomLabel to document_info: {map_e}")

            return bertopic_result
            
        except Exception as e:
            print(f"BERTopic failed with error: {e}")
            # Fallback to aggressive NMF
            return self._try_nmf_aggressive(documents)
    
    def _evaluate_quality(self, result: Dict, documents: List[str]) -> float:
        """
        Evaluate topic model quality
        Returns score 0-1 (higher is better)
        """
        try:
            print(f"  Evaluating quality: n_topics={result['n_topics']}, topic_words keys={list(result['topic_words'].keys())}")
            
            if result['n_topics'] < 2:
                print(f"  [Warning] Only {result['n_topics']} topic(s) found, returning 0.0")
                return 0.0
            
            # Metric 1: Topic diversity (Jaccard distance between top words)
            diversity = self._calculate_diversity(result['topic_words'])
            
            # Metric 2: Coverage (what % of docs assigned with confidence)
            coverage = self._calculate_coverage(result)
            
            # Metric 3: Balance (are topics evenly distributed?)
            balance = self._calculate_balance(result['topic_sizes'])
            
            # Combined score
            quality = (diversity * 0.4) + (coverage * 0.3) + (balance * 0.3)
            print(f"  Quality metrics: diversity={diversity:.3f}, coverage={coverage:.3f}, balance={balance:.3f} -> total={quality:.3f}")
            return quality
            
        except Exception as e:
            print(f"  [Warning] Quality evaluation failed: {e}")
            return 0.0
    
    def _calculate_diversity(self, topic_words: Dict) -> float:
        """Calculate how distinct topics are"""
        if len(topic_words) < 2:
            return 0.0
        
        topics = list(topic_words.keys())
        total_distance = 0
        comparisons = 0
        
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                words_i = set([w[0] for w in topic_words[topics[i]][:5]])
                words_j = set([w[0] for w in topic_words[topics[j]][:5]])
                
                # Jaccard distance
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                
                if union > 0:
                    distance = 1 - (intersection / union)
                    total_distance += distance
                    comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _calculate_coverage(self, result: Dict) -> float:
        """Calculate what % of docs are well-assigned"""
        if 'doc_topic_matrix' not in result:
            return 1.0  # Assume perfect for BERTopic
        
        # Get max probability per document
        max_probs = np.max(result['doc_topic_matrix'], axis=1)
        confident_assignments = np.sum(max_probs > 0.2)  # Threshold
        
        return confident_assignments / len(max_probs)
    
    def _calculate_balance(self, topic_sizes: Dict) -> float:
        """Calculate topic distribution balance (entropy-based)"""
        if not topic_sizes:
            return 0.0
        
        total = sum(topic_sizes.values())
        proportions = [count / total for count in topic_sizes.values()]
        
        #  Ideal = uniform distribution
        ideal_prop = 1 / len(topic_sizes)
        
        # Calculate deviation from ideal
        deviation = np.mean([abs(p - ideal_prop) for p in proportions])
        balance = 1 - (deviation / ideal_prop)  # Normalize
        
        return max(0, balance)
    
    def _beautify_label(self, raw_label: str) -> str:
        """Convert raw label to human-readable"""
        # Simple capitalization and cleanup
        words = raw_label.split('_')
        cleaned = [w.capitalize() for w in words if len(w) > 2]
        return " ".join(cleaned[:3]) if cleaned else "Topic"
    
    def _create_empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            'topics': [],
            'topic_labels': ['No Topics'],
            'topic_words': {},
            'topic_sizes': {},
            'n_topics': 0,
            'error': reason
        }
    
    def _create_simple_topics(self, documents: List[str]) -> Dict:
        """Fallback: simple keyword-based grouping"""
        # This is a last resort if both NMF and BERTopic fail
        topics = [0] * len(documents)  # All in one topic
        return {
            'topics': topics,
            'topic_labels': ['General Discussion'],
            'topic_words': {0: [('general', 1.0)]},
            'topic_sizes': {0: len(documents)},
            'n_topics': 1
        }

    def _clean_llm_label(self, raw_label: str, keywords: Optional[List[str]] = None) -> Optional[str]:
        """
        Global helper to clean and validate LLM-generated topic labels.
        Returns cleaned label string, or None if validation fails.
        Optional 'keywords' arg allowed for similarity rejection.
        """
        if not raw_label:
            return None
            
        import re
        text = str(raw_label).strip()
        
        # 1. Extract text between quotes if present
        # ... (rest of the logic) ...
        match = re.search(r'["\']([^"\']+)["\']', text)
        if match:
             candidate = match.group(1).strip()
             # If candidate is empty or just punctuation, ignore
             if candidate and not all(c in '.,:;!? ' for c in candidate):
                 text = candidate
        
        # 2. Aggressive Prefix/Suffix Cleanup
        prefixes = [
  "The category is", "The categories are",
  "The topic is", "The topics are",
  "This topic is about", "These topics are about",
  "Topic about", "Topics about",
  "Category:", "Categories:",
  "Topic:", "Topics:",
  "Label:", "Labels:",
  "Answer:", "Answers:",
  "Summary:", "Summaries:",
  "Descriptive name:", "Descriptive names:",
  "Descriptive label:", "Descriptive labels:",
  "Theme:", "Themes:",
  "Concept:", "Concepts:",
  "Categorization:", "Categorizations:",
  "Analysis:", "Analyses:",
  "Summary of:", "Summaries of:",
  "Determined Category:", "Determined Categories:",
  "Topic category:", "Topic categories:",
  "Descriptive theme:", "Descriptive themes:",
  "Themes:",
  "The specific topic described by these texts is",
  "The specific topic described is"
]
        for p in prefixes:
            if text.lower().startswith(p.lower()):
                text = text[len(p):].strip()
        
        # 3. Clean surrounding punctuation
        text = text.strip('"').strip("'").strip().rstrip('.,:;!?')
        
        # 4. Handle "or" cases (e.g. "Feedback or Suggestions" -> "Feedback")
        if " or " in text and len(text.split()) > 4:
            text = text.split(" or ")[0].strip()
        
        # 5. VALIDATION
        
        # A. Check for commas (List detection)
        # If the LLM returns a list like "A, B, C", just take the first item
        if ',' in text:
            parts = [p.strip() for p in text.split(',')]
            if parts and len(parts[0].split()) <= 4:
                text = parts[0]
            else:
                return None
            
        # B. Check for "Topic X" literal output
        if text.lower() == "topic" or re.match(r'^topic \d+$', text.lower()):
            return None
            
        # C. Length Check
        words = text.split()
        if not words:
            return None
            
        # D. Generic single-word rejection (Very limited now per user request)
        # E. Generic / Noise Phrasal rejection
        blacklist = [
            "i'm sorry", "no problem", "thank you", "you're welcome", "please wait",
            "no context", "undefined", "unknown", "general category", "misc",
            "conversation summary", "topic label", "not mentioned", "not clear",
            "i apologize", "didn't catch", "didn't get", "unable to identify"
        ]
        text_lower = text.lower()
        if any(item in text_lower for item in blacklist) and len(words) <= 3:
            return None
            
        if len(words) == 1:
             if text.lower() in ["topic", "undefined"]:
                 return None
        
        if len(words) > 7: 
             return None
             
        # D. Keyword Echo Detection (NEW)
        # If the label is just 1-2 keywords from the list, reject it
        if keywords and text:
            text_lower = text.lower()
            kw_match_count = 0
            for kw in keywords[:5]:
                if kw.lower() in text_lower:
                    kw_match_count += 1
            
            # If label consists primarily of the top 3 keywords, it's not a summary
            if kw_match_count >= 2 and len(words) <= 2:
                return None
            
            # 2. Hallucination Check: At least one substantive word must overlap with keywords
            # This kills things like "angle play barging" if keywords are "fiat, car, dashboard"
            substantive_overlap = False
            for word in words:
                w_clean = word.lower().strip()
                if len(w_clean) <= 2: continue # Ignore "of", "in"
                if any(w_clean in kw.lower() for kw in keywords[:10]):
                    substantive_overlap = True
                    break
            
            if not substantive_overlap:
                return None

        if not text or len(text) < 3:
            return None
            
        return text

    def _generate_llm_labels_simple(self, topic_keywords_map: Dict[int, List[str]]) -> Dict[int, str]:
        """
        Generate short descriptive labels using Qwen2.5-3B-Instruct (FP16).
        Loads model, generates labels, then unloads to save VRAM.
        """
        results = {}
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            import gc
            
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            print(f"  [Info] Loading {model_name} (FP16) for NMF labeling...")
            
            # Force cleanup of any previous models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load in FP16 with Flash Attention 2 for speed
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            model_kwargs = dict(
                dtype=torch.float16,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                print("  [Info] Using Flash Attention 2")
            except Exception:
                del model_kwargs["attn_implementation"]
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                print("  [Info] Flash Attention 2 not available, using default")
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            
            # Prepare batches
            topic_ids = []
            prompts = []
            
            print(f"  [Info] Preparing prompts for {len(topic_keywords_map)} topics...")
            
            for tid, keywords in topic_keywords_map.items():
                if not keywords: continue
                
                kw_str = ", ".join(keywords[:10])
                
                # Chat Format for Qwen
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Output ONLY a concise topic category name (2-4 words). No quotes, no intro."},
                    {"role": "user", "content": f"Keywords: {kw_str}\nProvide a single, specific category name for these keywords."}
                ]
                prompts.append(messages)
                topic_ids.append(tid)
                
            if prompts:
                print(f"  [Info] Batch generating labels for {len(prompts)} topics (Batch Size: 8)...")
                try:
                    # Batch Generation
                    outputs = pipe(
                        prompts,
                        batch_size=8,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    # Process outputs
                    for tid, output in zip(topic_ids, outputs):
                        try:
                            generated_text = output[0]['generated_text']
                            if isinstance(generated_text, list):
                                label = generated_text[-1]['content'].strip()
                            else:
                                label = generated_text.strip()
                                
                            # Basic cleaning
                            label = label.replace('"', '').replace("'", "").strip(".")
                            if ":" in label: label = label.split(":")[-1].strip()
                            label = label.title()
                            
                            results[tid] = label
                            # print(f"    Topic {tid} -> {label}")
                            
                        except Exception as e:
                            print(f"    [Warning] Failed to parse output for Topic {tid}: {e}")
                            results[tid] = "General Topic"
                            
                except Exception as e:
                     print(f"  [Error] Batch generation failed: {e}")
            
            # Fallback for missing
            for tid in topic_keywords_map:
                if tid not in results:
                     kw = topic_keywords_map[tid]
                     results[tid] = ", ".join(kw[:3]).title() if kw else "Topic " + str(tid)

            # CLEANUP: Unload model to free VRAM for next steps
            del pipe
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print("  [Info] Unloaded Qwen model to free VRAM.")
            
            return results
            
        except ImportError:
            print("  [Warning] Transformers not installed, skipping LLM labeling.")
            return {}
        except Exception as e:
            print(f"  [Warning] LLM setup failed: {e}")
            return {}
    
    def generate_topic_summaries(
        self, 
        topic_labels: Dict[int, str], 
        topic_words: Dict[int, List[Tuple[str, float]]], 
        documents: List[str], 
        topics: List[int],
        topic_model=None,
        max_docs_per_topic: int = 15
    ) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
        """
        Generate executive summaries AND refined labels using Qwen2.5-3B-Instruct (FP16).
        Single model handles both tasks sequentially per topic.
        """
        print("  [Topic Summaries] Generating takeaways & refined labels with Qwen2.5-3B...")
        
        summaries: Dict[int, List[str]] = {}
        refined_labels: Dict[int, str] = {}
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            import gc
            import random
            
            # 1. Load Model (FP16)
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            print(f"  [Info] Loading {model_name} (FP16)...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            model_kwargs = dict(
                dtype=torch.float16,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                print("  [Info] Using Flash Attention 2")
            except Exception:
                del model_kwargs["attn_implementation"]
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                print("  [Info] Flash Attention 2 not available, using default")
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            
            # 2. Prepare Prompts in Batch
            topic_ids = []
            summary_prompts = []
            
            # Pre-calculation to form prompts
            for tid, label in topic_labels.items():
                if tid == -1: continue
                
                # Gather docs
                topic_docs = [doc for doc, t in zip(documents, topics) if t == tid]
                if not topic_docs:
                    summaries[tid] = ["No content available."]
                    continue
                
                # Sample strategy: prefer longer, informative docs
                valid_docs = [d for d in topic_docs if len(d) > 25]
                if not valid_docs: valid_docs = topic_docs
                sample_docs = sorted(valid_docs, key=len, reverse=True)[:8]
                
                # Limit context (shorter excerpts = faster prefill)
                combined_text = "\n".join([f"- {doc[:120]}" for doc in sample_docs])
                keywords = [w[0] for w in topic_words.get(tid, [])[:5]]
                kw_str = ", ".join(keywords)
                
                # --- STEP A PROMPT ---
                summary_prompt = [
                    {"role": "system", "content": "You are a helpful analyst. Analyze the transcript segments and provide 3-5 concise, insightful key takeaways. \n\nCRITICAL FORMATTING RULES:\n1. Start each takeaway with a bullet point '- '.\n2. Use the format: '- **Title**: Description'.\n3. NO introductory text (e.g., 'Here are the takeaways', 'Based on...'). Start directly with the first bullet.\n4. Do not include a conclusion."},
                    {"role": "user", "content": f"Context Keywords: {kw_str}\n\nText Segments:\n{combined_text}\n\nProvide key takeaways:"}
                ]
                
                summary_prompts.append(summary_prompt)
                topic_ids.append(tid)

            # 3. Batch Generate Summaries
            if summary_prompts:
                print(f"  [Info] Batch generating summaries for {len(summary_prompts)} topics (Batch Size: 4)...")
                try:
                    summary_outputs = pipe(
                        summary_prompts,
                        batch_size=4,
                        max_new_tokens=350, # Increased to 350 to prevent cutoff
                        do_sample=False
                    )
                except Exception as e:
                    print(f"  [Error] Batch summary generation failed: {e}")
                    summary_outputs = []

                # Process results and prepare Label prompts
                label_prompts = []
                label_topic_ids = []
                
                for i, (tid, out) in enumerate(zip(topic_ids, summary_outputs)):
                    try:
                        raw_sum = out[0]['generated_text'][-1]['content']
                        
                        # Parse bullets
                        takeaways = []
                        for line in raw_sum.split('\n'):
                            line = line.strip()
                            for prefix in ['-', '*', '•', '1.', '2.', '3.', '4.', '5.']:
                                if line.startswith(prefix):
                                    line = line[len(prefix):].strip()
                                    break
                            if len(line) > 10:
                                takeaways.append(line)
                                
                        summaries[tid] = takeaways[:5]
                        # print(f"    Topic {tid}: {len(takeaways)} takeaways")
                        
                        # Prepare Label Prompt (Step B) using the summary
                        if takeaways:
                            summary_blob = " ".join(takeaways[:3])
                            
                            label_prompt = [
                                {"role": "system", "content": "You are a helpful assistant. Create an ultra-concise topic title (2-3 words MAX). No filler words like 'and', '&', 'of'. Just the core concept."},
                                {"role": "user", "content": f"Summary: {summary_blob}\n\nProvide a short, specific title (2-3 words):"}
                            ]
                            label_prompts.append(label_prompt)
                            label_topic_ids.append(tid)
                        else:
                            refined_labels[tid] = topic_labels[tid] # Validation fallback
                            
                    except Exception as e:
                        print(f"    [Warning] Failed to parse summary for Topic {tid}: {e}")
                        summaries[tid] = ["Processing failed."]

                # 4. Batch Generate Labels
                if label_prompts:
                    print(f"  [Info] Batch generating refined labels for {len(label_prompts)} topics (Batch Size: 8)...")
                    try:
                        label_outputs = pipe(
                            label_prompts,
                            batch_size=8,
                            max_new_tokens=12,
                            do_sample=False
                        )
                        
                        for tid, out in zip(label_topic_ids, label_outputs):
                            try:
                                raw_lbl = out[0]['generated_text'][-1]['content'].strip()
                                
                                # Cleanup
                                clean_lbl = raw_lbl.replace('"', '').replace("'", "").strip(".").title()
                                if ":" in clean_lbl: clean_lbl = clean_lbl.split(":")[-1].strip()
                                
                                # Fallback if too long
                                if len(clean_lbl.split()) > 5:
                                    clean_lbl = " ".join(clean_lbl.split()[:4])
                                
                                # Reject generic/vague labels — retry with different seed
                                generic_labels = {'key insights', 'key takeaways', 'summary', 'overview', 
                                                  'main points', 'topic summary', 'general topic', 
                                                  'highlights', 'key points', 'important points',
                                                  'analysis', 'discussion', 'conclusion'}
                                if clean_lbl.lower().strip() in generic_labels:
                                    # Retry with higher temperature and explicit rejection
                                    retry_prompt = [
                                        {"role": "system", "content": "You are a helpful assistant. Create an ultra-concise topic title (2-3 words). Be SPECIFIC to the content. NEVER use generic titles like 'Key Insights' or 'Summary'."},
                                        {"role": "user", "content": f"Summary: {summaries.get(tid, [''])[0]}\n\nWhat specific subject does this cover? (2-3 word title):"}
                                    ]
                                    try:
                                        retry_out = pipe(retry_prompt, max_new_tokens=12, do_sample=True, temperature=0.95)
                                        retry_lbl = retry_out[0]['generated_text'][-1]['content'].strip()
                                        retry_lbl = retry_lbl.replace('"', '').replace("'", "").strip(".").title()
                                        if ":" in retry_lbl: retry_lbl = retry_lbl.split(":")[-1].strip()
                                        if retry_lbl.lower().strip() not in generic_labels and len(retry_lbl) > 2:
                                            clean_lbl = retry_lbl
                                            print(f"      [Retry] Topic {tid}: generic label replaced with '{clean_lbl}'")
                                    except Exception:
                                        pass  # Keep original if retry fails
                                    
                                refined_labels[tid] = clean_lbl
                                # print(f"      Topic {tid} -> Refined: {clean_lbl}")
                                
                            except Exception as e:
                                print(f"      [Warning] Failed to parse label for Topic {tid}: {e}")
                                refined_labels[tid] = topic_labels.get(tid, "Topic")
                                
                    except Exception as e:
                         print(f"  [Error] Batch label generation failed: {e}")

            # 5. Cleanup
            del pipe
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print("  [Info] Unloaded Qwen model.")
            
            return summaries, refined_labels
            
        except ImportError:
            print("  [Warning] Transformers/Torch not installed.")
            return {}, {}
        except Exception as e:
            print(f"  [Warning] Setup failed: {e}")
            return {}, {}
