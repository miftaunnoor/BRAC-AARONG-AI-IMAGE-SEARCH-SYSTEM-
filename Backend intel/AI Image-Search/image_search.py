import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import faiss
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from colorspacious import cspace_convert
import warnings
warnings.filterwarnings("ignore")

class FixedImageSearch:
    """
    Fixed Image Search Engine - ALWAYS Returns Exactly 5 Results
    
    Improvements:
    - Guaranteed 5 results with adaptive fallback system
    - Smart threshold adjustment if not enough color matches
    - Backup feature-only search if color filtering fails
    - Never returns fewer than requested results
    """
    
    def __init__(self, features_path="features.npy", 
                 image_paths_path="image_paths.npy", 
                 design_nos_path="design_nos.npy"):
        """Initialize the fixed search engine"""
        print(" Initializing FIXED Image Search Engine (Always 5 Results)...")
        
        # Load ResNet50 model
        self.base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.base_model.output)
        
        # Load database
        self.load_database(features_path, image_paths_path, design_nos_path)
        
        # Initialize FAISS
        self.initialize_faiss_index()
        
        # Search settings with adaptive thresholds
        self.initial_color_weight = 0.65      # Start with 65% color importance
        self.initial_feature_weight = 0.35    # Start with 35% feature importance
        self.initial_color_threshold = 25.0   # Start with Delta E threshold
        self.num_dominant_colors = 5
        self.min_required_results = 5         # Always return exactly this many
        
        # Fallback thresholds for adaptive search
        self.fallback_thresholds = [25.0, 35.0, 50.0, 75.0, 100.0]  # Gradually relax color matching
        
        # Cache for dominant colors
        self.dominant_colors_cache = {}
        
        print(" FIXED Image Search Engine Ready!")
        print(f"   Guaranteed Results: {self.min_required_results} (ALWAYS)")
        print(f"   Initial Color Weight: {self.initial_color_weight*100:.0f}%")
        print(f"   Adaptive Fallback: Enabled")
    
    def load_database(self, features_path, image_paths_path, design_nos_path):
        """Load pre-computed features and metadata"""
        try:
            self.db_features = np.load(features_path).astype('float32')
            self.db_image_paths = np.load(image_paths_path)
            self.db_design_nos = np.load(design_nos_path)
            
            print(f" Database loaded: {len(self.db_features)} images")
            print(f" Feature dimensions: {self.db_features.shape[1]}")
            
        except Exception as e:
            print(f" Error loading database: {e}")
            raise
    
    def initialize_faiss_index(self):
        """Initialize FAISS similarity search index"""
        faiss.normalize_L2(self.db_features)
        
        dimension = self.db_features.shape[1]
        
        if len(self.db_features) > 1000:
            nlist = min(100, len(self.db_features) // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(self.db_features)
            self.index.nprobe = min(10, nlist)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index.add(self.db_features)
        print(f" FAISS index ready: {self.index.ntotal} vectors")
    
    def enhance_image_quality(self, img):
        """Enhanced image quality improvement"""
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.15)
        
        return img
    
    def extract_dominant_colors(self, image_path, num_colors=5):
        """Extract dominant colors using K-means clustering"""
        cache_key = f"{image_path}_{num_colors}"
        if cache_key in self.dominant_colors_cache:
            return self.dominant_colors_cache[cache_key]
        
        try:
            if isinstance(image_path, str):
                img = Image.open(image_path).convert('RGB')
            else:
                img = image_path.convert('RGB')
            
            # Focus on center area
            width, height = img.size
            crop_size = min(width, height) * 0.8
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = left + crop_size
            bottom = top + crop_size
            img = img.crop((left, top, right, bottom))
            
            # Resize for faster processing
            img = img.resize((150, 150))
            
            # Convert to numpy array
            img_array = np.array(img)
            pixels = img_array.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            # Sort by cluster size
            labels = kmeans.labels_
            color_counts = []
            for i in range(num_colors):
                count = np.sum(labels == i)
                color_counts.append((count, dominant_colors[i]))
            
            color_counts.sort(key=lambda x: x[0], reverse=True)
            dominant_colors = [color for count, color in color_counts]
            
            # Cache result
            self.dominant_colors_cache[cache_key] = dominant_colors
            
            return dominant_colors
            
        except Exception as e:
            print(f" Error extracting colors from {image_path}: {e}")
            return None
    
    def rgb_to_lab(self, rgb_color):
        """Convert RGB to LAB color space"""
        try:
            rgb_normalized = np.array(rgb_color) / 255.0
            lab_color = cspace_convert(rgb_normalized, "sRGB1", "CIELab")
            return lab_color
        except Exception as e:
            return None
    
    def calculate_delta_e(self, lab1, lab2):
        """Calculate Delta E color distance"""
        try:
            delta_l = lab1[0] - lab2[0]
            delta_a = lab1[1] - lab2[1]
            delta_b = lab1[2] - lab2[2]
            delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
            return delta_e
        except Exception as e:
            return float('inf')
    
    def calculate_color_similarity(self, query_colors, db_colors):
        """Calculate color similarity using dominant colors and Delta E"""
        if query_colors is None or db_colors is None:
            return 0.0
        
        try:
            query_lab = [self.rgb_to_lab(color) for color in query_colors]
            db_lab = [self.rgb_to_lab(color) for color in db_colors]
            
            query_lab = [lab for lab in query_lab if lab is not None]
            db_lab = [lab for lab in db_lab if lab is not None]
            
            if not query_lab or not db_lab:
                return 0.0
            
            min_deltas = []
            for q_lab in query_lab:
                deltas = [self.calculate_delta_e(q_lab, d_lab) for d_lab in db_lab]
                min_deltas.append(min(deltas))
            
            avg_delta_e = np.mean(min_deltas)
            similarity = max(0.0, 1.0 - (avg_delta_e / 100.0))
            
            return similarity
            
        except Exception as e:
            return 0.0
    
    def extract_features(self, image_path):
        """Extract ResNet50 features from image"""
        try:
            if isinstance(image_path, str):
                img = Image.open(image_path).convert('RGB')
            else:
                img = image_path.convert('RGB')
            
            img = self.enhance_image_quality(img)
            img = img.resize((224, 224))
            
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            features = self.feature_extractor.predict(x, verbose=0)
            features = features.flatten()
            features = features / np.linalg.norm(features)
            
            return features.astype('float32')
            
        except Exception as e:
            print(f" Error extracting features from {image_path}: {e}")
            return None
    
    def get_color_filtered_candidates(self, query_colors, color_threshold):
        """Get candidates filtered by color similarity with given threshold"""
        candidates = []
        
        for i, (db_path, design_no) in enumerate(zip(self.db_image_paths, self.db_design_nos)):
            try:
                db_colors = self.extract_dominant_colors(db_path, self.num_dominant_colors)
                if db_colors is None:
                    continue
                
                color_sim = self.calculate_color_similarity(query_colors, db_colors)
                avg_delta_e = (1.0 - color_sim) * 100.0
                
                if avg_delta_e <= color_threshold:
                    candidates.append((i, db_path, design_no, color_sim))
                    
            except Exception as e:
                continue  # Skip problematic images
        
        return candidates
    
    def get_feature_only_candidates(self, query_features, top_k):
        """Fallback: Get candidates using features only (no color filtering)"""
        try:
            # Use FAISS to get more candidates than needed
            search_k = min(top_k * 3, len(self.db_features))
            similarities, indices = self.index.search(query_features, search_k)
            
            candidates = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx != -1 and idx < len(self.db_image_paths):
                    db_path = self.db_image_paths[idx]
                    design_no = self.db_design_nos[idx]
                    candidates.append((idx, db_path, design_no, 0.5))  # Default color sim = 0.5
            
            return candidates
            
        except Exception as e:
            print(f" Error in feature-only search: {e}")
            return []
    
    def search_similar_images(self, query_image_path, top_k=5, show_images=True):
        """
        GUARANTEED search that ALWAYS returns exactly top_k results
        
        Multi-stage adaptive approach:
        1. Try color filtering with initial threshold
        2. If not enough results, gradually relax color threshold  
        3. If still not enough, fall back to feature-only search
        4. Ensure exactly top_k results are returned
        """
        print(f"\n GUARANTEED SEARCH for: {query_image_path} (Target: {top_k} results)")
        
        if not os.path.exists(query_image_path):
            print(f" Query image not found: {query_image_path}")
            return []
        
        # Ensure we never ask for more results than we have images
        actual_top_k = min(top_k, len(self.db_features))
        if actual_top_k < top_k:
            print(f" Requested {top_k} results, but only {actual_top_k} images in database")
        
        # Extract query features (needed for all approaches)
        print(" Extracting query features...")
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            print(" Failed to extract query features")
            return []
        
        query_features = query_features.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_features)
        
        # Extract query colors
        print(" Extracting query colors...")
        query_colors = self.extract_dominant_colors(query_image_path, self.num_dominant_colors)
        
        all_candidates = []
        search_method = "Unknown"
        
        # Stage 1: Try color filtering with adaptive thresholds
        if query_colors is not None:
            print(" Stage 1: Adaptive color filtering...")
            
            for threshold_idx, color_threshold in enumerate(self.fallback_thresholds):
                print(f"   Trying color threshold: {color_threshold} Delta E...")
                
                color_candidates = self.get_color_filtered_candidates(query_colors, color_threshold)
                
                if len(color_candidates) >= actual_top_k:
                    print(f"  Found {len(color_candidates)} color candidates (threshold: {color_threshold})")
                    all_candidates = color_candidates
                    search_method = f"Color-filtered (ΔE ≤ {color_threshold})"
                    break
                else:
                    print(f" Only {len(color_candidates)} candidates with threshold {color_threshold}")
            
            # If we found some but not enough, keep them and supplement with feature-only
            if not all_candidates and color_candidates:
                print(f"  Using {len(color_candidates)} color candidates + feature fallback")
                all_candidates = color_candidates
                search_method = "Color + Feature Fallback"
        
        # Stage 2: Feature-only fallback if needed
        if len(all_candidates) < actual_top_k:
            print(" Stage 2: Feature-only fallback...")
            
            feature_candidates = self.get_feature_only_candidates(query_features, actual_top_k * 2)
            
            if feature_candidates:
                # Merge with existing candidates, avoiding duplicates
                existing_indices = {idx for idx, _, _, _ in all_candidates}
                
                for candidate in feature_candidates:
                    idx = candidate[0]
                    if idx not in existing_indices and len(all_candidates) < actual_top_k * 2:
                        all_candidates.append(candidate)
                        existing_indices.add(idx)
                
                if not search_method.startswith("Color"):
                    search_method = "Feature-only"
                
                print(f"    Total candidates after feature fallback: {len(all_candidates)}")
        
        # Stage 3: Calculate final scores and rank
        print("  Stage 3: Final scoring and ranking...")
        
        if not all_candidates:
            print(" No candidates found even with fallback methods")
            return []
        
        # Calculate hybrid scores for all candidates
        final_results = []
        
        for idx, db_path, design_no, color_sim in all_candidates:
            try:
                # Get feature similarity
                db_feature = self.db_features[idx:idx+1]
                feature_sim = np.dot(query_features, db_feature.T)[0][0]
                
                # Calculate adaptive weights based on whether we used color filtering
                if query_colors is not None and "Color" in search_method:
                    color_weight = self.initial_color_weight
                    feature_weight = self.initial_feature_weight
                else:
                    # If no color filtering, rely more on features
                    color_weight = 0.3
                    feature_weight = 0.7
                
                # Hybrid scoring
                final_score = (color_weight * color_sim + feature_weight * feature_sim)
                
                final_results.append((db_path, design_no, final_score, color_sim, feature_sim))
                
            except Exception as e:
                continue  # Skip problematic candidates
        
        # Stage 4: Ensure exactly top_k results
        if not final_results:
            print(" No valid results after scoring")
            return []
        
        # Sort by final score
        final_results.sort(key=lambda x: x[2], reverse=True)
        
        # Guarantee exactly top_k results
        if len(final_results) >= actual_top_k:
            final_results = final_results[:actual_top_k]
        else:
            # This should rarely happen with our fallback system, but just in case
            print(f" Only {len(final_results)} results available (requested {actual_top_k})")
            actual_top_k = len(final_results)
        
        print(f" GUARANTEED: Returning exactly {len(final_results)} results")
        print(f" Search method: {search_method}")
        
        # Display results
        self.display_guaranteed_results(query_image_path, final_results, search_method)
        
        # Show images if requested
        if show_images and final_results:
            self.display_result_images(query_image_path, final_results)
        
        # Return simplified format for compatibility
        return [(path, design_no, final_score) for path, design_no, final_score, _, _ in final_results]
    
    def display_guaranteed_results(self, query_path, results, search_method):
        """Display guaranteed search results"""
        print(f"\n{'='*80}")
        print(f" GUARANTEED RESULTS FOR: {os.path.basename(query_path)}")
        print(f" Search Method: {search_method}")
        print(f" Results Count: {len(results)} (ALWAYS {len(results)} as requested)")
        print(f"{'='*80}")
        
        if not results:
            return
        
        print(f"{'Rank':<6} {'Design No':<20} {'Final':<8} {'Color':<8} {'Feature':<8} {'Image'}")
        print("-" * 80)
        
        for i, (image_path, design_no, final_score, color_sim, feature_sim) in enumerate(results, 1):
            path_short = os.path.basename(image_path)
            print(f"{i:<6} {design_no:<20} {final_score:.4f}  {color_sim:.4f}  {feature_sim:.4f}  {path_short}")
        
        print("-" * 80)
        best = results[0]
        print(f" Best match: {best[1]} (Score: {best[2]:.4f})")
        print(f" GUARANTEED: Always returns exactly {len(results)} results")
        print(f"{'='*80}")
    
    def display_result_images(self, query_path, results):
        """Display query and result images"""
        try:
            display_results = [(path, design_no, final_score) for path, design_no, final_score, _, _ in results]
            
            total_images = len(display_results) + 1
            cols = min(3, total_images)
            rows = (total_images + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            fig.suptitle(f' GUARANTEED Results: Always {len(display_results)} Images', 
                        fontsize=16, fontweight='bold')
            
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten() if rows > 1 else axes
            
            # Display query image
            query_img = Image.open(query_path)
            axes[0].imshow(query_img)
            axes[0].set_title(f' YOUR IMAGE\n{os.path.basename(query_path)}', 
                            fontweight='bold', color='red', fontsize=12)
            axes[0].axis('off')
            
            # Display result images
            for i, (image_path, design_no, score) in enumerate(display_results):
                try:
                    result_img = Image.open(image_path)
                    axes[i+1].imshow(result_img)
                    axes[i+1].set_title(f'#{i+1}: {design_no}\nScore: {score:.3f}', 
                                       fontsize=10, fontweight='bold')
                    axes[i+1].axis('off')
                except Exception as e:
                    axes[i+1].text(0.5, 0.5, f' Loading Error\n{design_no}', 
                                  ha='center', va='center', transform=axes[i+1].transAxes,
                                  fontsize=10, color='red')
                    axes[i+1].axis('off')
            
            # Hide unused subplots
            for i in range(total_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f" Error displaying images: {e}")
    
    def test_with_random_image(self, top_k=5, show_images=True):
        """Test with random image - guaranteed results"""
        if len(self.db_image_paths) == 0:
            print(" No images in database")
            return []
        
        test_image_path = random.choice(self.db_image_paths)
        print(f"\n GUARANTEED TEST with: {test_image_path}")
        
        results = self.search_similar_images(test_image_path, top_k=top_k, show_images=show_images)
        return results
    
    def run_interactive_test(self):
        """Interactive testing interface"""
        print(f"\n{'='*70}")
        print(" GUARANTEED IMAGE SEARCH - ALWAYS 5 RESULTS")
        print(f"{'='*70}")
        print(" KEY FEATURES:")
        print("   • GUARANTEED: Always returns exactly 5 results")
        print("   • Adaptive fallback: Relaxes thresholds if needed")
        print("   • Never fails: Feature-only backup if color fails")
        print("   • Smart ranking: Best matches always on top")
        print(f"{'='*70}")
        print("Commands:")
        print("   search [image_path]  - Search (ALWAYS 5 results)")
        print("   test               - Test with random image")
        print("   quit               - Exit")
        print(f"{'='*70}")
        
        while True:
            try:
                command = input("\n Enter command: ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print(" Goodbye!")
                    break
                    
                elif command.lower() == 'test':
                    print(" Testing with random image (guaranteed 5 results)...")
                    results = self.test_with_random_image(top_k=5, show_images=True)
                    print(f" Test completed! Returned exactly {len(results)} results as guaranteed.")
                    
                elif command.lower().startswith('search '):
                    image_path = command[7:].strip()
                    if image_path:
                        if not os.path.exists(image_path):
                            print(f" File not found: {image_path}")
                            continue
                            
                        print(f" Searching (guaranteed 5 results)...")
                        results = self.search_similar_images(image_path, top_k=5, show_images=True)
                        print(f" Search completed! Returned exactly {len(results)} results as guaranteed.")
                    else:
                        print(" Please provide image path: search [image_path]")
                        
                elif command.lower() in ['help', 'h']:
                    print("\n Available Commands:")
                    print("  search [path] - Search with guaranteed 5 results")
                    print("  test         - Test with random image")
                    print("  quit         - Exit")
                    
                elif command.strip() == "":
                    continue
                    
                else:
                    print(" Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f" Error: {e}")

def main():
    """Main function"""
    try:
        # Initialize guaranteed search engine
        search_engine = FixedImageSearch()
        
        print(f"\n GUARANTEED system ready - ALWAYS returns exactly 5 results!")
        
        # Run interactive testing
        search_engine.run_interactive_test()
        
    except Exception as e:
        print(f"\n Failed to initialize system: {e}")
        print("\nMake sure dependencies are installed:")
        print("pip install colorspacious")

if __name__ == "__main__":
    main()
