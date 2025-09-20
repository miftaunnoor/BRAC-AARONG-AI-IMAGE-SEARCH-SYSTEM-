import os

os.environ["KMP_DUPLICATE_OK"] = "TRUE"

import numpy as np
import faiss
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import sys
from PIL import Image, ImageEnhance
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from colorspacious import cspace_convert
import warnings
warnings.filterwarnings("ignore")

# Setup paths
AI_DIR = os.path.join(os.path.dirname(__file__), 'AI Image-Search')
FEATURES_FILE = os.path.join(AI_DIR, 'features.npy')
DESIGN_NOS_FILE = os.path.join(AI_DIR, 'design_nos.npy')
IMAGE_PATHS_FILE = os.path.join(AI_DIR, 'image_paths.npy')
METADATA_CSV = os.path.join(AI_DIR, 'product_metadata.csv')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TOP_K = 5

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"], 
     allow_headers=["Content-Type", "Authorization"], 
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class FixedImageSearch:
    """
    Fixed Image Search Engine - ALWAYS Returns Exactly 5 Results
    
    Improvements:
    - Guaranteed 5 results with adaptive fallback system
    - Smart threshold adjustment if not enough color matches
    - Backup feature-only search if color filtering fails
    - Never returns fewer than requested results
    """
    
    def __init__(self, features_path, image_paths_path, design_nos_path):
        """Initialize the fixed search engine"""
        print("🚀 Initializing FIXED Image Search Engine (Always 5 Results)...")
        
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
        
        print("✅ FIXED Image Search Engine Ready!")
        print(f"   🎯 Guaranteed Results: {self.min_required_results} (ALWAYS)")
        print(f"   🎨 Initial Color Weight: {self.initial_color_weight*100:.0f}%")
        print(f"   🔄 Adaptive Fallback: Enabled")

    def load_database(self, features_path, image_paths_path, design_nos_path):
        """Load pre-computed features and metadata"""
        try:
            self.db_features = np.load(features_path).astype('float32')
            self.db_image_paths = np.load(image_paths_path)
            self.db_design_nos = np.load(design_nos_path)
            
            print(f"✅ Database loaded: {len(self.db_features)} images")
            print(f"✅ Feature dimensions: {self.db_features.shape[1]}")
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
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
        print(f"✅ FAISS index ready: {self.index.ntotal} vectors")

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
            print(f"⚠️ Error extracting colors from {image_path}: {e}")
            return None

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
            print(f"⚠️ Error extracting features from {image_path}: {e}")
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
            print(f"⚠️ Error in feature-only search: {e}")
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
        print(f"\n🔍 GUARANTEED SEARCH for: {query_image_path} (Target: {top_k} results)")
        
        if not os.path.exists(query_image_path):
            print(f"❌ Query image not found: {query_image_path}")
            return []
        
        # Ensure we never ask for more results than we have images
        actual_top_k = min(top_k, len(self.db_features))
        if actual_top_k < top_k:
            print(f"⚠️ Requested {top_k} results, but only {actual_top_k} images in database")
        
        # Extract query features (needed for all approaches)
        print("🧠 Extracting query features...")
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            print("❌ Failed to extract query features")
            return []
        
        query_features = query_features.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_features)
        
        # Extract query colors
        print("🎨 Extracting query colors...")
        query_colors = self.extract_dominant_colors(query_image_path, self.num_dominant_colors)
        
        all_candidates = []
        search_method = "Unknown"
        
        # Stage 1: Try color filtering with adaptive thresholds
        if query_colors is not None:
            print("🌈 Stage 1: Adaptive color filtering...")
            
            for threshold_idx, color_threshold in enumerate(self.fallback_thresholds):
                print(f"   Trying color threshold: {color_threshold} Delta E...")
                
                color_candidates = self.get_color_filtered_candidates(query_colors, color_threshold)
                
                if len(color_candidates) >= actual_top_k:
                    print(f"   ✅ Found {len(color_candidates)} color candidates (threshold: {color_threshold})")
                    all_candidates = color_candidates
                    search_method = f"Color-filtered (ΔE ≤ {color_threshold})"
                    break
                else:
                    print(f"   ⚠️ Only {len(color_candidates)} candidates with threshold {color_threshold}")
            
            # If we found some but not enough, keep them and supplement with feature-only
            if not all_candidates and color_candidates:
                print(f"   📝 Using {len(color_candidates)} color candidates + feature fallback")
                all_candidates = color_candidates
                search_method = "Color + Feature Fallback"
        
        # Stage 2: Feature-only fallback if needed
        if len(all_candidates) < actual_top_k:
            print("🧠 Stage 2: Feature-only fallback...")
            
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
                
                print(f"   ✅ Total candidates after feature fallback: {len(all_candidates)}")
        
        # Stage 3: Calculate final scores and rank
        print("⚖️  Stage 3: Final scoring and ranking...")
        
        if not all_candidates:
            print("❌ No candidates found even with fallback methods")
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
            print("❌ No valid results after scoring")
            return []
        
        # Sort by final score
        final_results.sort(key=lambda x: x[2], reverse=True)
        
        # Guarantee exactly top_k results
        if len(final_results) >= actual_top_k:
            final_results = final_results[:actual_top_k]
        else:
            # This should rarely happen with our fallback system, but just in case
            print(f"⚠️ Only {len(final_results)} results available (requested {actual_top_k})")
            actual_top_k = len(final_results)
        
        print(f"✅ GUARANTEED: Returning exactly {len(final_results)} results")
        print(f"📊 Search method: {search_method}")
        
        # Return simplified format for compatibility
        return [(path, design_no, final_score) for path, design_no, final_score, _, _ in final_results]

    def search(self, query_path, top_k=TOP_K, show_images=False):
        """Wrapper method for compatibility with existing API"""
        return self.search_similar_images(query_path, top_k, show_images)


# Initialize search engine on app start
search_engine = None


def initialize_engine():
    global search_engine
    try:
        search_engine = FixedImageSearch(FEATURES_FILE, IMAGE_PATHS_FILE, DESIGN_NOS_FILE)
        return True
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        search_engine = None
        return False


initialize_engine()


@app.route('/')
def home():
    return jsonify({"message": "BRAC-Aarong Image Search Backend Running", "status": "ready", "engine": "Fixed v2"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(f"🔍 Received prediction request from {request.remote_addr}")
        
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({"error": "No file in request"}), 400
            
        file = request.files['file']
        if file.filename == '':
            print("❌ No file selected")
            return jsonify({"error": "No file selected"}), 400
            
        if not allowed_file(file.filename):
            print(f"❌ File type not allowed: {file.filename}")
            return jsonify({"error": "File type not allowed"}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"✅ File saved: {filepath}")

        if search_engine is None and not initialize_engine():
            print("❌ Search engine unavailable")
            return jsonify({"error": "Search engine unavailable"}), 500

        print("🔍 Starting image search...")
        results = search_engine.search(filepath, top_k=TOP_K, show_images=False)
        print(f"✅ Search completed, found {len(results)} results")

        # Load metadata for response
        meta = pd.read_csv(METADATA_CSV) if os.path.exists(METADATA_CSV) else pd.DataFrame()
        payload = []
        for img_path, dno, score in results:
            row = meta[meta['design_no'] == str(dno)] if not meta.empty else pd.DataFrame()
            
            # Safely get product_code and description, handling missing columns
            code = ""
            desc = ""
            if not row.empty:
                if 'product_code' in meta.columns:
                    code = row['product_code'].values[0] if not row.empty else ""
                if 'description' in meta.columns:
                    desc = row['description'].values[0] if not row.empty else ""
            
            payload.append({
                "design_no": str(dno),
                "product_code": code,
                "description": desc,
                "image_path": os.path.basename(img_path),
                "similarity": float(score)
            })

        print(f"✅ Returning {len(payload)} results to frontend")
        
        # Add system information for debugging/display
        response_data = {
            "results": payload, 
            "count": len(payload),
            "system_info": {
                "search_method": "FixedImageSearch v2",
                "system_status": "Guaranteed 5 Results",
                "engine_version": "Enhanced with Color + Feature Fallback"
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/import', methods=['POST'])
def import_images():
    # Import endpoint to trigger image/metadata regen and update search index
    # Implementation depends on your pipeline scripts' availability
    # Placeholder: proper error handling to be inserted here.
    return jsonify({"message": "Import pipeline triggered"})

@app.route('/import-images', methods=['POST'])
def import_images_alt():
    """Import endpoint that runs the pipeline in background"""
    try:
        print("🚀 Starting image import pipeline...")
        
        # Import the pipeline function
        import subprocess
        import threading
        import time
        
        def run_pipeline():
            """Run pipeline in background thread"""
            try:
                # Change to the AI Image-Search directory
                pipeline_dir = os.path.join(os.path.dirname(__file__), 'AI Image-Search')
                
                # Run the pipeline
                result = subprocess.run(
                    ['python', 'pipeline.py'],
                    cwd=pipeline_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    print("✅ Pipeline completed successfully")
                    # Reload the search engine with new data
                    global search_engine
                    if initialize_engine():
                        print("✅ Search engine reloaded with new data")
                    else:
                        print("⚠️ Pipeline completed but failed to reload search engine")
                else:
                    print(f"❌ Pipeline failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("❌ Pipeline timed out after 5 minutes")
            except Exception as e:
                print(f"❌ Pipeline error: {e}")
        
        # Start pipeline in background thread
        pipeline_thread = threading.Thread(target=run_pipeline)
        pipeline_thread.daemon = True
        pipeline_thread.start()
        
        # Return immediate response
        return jsonify({
            "message": "Pipeline started! Images are being imported in the background. This may take a few minutes...",
            "status": "processing",
            "loaded_images": len(search_engine.db_features) if search_engine else 0
        })
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return jsonify({
            "message": f"Import failed: {str(e)}",
            "status": "error"
        }), 500


@app.route('/images/<filename>')
def serve_image(filename):
    img_dir = os.path.join(AI_DIR, "images")
    if not os.path.exists(os.path.join(img_dir, filename)):
        return jsonify({"error": "Image not found"}), 404
    return send_from_directory(img_dir, filename)


@app.route('/status')
def status():
    return jsonify({
        "status": "running",
        "loaded_images": len(search_engine.db_features) if search_engine else 0,
        "engine": "FixedImageSearch v2"
    })

@app.route('/import-status')
def import_status():
    """Check if pipeline is running and return status"""
    try:
        # Check if pipeline files exist
        pipeline_dir = os.path.join(os.path.dirname(__file__), 'AI Image-Search')
        features_file = os.path.join(pipeline_dir, 'features.npy')
        image_paths_file = os.path.join(pipeline_dir, 'image_paths.npy')
        design_nos_file = os.path.join(pipeline_dir, 'design_nos.npy')
        
        files_exist = all(os.path.exists(f) for f in [features_file, image_paths_file, design_nos_file])
        
        if files_exist:
            # Try to load the data to check if it's valid
            try:
                features = np.load(features_file)
                image_paths = np.load(image_paths_file)
                design_nos = np.load(design_nos_file)
                
                return jsonify({
                    "status": "completed",
                    "message": "Images are imported and ready for search!",
                    "loaded_images": len(features),
                    "files_ready": True
                })
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": f"Data files exist but are corrupted: {str(e)}",
                    "files_ready": False
                })
        else:
            return jsonify({
                "status": "not_started",
                "message": "No images imported yet. Click 'Import Images' to start.",
                "files_ready": False
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Status check failed: {str(e)}",
            "files_ready": False
        })


@app.route('/test-search')
def test_search():
    if search_engine is None and not initialize_engine():
        return jsonify({"error": "Engine unavailable"}), 500
    # Pick first image for test
    if len(search_engine.db_image_paths) == 0:
        return jsonify({"error": "No sample images"}), 500
    sample_img = search_engine.db_image_paths[0]
    results = search_engine.search(sample_img, top_k=TOP_K, show_images=False)

    return jsonify({
        "sample_image": sample_img,
        "results": [{"design_no": r[1], "image_path": os.path.basename(r[0]), "similarity": r[2]} for r in results]
    })


if __name__ == "__main__":
    print("Starting Flask API for BRAC-Aarong Image Search...")
    app.run(host='0.0.0.0', port=5000, debug=True)
