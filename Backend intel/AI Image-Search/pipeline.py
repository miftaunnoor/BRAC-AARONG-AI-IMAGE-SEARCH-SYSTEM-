#!/usr/bin/env python3
"""
Complete Image Search Pipeline (Incremental Updates)
Combines all image processing steps into one cohesive pipeline with incremental updates:
1. Export images from database (preserves existing, adds/updates new)
2. Generate metadata CSV (appends to existing, updates changed)
3. Generate augmented features (only processes new images)
4. Initialize image search system

Key Benefits:
- Preserves all existing data
- Only processes new images (saves significant time)
- Maintains data integrity
- Efficient daily updates
"""

import os
import base64
import pyodbc
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings("ignore")

def main():
    """Main pipeline execution function (incremental updates)"""
    print("Starting Complete Image Search Pipeline (Incremental Updates)")
    print("=" * 70)
    print("🔄 This pipeline preserves existing data and only processes new images")
    print("=" * 70)
    
    # Step 1: Export Images from Database
    print("\nSTEP 1: Exporting Images from Database (Incremental)")
    print("-" * 50)
    export_images_from_database()
    
    # Step 2: Generate Metadata CSV
    print("\nSTEP 2: Generating Metadata CSV (Incremental)")
    print("-" * 50)
    generate_metadata_csv()
    
    # Step 3: Generate Augmented Features
    print("\nSTEP 3: Generating Augmented Features (Incremental)")
    print("-" * 50)
    generate_augmented_features()
    
    # Step 4: Initialize Image Search System
    print("\nSTEP 4: Initializing Image Search System")
    print("-" * 50)
    initialize_image_search_system()
    
    print("\n🎉 Pipeline completed successfully!")
    print("📊 All existing data preserved, new data added incrementally")
    print("=" * 70)

def export_images_from_database():
    """Step 1: Export images from database to local folder (incremental)"""
    print("Connecting to database and exporting images (incremental)...")
    
    # Create output directory
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get existing images to avoid duplicates
    existing_images = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                design_no = os.path.splitext(filename)[0]
                existing_images.add(design_no)
    
    print(f"  📊 Found {len(existing_images)} existing images in database")
    
    # Database connection parameters
    server = "172.29.50.76"
    database = "ascm_working"
    username = "common"
    password = "com123"
    
    # Connection string for SQL Server
    conn_str = (
        'DRIVER={SQL Server};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password}'
    )
    
    try:
        # Connect to the database
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Query to get design_no and Base64 image
        cursor.execute("SELECT design_no, Base64 FROM design_wise_base64_image_for_image_search_t")
        
        new_images_count = 0
        updated_images_count = 0
        skipped_images_count = 0
        
        for design_no, base64_str in cursor.fetchall():
            if base64_str:
                try:
                    # Check if image already exists
                    if design_no in existing_images:
                        # Update existing image
                        img_data = base64.b64decode(base64_str)
                        img_path = os.path.join(output_dir, f"{design_no}.png")
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        updated_images_count += 1
                        if updated_images_count <= 3:  # Show first 3 for confirmation
                            print(f"  🔄 Updated: {img_path}")
                    else:
                        # Add new image
                        img_data = base64.b64decode(base64_str)
                        img_path = os.path.join(output_dir, f"{design_no}.png")
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        new_images_count += 1
                        if new_images_count <= 3:  # Show first 3 for confirmation
                            print(f"  ✅ Added new: {img_path}")
                except Exception as e:
                    skipped_images_count += 1
                    print(f"  ❌ Error with {design_no}: {e}")
        
        cursor.close()
        conn.close()
        
        print(f"  📊 Export Summary:")
        print(f"     ✅ New images added: {new_images_count}")
        print(f"     🔄 Existing images updated: {updated_images_count}")
        print(f"     ⚠️  Images skipped (errors): {skipped_images_count}")
        print(f"     📁 Total images in database: {len(existing_images) + new_images_count}")
        
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        print("  ⚠️  Please check your database connection parameters")
        raise

def generate_metadata_csv():
    """Step 2: Generate metadata CSV from exported images (incremental)"""
    print("📋 Creating metadata CSV from exported images (incremental)...")
    
    # Folder containing images
    image_folder = 'images'
    
    # Check if images folder exists
    if not os.path.exists(image_folder):
        print(f"  ❌ Images folder '{image_folder}' not found!")
        return
    
    # List all image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"  ❌ No image files found in '{image_folder}'!")
        return
    
    print(f"  📊 Found {len(image_files)} image files")
    
    # Load existing metadata if it exists
    existing_metadata = pd.DataFrame()
    existing_design_nos = set()
    
    if os.path.exists('product_metadata.csv'):
        try:
            existing_metadata = pd.read_csv('product_metadata.csv')
            existing_design_nos = set(existing_metadata['design_no'].astype(str))
            print(f"  📊 Found existing metadata with {len(existing_metadata)} entries")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load existing metadata: {e}")
            existing_metadata = pd.DataFrame()
    
    # Prepare data for new/updated images
    new_data = []
    updated_count = 0
    new_count = 0
    
    for filename in image_files:
        design_no = os.path.splitext(filename)[0]  # Extract design number from filename
        image_path = os.path.join(image_folder, filename)
        
        if design_no in existing_design_nos:
            # Update existing entry
            updated_count += 1
            if updated_count <= 3:  # Show first 3 for confirmation
                print(f"  🔄 Updating metadata for: {design_no}")
        else:
            # New entry
            new_count += 1
            if new_count <= 3:  # Show first 3 for confirmation
                print(f"  ✅ Adding new metadata for: {design_no}")
        
        new_data.append({
            'design_no': design_no,
            'image_path': image_path,
            # Add other fields as needed, e.g., 'product_name', 'category', etc.
        })
    
    # Create DataFrame for new data
    new_df = pd.DataFrame(new_data)
    
    # Merge with existing data (replace existing entries with updated ones)
    if not existing_metadata.empty:
        # Remove existing entries that are being updated
        existing_metadata = existing_metadata[~existing_metadata['design_no'].isin([row['design_no'] for row in new_data])]
        # Combine existing and new data
        combined_df = pd.concat([existing_metadata, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Save to CSV
    csv_path = 'product_metadata.csv'
    combined_df.to_csv(csv_path, index=False)
    
    print(f"  ✅ Metadata CSV updated: {csv_path}")
    print(f"  📊 Metadata Update Summary:")
    print(f"     ✅ New entries added: {new_count}")
    print(f"     🔄 Existing entries updated: {updated_count}")
    print(f"     📁 Total entries in metadata: {len(combined_df)}")

def generate_augmented_features():
    """Step 3: Generate augmented features using ResNet50 (incremental)"""
    print("🧠 Loading ResNet50 model and extracting features (incremental)...")
    
    # Load ResNet50 model (without the final classification layer)
    print("  🔄 Loading ResNet50 model...")
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base_model.input, outputs=base_model.output)
    print("  ✅ ResNet50 model loaded successfully")
    
    # Load existing features if they exist
    existing_features = np.array([])
    existing_paths = np.array([])
    existing_design_nos = np.array([])
    
    if os.path.exists("features.npy") and os.path.exists("image_paths.npy") and os.path.exists("design_nos.npy"):
        try:
            existing_features = np.load("features.npy")
            existing_paths = np.load("image_paths.npy")
            existing_design_nos = np.load("design_nos.npy")
            print(f"  📊 Loaded existing features: {len(existing_features)} images")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load existing features: {e}")
            existing_features = np.array([])
            existing_paths = np.array([])
            existing_design_nos = np.array([])
    
    # Get existing image design numbers for comparison
    existing_design_set = set()
    if len(existing_design_nos) > 0:
        existing_design_set = set(existing_design_nos)
    
    # Folder containing all images
    image_folder = "images"
    
    # Check if images folder exists
    if not os.path.exists(image_folder):
        print(f"  ❌ Images folder '{image_folder}' not found!")
        return
    
    # Collect all current image paths
    all_image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) 
                       if fname.endswith((".jpg", ".png", ".jpeg"))]
    
    if not all_image_paths:
        print(f"  ❌ No image files found in '{image_folder}'!")
        return
    
    # Identify new images that need feature extraction
    new_image_paths = []
    for img_path in all_image_paths:
        design_no = os.path.splitext(os.path.basename(img_path))[0]
        if design_no not in existing_design_set:
            new_image_paths.append(img_path)
    
    print(f"  📊 Total images in folder: {len(all_image_paths)}")
    print(f"  📊 Existing features: {len(existing_features)}")
    print(f"  📊 New images to process: {len(new_image_paths)}")
    
    # Extract features for new images only
    new_features = []
    new_valid_paths = []
    error_count = 0
    
    if new_image_paths:
        print("  🔄 Processing new images...")
        for i, img_path in enumerate(tqdm(new_image_paths, desc="  Extracting features for new images", unit="img")):
            try:
                # Load and resize image to 224x224 (required by ResNet50)
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                # Extract features
                feat = model.predict(x, verbose=0)
                new_features.append(feat.flatten())
                new_valid_paths.append(img_path)
                
                # Show first 3 processed images for confirmation
                if i < 3:
                    print(f"    ✅ Processed new: {os.path.basename(img_path)}")
                    
            except Exception as e:
                error_count += 1
                print(f"    ❌ Skipping {os.path.basename(img_path)}, error: {e}")
        
        # Convert new features to numpy arrays
        if new_features:
            new_features = np.array(new_features)
            new_valid_paths = np.array(new_valid_paths)
            print(f"  ✅ Successfully processed {len(new_features)} new images")
        else:
            print("  ⚠️  No new features extracted")
            new_features = np.array([])
            new_valid_paths = np.array([])
    else:
        print("  ℹ️  No new images to process")
        new_features = np.array([])
        new_valid_paths = np.array([])
    
    # Combine existing and new features
    if len(existing_features) > 0 and len(new_features) > 0:
        # Both existing and new features exist
        combined_features = np.vstack([existing_features, new_features])
        combined_paths = np.concatenate([existing_paths, new_valid_paths])
        print(f"  📊 Combined features: {len(combined_features)} total")
    elif len(existing_features) > 0:
        # Only existing features
        combined_features = existing_features
        combined_paths = existing_paths
        print(f"  📊 Using existing features: {len(combined_features)} total")
    elif len(new_features) > 0:
        # Only new features
        combined_features = new_features
        combined_paths = new_valid_paths
        print(f"  📊 Using new features: {len(combined_features)} total")
    else:
        print("  ❌ No features available!")
        return
    
    # Save combined features and paths
    print("  💾 Saving combined features and paths...")
    np.save("features.npy", combined_features)
    np.save("image_paths.npy", combined_paths)
    print("  ✅ Features saved: features.npy")
    print("  ✅ Paths saved: image_paths.npy")
    
    # Generate design numbers
    print("  🔢 Generating design numbers...")
    design_nos = generate_design_numbers(combined_paths)
    np.save("design_nos.npy", design_nos)
    print("  ✅ Design numbers saved: design_nos.npy")
    
    print(f"  🎉 Incremental feature extraction completed!")
    print(f"     📊 Total features: {len(combined_features)}")
    print(f"     🆕 New features processed: {len(new_features) if len(new_features) > 0 else 0}")
    print(f"     📁 Total design numbers: {len(design_nos)}")
    if error_count > 0:
        print(f"     ❌ Errors: {error_count} images")

def generate_design_numbers(valid_paths):
    """Generate design numbers from metadata or filenames"""
    # Look for metadata file
    metadata_candidates = ["product_metadata.csv"]
    
    metadata_file = None
    for candidate in metadata_candidates:
        if os.path.exists(candidate):
            metadata_file = candidate
            break
    
    design_nos = []
    
    if metadata_file is not None:
        try:
            df = pd.read_csv(metadata_file)
            map_by_path = {}
            map_by_basename = {}
            map_by_stem = {}
            
            if "design_no" in df.columns:
                if "image_path" in df.columns:
                    for _, row in df.iterrows():
                        img_path_val = str(row["image_path"]).strip()
                        design_val = str(row["design_no"]).strip()
                        map_by_path[os.path.normpath(img_path_val)] = design_val
                        map_by_basename[os.path.basename(img_path_val)] = design_val
                        map_by_stem[os.path.splitext(os.path.basename(img_path_val))[0]] = design_val
                else:
                    # If no image_path column, build design number set
                    for _, row in df.iterrows():
                        design_val = str(row["design_no"]).strip()
                        map_by_stem[design_val] = design_val
            
            # Align design numbers to valid_paths
            for p in valid_paths:
                p_norm = os.path.normpath(str(p))
                base = os.path.basename(p_norm)
                stem = os.path.splitext(base)[0]
                design = (
                    map_by_path.get(p_norm)
                    or map_by_basename.get(base)
                    or map_by_stem.get(stem)
                    or stem
                )
                design_nos.append(design)
                
        except Exception as e:
            print(f"    ⚠️  Warning: Failed to read metadata from {metadata_file}: {e}")
            print("    📝 Falling back to filename stems for design numbers")
            for p in valid_paths:
                stem = os.path.splitext(os.path.basename(str(p)))[0]
                design_nos.append(stem)
    else:
        print("    ⚠️  Warning: No metadata CSV found. Using filename stems as design numbers.")
        for p in valid_paths:
            stem = os.path.splitext(os.path.basename(str(p)))[0]
            design_nos.append(stem)
    
    return np.array(design_nos)

def initialize_image_search_system():
    """Step 4: Initialize the image search system"""
    print("🔍 Initializing image search system...")
    
    # Check if required files exist
    required_files = ["features.npy", "image_paths.npy", "design_nos.npy"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"  ❌ Missing required files: {missing_files}")
        print("  ⚠️  Please run the previous steps first")
        return
    
    # Load the data
    try:
        features = np.load("features.npy")
        image_paths = np.load("image_paths.npy")
        design_nos = np.load("design_nos.npy")
        
        print(f"  ✅ Loaded features: {features.shape}")
        print(f"  ✅ Loaded image paths: {len(image_paths)}")
        print(f"  ✅ Loaded design numbers: {len(design_nos)}")
        
        # Verify data consistency
        if len(features) != len(image_paths) or len(features) != len(design_nos):
            print("  ⚠️  Warning: Data length mismatch detected")
            print(f"     Features: {len(features)}")
            print(f"     Image paths: {len(image_paths)}")
            print(f"     Design numbers: {len(design_nos)}")
        
        print("  🎉 Image search system is ready!")
        print("  📝 You can now use image_search.py for searching")
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")

def test_pipeline():
    """Test the pipeline with a sample search"""
    print("\n🧪 Testing pipeline with sample search...")
    
    try:
        # Import the image search class
        from image_search import FixedImageSearch
        
        # Initialize search engine
        search_engine = FixedImageSearch()
        
        # Test with a random image
        if len(search_engine.db_image_paths) > 0:
            test_image = search_engine.db_image_paths[0]
            print(f"  🔍 Testing with: {os.path.basename(test_image)}")
            
            results = search_engine.search_similar_images(test_image, top_k=3, show_images=False)
            
            if results:
                print(f"  ✅ Test successful! Found {len(results)} similar images")
                for i, (path, design_no, score) in enumerate(results[:3]):
                    print(f"     {i+1}. {design_no} (score: {score:.4f})")
            else:
                print("  ⚠️  Test completed but no results returned")
        else:
            print("  ❌ No images available for testing")
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")

if __name__ == "__main__":
    try:
        main()
        
        # Ask if user wants to run a test
        print("\n" + "=" * 60)
        test_choice = input("🧪 Would you like to run a test search? (y/n): ").lower().strip()
        if test_choice in ['y', 'yes']:
            test_pipeline()
        
        print("\n🎉 Pipeline execution completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

