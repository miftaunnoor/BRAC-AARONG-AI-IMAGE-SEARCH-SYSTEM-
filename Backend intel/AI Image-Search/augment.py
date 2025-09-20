import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
import pandas as pd

# Load ResNet50 model (without the final classification layer)
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# Folder containing all images     
image_folder = "images"

# Load existing features if they exist
existing_features = np.array([])
existing_paths = np.array([])
existing_design_nos = np.array([])

if os.path.exists("features.npy") and os.path.exists("image_paths.npy") and os.path.exists("design_nos.npy"):
    try:
        existing_features = np.load("features.npy")
        existing_paths = np.load("image_paths.npy")
        existing_design_nos = np.load("design_nos.npy")
        print(f"Loaded existing features: {len(existing_features)} images")
    except Exception as e:
        print(f"Warning: Could not load existing features: {e}")
        existing_features = np.array([])
        existing_paths = np.array([])
        existing_design_nos = np.array([])

# Get existing image design numbers for comparison
existing_design_set = set()
if len(existing_design_nos) > 0:
    existing_design_set = set(existing_design_nos)

# Collect all current image paths
all_image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith((".jpg", ".png"))]

# Identify new images that need feature extraction
new_image_paths = []
for img_path in all_image_paths:
    design_no = os.path.splitext(os.path.basename(img_path))[0]
    if design_no not in existing_design_set:
        new_image_paths.append(img_path)

print(f"Total images in folder: {len(all_image_paths)}")
print(f"Existing features: {len(existing_features)}")
print(f"New images to process: {len(new_image_paths)}")

# Extract features for new images only
new_features = []
new_valid_paths = []

if new_image_paths:
    print("Processing new images...")
    for i, img_path in enumerate(tqdm(new_image_paths, desc="Extracting features for new images", unit="img")):
        try:
            # Load and resize image to 224x224 (faster and required by ResNet50)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Extract features
            feat = model.predict(x, verbose=0)
            new_features.append(feat.flatten())
            new_valid_paths.append(img_path)

            # Show first 3 processed images so you know it's working
            if i < 3:
                print(f"Processed new: {img_path}")

        except Exception as e:
            print(f"Skipping {img_path}, error: {e}")

    # Convert new features to numpy arrays
    if new_features:
        new_features = np.array(new_features)
        new_valid_paths = np.array(new_valid_paths)
        print(f"Successfully processed {len(new_features)} new images")
    else:
        print("No new features extracted")
        new_features = np.array([])
        new_valid_paths = np.array([])
else:
    print("No new images to process")
    new_features = np.array([])
    new_valid_paths = np.array([])

# Combine existing and new features
if len(existing_features) > 0 and len(new_features) > 0:
    # Both existing and new features exist
    combined_features = np.vstack([existing_features, new_features])
    combined_paths = np.concatenate([existing_paths, new_valid_paths])
    print(f"Combined features: {len(combined_features)} total")
elif len(existing_features) > 0:
    # Only existing features
    combined_features = existing_features
    combined_paths = existing_paths
    print(f"Using existing features: {len(combined_features)} total")
elif len(new_features) > 0:
    # Only new features
    combined_features = new_features
    combined_paths = new_valid_paths
    print(f"Using new features: {len(combined_features)} total")
else:
    print("No features available!")
    combined_features = np.array([])
    combined_paths = np.array([])

# Save combined features and paths
if len(combined_features) > 0:
    np.save("features.npy", combined_features)
    np.save("image_paths.npy", combined_paths)
    print(f"Saved combined features: {len(combined_features)} images")

# Derive and save design numbers from metadata
metadata_candidates = [
    "product_metadata.csv", 
]

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
                # If no image_path column, we will try to match by filename stem later
                # Build a set of known design numbers to validate against
                for _, row in df.iterrows():
                    design_val = str(row["design_no"]).strip()
                    map_by_stem[design_val] = design_val
        # Align design numbers to combined_paths
        for p in combined_paths:
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
        print(f"Warning: Failed to read metadata from {metadata_file}: {e}. Falling back to filename stems.")
        for p in combined_paths:
            stem = os.path.splitext(os.path.basename(str(p)))[0]
            design_nos.append(stem)
else:
    print("Warning: No metadata CSV found. Using filename stems as design numbers.")
    for p in combined_paths:
        stem = os.path.splitext(os.path.basename(str(p)))[0]
        design_nos.append(stem)

design_nos = np.array(design_nos)
np.save("design_nos.npy", design_nos)

print(f"✅ Incremental feature extraction completed!")
print(f"   📊 Total features: {len(combined_features)}")
print(f"   🆕 New features processed: {len(new_features) if len(new_features) > 0 else 0}")
print(f"   📁 Total design numbers: {len(design_nos)}")
