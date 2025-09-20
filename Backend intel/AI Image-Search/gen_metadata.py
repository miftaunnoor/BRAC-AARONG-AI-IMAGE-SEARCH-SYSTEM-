import os
import pandas as pd

# Folder containing images
image_folder = 'images'

# List all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Load existing metadata if it exists
existing_metadata = pd.DataFrame()
existing_design_nos = set()

if os.path.exists('product_metadata.csv'):
    try:
        existing_metadata = pd.read_csv('product_metadata.csv')
        existing_design_nos = set(existing_metadata['design_no'].astype(str))
        print(f"Found existing metadata with {len(existing_metadata)} entries")
    except Exception as e:
        print(f"Warning: Could not load existing metadata: {e}")
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
        print(f"Updating metadata for: {design_no}")
    else:
        # New entry
        new_count += 1
        print(f"Adding new metadata for: {design_no}")
    
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
combined_df.to_csv('product_metadata.csv', index=False)

print(f"\n📊 Metadata Update Summary:")
print(f"   ✅ New entries added: {new_count}")
print(f"   🔄 Existing entries updated: {updated_count}")
print(f"   📁 Total entries in metadata: {len(combined_df)}")
