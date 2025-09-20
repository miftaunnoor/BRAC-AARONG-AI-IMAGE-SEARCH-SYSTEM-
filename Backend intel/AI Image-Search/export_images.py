import os 
import base64
import pyodbc

output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

#Database connection Parameters 

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

# Connect to the database
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Get existing images to avoid duplicates
existing_images = set()
if os.path.exists(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            design_no = os.path.splitext(filename)[0]
            existing_images.add(design_no)

print(f"Found {len(existing_images)} existing images in database")

# Query to get design_no and Base64 image
cursor.execute("SELECT  design_no, Base64 FROM design_wise_base64_image_for_image_search_t")

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
                print(f"Updated: {img_path}")
                updated_images_count += 1
            else:
                # Add new image
                img_data = base64.b64decode(base64_str)
                img_path = os.path.join(output_dir, f"{design_no}.png")
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
                print(f"Added new: {img_path}")
                new_images_count += 1
        except Exception as e:
            print(f"Error with {design_no}: {e}")
            skipped_images_count += 1

cursor.close()
conn.close()

print(f"\n📊 Export Summary:")
print(f"   ✅ New images added: {new_images_count}")
print(f"   🔄 Existing images updated: {updated_images_count}")
print(f"   ⚠️  Images skipped (errors): {skipped_images_count}")
print(f"   📁 Total images in database: {len(existing_images) + new_images_count}")
