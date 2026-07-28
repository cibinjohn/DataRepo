# Step 3: resolve the drive — use "Documents", matching your actual drive list output
LIBRARY = "Documents"
drive_id = next(d.id for d in drives.value if d.name == LIBRARY)
print(drive_id)

# Step 4: confirm the target folder resolves
FOLDER_PATH = "Advanced Analytics/Media Analytics/WIP/Miguel/Databricks - Sharepoint Test"

folder_item = await client.drives.by_drive_id(drive_id).items.by_drive_item_id(f"root:/{FOLDER_PATH}:").get()
print(folder_item.id, folder_item.web_url)

# Step 5: upload the file
file_path = "/local/path/to/your_file.pptx"
file_name = "your_file.pptx"

with open(file_path, "rb") as f:
    file_bytes = f.read()

target = f"root:/{FOLDER_PATH}/{file_name}:"
uploaded_item = await client.drives.by_drive_id(drive_id).items.by_drive_item_id(target).content.put(file_bytes)
print(uploaded_item.web_url)
