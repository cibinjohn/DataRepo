HOSTNAME = "nbcuni.sharepoint.com"
SITE_PATH = "sites/skoonie/data"
LIBRARY = "Shared Documents"
FOLDER_PATH = "Advanced Analytics/Media Analytics/WIP/Miguel/Databricks - Sharepoint Test"

# Step1 resolve the site
# https://learn.microsoft.com/en-us/graph/api/site-getbypath?view=graph-rest-1.0
site = await client.sites.by_site_id(f"{HOSTNAME}:/{SITE_PATH}:").get()

# Step2 resolve the drive
# https://learn.microsoft.com/en-us/graph/api/drive-list?view=graph-rest-1.0
drives = await client.sites.by_site_id(site.id).drives.get()
drive_id = next(d.id for d in drives.value if d.name == LIBRARY)

# Step 3 — Upload (small files, <4MB)
# The Graph endpoint is a direct PUT to content, addressed by path relative to the drive root:
# https://learn.microsoft.com/en-us/graph/api/driveitem-put-content?view=graph-rest-1.0
# https://github.com/microsoftgraph/msgraph-sdk-python/discussions/874

# Doc: Upload or replace the contents of a driveItem — a PUT to /drives/{drive-id}/items/{item-id}/content returns a driveItem resource for the newly created file on success, with HTTP/1.1 201 Created. The root:/{path}: addressing syntax (path-based item addressing instead of a raw item ID) is confirmed working in the SDK per this GitHub discussion, which shows the identical pattern: items.by_drive_item_id('root:/Project/test_excel_upload.xlsx:').
# GitHub Discussion #874 — How to create an upload session 
# Microsoft Learn

file_bytes = open("local_file.pptx", "rb").read()
target_path = f"root:/{FOLDER_PATH}/local_file.pptx:"

uploaded_item = await client.drives.by_drive_id(drive_id).items.by_drive_item_id(target_path).content.put(file_bytes)
print(uploaded_item.web_url)




