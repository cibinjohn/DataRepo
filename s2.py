import os
import requests
import msal

SHAREPOINT_DOMAIN = "nbcuni.sharepoint.com"
SITE_NAME = "sites/skoonie/data"
TARGET_FOLDER = "Advanced Analytics/Media Analytics/WIP/Miguel/Databricks - Sharepoint Test"
LOCAL_FILE_PATH = "your_local_file.csv"  # Replace with your local file name

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
TENANT_ID = "YOUR_TENANT_ID"
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"

# SharePoint Site and Path Configuration
SHAREPOINT_DOMAIN = "yourtenant.sharepoint.com"  # e.g., contoso.sharepoint.com
SITE_NAME = "your-site-name"                      # e.g., 'Finance' or 'sites/Finance'
TARGET_FOLDER = "General"                        # Target folder path in the default Document Library
LOCAL_FILE_PATH = "sample.pdf"                   # Path to the local file to upload

# ------------------------------------------------------------------
# 2. Acquire Access Token using Client Credentials Flow
# ------------------------------------------------------------------
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]

app = msal.ConfidentialClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    client_credential=CLIENT_SECRET
)

token_result = app.acquire_token_for_client(scopes=SCOPE)

if "access_token" not in token_result:
    raise Exception(f"Authentication failed: {token_result.get('error_description')}")

access_token = token_result["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}

# ------------------------------------------------------------------
# 3. Get SharePoint Site ID & Drive ID
# ------------------------------------------------------------------
# Fetch Site ID
site_url = f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_DOMAIN}:/sites/{SITE_NAME}"
site_response = requests.get(site_url, headers=headers)
site_response.raise_for_status()
site_id = site_response.json()["id"]

# Fetch Default Document Library (Drive ID)
drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
drive_response = requests.get(drive_url, headers=headers)
drive_response.raise_for_status()
drive_id = drive_response.json()["id"]

# ------------------------------------------------------------------
# 4. Upload File to SharePoint
# ------------------------------------------------------------------
file_name = os.path.basename(LOCAL_FILE_PATH)
upload_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{TARGET_FOLDER}/{file_name}:/content"

with open(LOCAL_FILE_PATH, "rb") as file_data:
    upload_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/octet-stream"
    }
    response = requests.put(upload_url, headers=upload_headers, data=file_data)

if response.status_code in [200, 201]:
    print(f"✅ File '{file_name}' successfully uploaded to SharePoint!")
else:
    print(f"❌ Upload failed ({response.status_code}): {response.text}")
