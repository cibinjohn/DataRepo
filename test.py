import os, requests

TENANT_ID     = dbutils.secrets.get("sharepoint", "tenant_id")
CLIENT_ID     = dbutils.secrets.get("sharepoint", "client_id")
CLIENT_SECRET = dbutils.secrets.get("sharepoint", "client_secret")

HOSTNAME   = "nbcuni.sharepoint.com"
SITE_PATH  = "sites/YourTeamSite"      # the /sites/<name> segment
LIBRARY    = "Documents"               # document library display name
LOCAL_FILE = "/Volumes/my_catalog/my_schema/outputs/report.pdf"
DEST_PATH  = "Reports/2026/report.pdf" # path inside the library

GRAPH = "https://graph.microsoft.com/v1.0"

# 1. Token (client credentials)
tok = requests.post(
    f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
    data={"grant_type": "client_credentials", "client_id": CLIENT_ID,
          "client_secret": CLIENT_SECRET, "scope": "https://graph.microsoft.com/.default"},
).json()["access_token"]
H = {"Authorization": f"Bearer {tok}"}

# 2. Resolve site -> drive
site_id  = requests.get(f"{GRAPH}/sites/{HOSTNAME}:/{SITE_PATH}", headers=H).json()["id"]
drives   = requests.get(f"{GRAPH}/sites/{site_id}/drives", headers=H).json()["value"]
drive_id = next(d["id"] for d in drives if d["name"] == LIBRARY)

# 3. Upload (simple PUT — files up to 4 MiB)
with open(LOCAL_FILE, "rb") as f:
    r = requests.put(
        f"{GRAPH}/drives/{drive_id}/root:/{DEST_PATH}:/content"
        f"?@microsoft.graph.conflictBehavior=replace",
        headers={**H, "Content-Type": "application/octet-stream"},
        data=f.read(),
    )
r.raise_for_status()
print(r.json()["webUrl"])
