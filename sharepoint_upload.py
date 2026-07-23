# Databricks notebook source
# =============================================================================
# Upload a file from Databricks -> SharePoint (or OneDrive) via Microsoft Graph
# -----------------------------------------------------------------------------
# Auth:  OAuth2 client credentials (app-only / M2M).
# Write permission required (admin-consented):
#   SharePoint site libraries : Sites.ReadWrite.All  (or Sites.Selected + write role)
#   OneDrive                  : Files.ReadWrite.All
#
# Two upload modes, chosen automatically by file size:
#   <= 4 MiB  -> simple PUT .../content
#   >  4 MiB  -> resumable upload session (chunked PUT to the session URL)
#
# Official docs:
#   Upload small files   : https://learn.microsoft.com/en-us/graph/api/driveitem-put-content
#   Create upload session: https://learn.microsoft.com/en-us/graph/api/driveitem-createuploadsession
#   Permissions reference: https://learn.microsoft.com/en-us/graph/permissions-reference
# =============================================================================

import os
import requests

# COMMAND ----------
# ----------------------------- CONFIG ----------------------------------------
SECRET_SCOPE = "sharepoint"            # holds tenant_id, client_id, client_secret

# --- Source file in Databricks (UC Volume path or local /tmp) ---
SOURCE_PATH = "/Volumes/my_catalog/my_schema/outputs/report.pdf"

# --- Destination (SharePoint site) ---
SHAREPOINT_HOSTNAME  = "nbcuni.sharepoint.com"
SHAREPOINT_SITE_PATH = "sites/YourTeamSite"   # the /sites/<name> part of the URL
SHAREPOINT_LIBRARY   = "Documents"            # display name of the document library
DEST_FOLDER          = "Reports/2026"         # folder within the library ("" = root)
DEST_FILENAME        = None                   # None = keep the source filename

# replace | rename | fail
CONFLICT_BEHAVIOR = "replace"

GRAPH = "https://graph.microsoft.com/v1.0"
SIMPLE_UPLOAD_MAX = 4 * 1024 * 1024           # 4 MiB threshold
CHUNK = 10 * 1024 * 1024                       # 10 MiB == 32 * 320 KiB (valid multiple)

# COMMAND ----------
# ----------------------------- AUTH ------------------------------------------
def get_token() -> str:
    tenant_id     = dbutils.secrets.get(SECRET_SCOPE, "tenant_id")      # noqa: F821
    client_id     = dbutils.secrets.get(SECRET_SCOPE, "client_id")      # noqa: F821
    client_secret = dbutils.secrets.get(SECRET_SCOPE, "client_secret")  # noqa: F821
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }
    r = requests.post(url, data=data, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

# COMMAND ----------
# --------------------------- RESOLVERS ---------------------------------------
def get_site_id(token: str, hostname: str, site_path: str) -> str:
    url = f"{GRAPH}/sites/{hostname}:/{site_path}"
    r = requests.get(url, headers=auth_header(token), timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def get_drive_id(token: str, site_id: str, library_name: str) -> str:
    url = f"{GRAPH}/sites/{site_id}/drives"
    r = requests.get(url, headers=auth_header(token), timeout=30)
    r.raise_for_status()
    for d in r.json().get("value", []):
        if d.get("name") == library_name:
            return d["id"]
    raise ValueError(f"Library '{library_name}' not found on site {site_id}")


def dest_item_path(folder: str, filename: str) -> str:
    """Build the library-relative path for the destination file."""
    folder = (folder or "").strip("/")
    return f"{folder}/{filename}".strip("/") if folder else filename

# COMMAND ----------
# ------------------------- SMALL FILE UPLOAD ---------------------------------
def upload_small(token: str, drive_id: str, item_path: str, local_path: str) -> dict:
    """PUT the whole file in one request (<= 4 MiB)."""
    url = f"{GRAPH}/drives/{drive_id}/root:/{item_path}:/content"
    url += f"?@microsoft.graph.conflictBehavior={CONFLICT_BEHAVIOR}"
    with open(local_path, "rb") as f:
        data = f.read()
    headers = auth_header(token)
    headers["Content-Type"] = "application/octet-stream"
    r = requests.put(url, headers=headers, data=data, timeout=300)
    r.raise_for_status()
    return r.json()

# COMMAND ----------
# ------------------------- LARGE FILE UPLOAD ---------------------------------
def upload_large(token: str, drive_id: str, item_path: str, local_path: str) -> dict:
    """Resumable upload session with chunked PUTs (> 4 MiB)."""
    # 1) Create the upload session
    create_url = f"{GRAPH}/drives/{drive_id}/root:/{item_path}:/createUploadSession"
    body = {"item": {"@microsoft.graph.conflictBehavior": CONFLICT_BEHAVIOR}}
    r = requests.post(create_url, headers=auth_header(token), json=body, timeout=30)
    r.raise_for_status()
    upload_url = r.json()["uploadUrl"]

    # 2) PUT byte ranges in order. Each non-final chunk must be a multiple of
    #    320 KiB; the upload URL is pre-authenticated (no auth header needed).
    file_size = os.path.getsize(local_path)
    with open(local_path, "rb") as f:
        start = 0
        while start < file_size:
            chunk = f.read(CHUNK)
            end = start + len(chunk) - 1
            headers = {
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {start}-{end}/{file_size}",
            }
            resp = requests.put(upload_url, headers=headers, data=chunk, timeout=600)
            # 202 = more chunks expected; 200/201 = final chunk committed.
            if resp.status_code in (200, 201):
                return resp.json()
            if resp.status_code != 202:
                resp.raise_for_status()
            start = end + 1
    raise RuntimeError("Upload finished without a 200/201 completion response")

# COMMAND ----------
# ------------------------------ RUN ------------------------------------------
def upload_file(local_path: str) -> dict:
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)
    token = get_token()
    site_id  = get_site_id(token, SHAREPOINT_HOSTNAME, SHAREPOINT_SITE_PATH)
    drive_id = get_drive_id(token, site_id, SHAREPOINT_LIBRARY)

    filename = DEST_FILENAME or os.path.basename(local_path)
    item_path = dest_item_path(DEST_FOLDER, filename)
    size = os.path.getsize(local_path)

    print(f"Uploading {local_path} ({size/1e6:.2f} MB) -> {SHAREPOINT_LIBRARY}/{item_path}")
    if size <= SIMPLE_UPLOAD_MAX:
        result = upload_small(token, drive_id, item_path, local_path)
    else:
        result = upload_large(token, drive_id, item_path, local_path)
    print(f"  done. webUrl: {result.get('webUrl')}")
    return result


result = upload_file(SOURCE_PATH)

# COMMAND ----------
# ---- Notes -----------------------------------------------------------------
# * To upload to OneDrive instead of a site, resolve the drive with
#     GET /users/{user-upn}/drive   (needs Files.ReadWrite.All)
#   and reuse upload_small / upload_large unchanged with that drive_id.
# * conflictBehavior=replace overwrites an existing file of the same name;
#   use 'rename' to keep both, or 'fail' to error on a name clash.
