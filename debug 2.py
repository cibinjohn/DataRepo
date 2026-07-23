HOSTNAME = "nbcuni.sharepoint.com"
SITE_PATH = "sites/skoonie/data"
LIBRARY = "Shared Documents"
FOLDER_PATH = "Advanced Analytics/Media Analytics/WIP/Miguel/Databricks - Sharepoint Test"

site = await client.sites.by_site_id(f"{HOSTNAME}:/{SITE_PATH}:").get()
print(site.display_name, site.id)

drives = await client.sites.by_site_id(site.id).drives.get()
drive_id = next(d.id for d in drives.value if d.name == LIBRARY)
print(drive_id)

folder_item = await client.drives.by_drive_id(drive_id).root.item_with_path(FOLDER_PATH).get()
print(folder_item.id, folder_item.web_url)
-----------------------
import requests
resp = requests.get("https://login.microsoftonline.com/nbcuni.onmicrosoft.com/v2.0/.well-known/openid-configuration")
tenant_id = resp.json()["authorization_endpoint"].split("/")[3]

-----------------------
token = await credential.get_token("https://graph.microsoft.com/.default")
print("Token acquired:", token.token[:20], "...")
print("Expires on:", token.expires_on)
