import msal
import requests
import jwt

# --------- CONFIGURATION ---------
CLIENT_ID = "appId"
CLIENT_SECRET = "secret"
TENANT_ID = "tenantId"

# Authority URL for your tenant
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

# Scope for application permissions (use .default)
SCOPE = ["https://graph.microsoft.com/.default"]

# Microsoft Graph API endpoint you want to call
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0/users"

# --------- CREATE MSAL CLIENT ---------
app = msal.ConfidentialClientApplication(
    client_id=CLIENT_ID,
    client_credential=CLIENT_SECRET,
    authority=AUTHORITY
)

# --------- ACQUIRE TOKEN ---------
result = app.acquire_token_for_client(scopes=SCOPE)

if "access_token" in result:
    access_token = result["access_token"]
    print("Access token acquired successfully!")

    # Decode JWT to inspect claims
    decoded_token = jwt.decode(access_token, options={"verify_signature": False})
    print("\nDecoded Access Token Claims:")
    for k, v in decoded_token.items():
        print(f"{k}: {v}")

    # --------- CALL MICROSOFT GRAPH ---------
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(GRAPH_API_ENDPOINT, headers=headers)

    if response.status_code == 200:
        print("\nGraph API Response:")
        print(response.json())
    else:
        print("\nGraph API Error:")
        print(response.status_code, response.text)
else:
    print("Failed to acquire token:")
    print(result.get("error"))
    print(result.get("error_description"))
