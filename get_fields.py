import requests
import json
import sys

# OAuth configuration with provided credentials
client_id = "6fae0f29-b734-45e9-9e8e-68e261a6a0ce"
client_secret = "q.6.G-PEPSMnat~3gvK6I.2L4SK0J.BrzoskaFound.wkapi"
token_url = "https://borg.wolterskluwer.pl/core/v2.0/connect/token"

# User credentials required for ROPC flow
username = "michal.wyrebkowski@sprawdzamy.com"
password = "cay-wvt9UNA*hqr7cnc"

print("Starting fields retrieval script...")

# Prepare data for token request using the ROPC flow
token_data = {
    "client_id": client_id,
    "client_secret": client_secret,
    "grant_type": "password",
    "username": username,
    "password": password,
    "scope": "api://7f0641a3-1b27-45a5-b704-5d812c4c47a8/access_as_user offline_access"
}

# Request an access token
print("Requesting access token...")
try:
    token_response = requests.post(token_url, data=token_data, timeout=30)
    token_response.raise_for_status()
    token_json = token_response.json()
    access_token = token_json.get("access_token")
    print("Access token obtained:", access_token[:30] + "..." if access_token else "None")
except Exception as e:
    print(f"Error obtaining token: {str(e)}")
    sys.exit(1)

# Set up the API URL for fields endpoint
fields_api_url = "https://api.lex.pl/v4/fields"

# Set the authorization header with the access token
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# Function to get all fields from the API
def get_all_fields():
    try:
        print("\nFetching available fields from API...")
        response = requests.get(fields_api_url, headers=headers, timeout=60)
        
        if response.status_code != 200:
            print(f"Error response: {response.status_code}")
            print(f"Error details: {response.text}")
            return None
            
        # Save the raw response for reference
        with open("fields_raw_response.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
            
        # Parse the JSON response
        fields_data = response.json()
        
        # Save the structured JSON for reference
        with open("fields_raw.json", "w", encoding="utf-8") as f:
            json.dump(fields_data, f, indent=2)
            
        # Process and return the fields information
        return fields_data
        
    except Exception as e:
        print(f"Error retrieving fields: {str(e)}")
        return None

# Function to extract field information in a more readable format
def process_fields(fields_data):
    if not fields_data:
        return []
        
    field_info = []
    
    for field in fields_data:
        field_name = field.get("fieldId")
        field_type = field.get("type")
        field_description = field.get("description", "")
        filterable = field.get("canFilter", False)
        sortable = field.get("canSort", False)
        
        field_info.append({
            "field": field_name,
            "type": field_type,
            "description": field_description,
            "filterable": filterable,
            "sortable": sortable
        })
        
    return field_info

# Get the fields data
fields_data = get_all_fields()

if fields_data:
    # Process the fields data
    field_info = process_fields(fields_data)
    
    # Save to CSV for easy viewing
    import csv
    with open("available_fields.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["field", "type", "description", "filterable", "sortable"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for field in field_info:
            writer.writerow(field)
    
    # Print summary
    print(f"\nFound {len(field_info)} available fields")
    print(f"Full field data saved to fields_raw.json and fields_raw_response.txt")
    print(f"Formatted fields saved to available_fields.csv")
    
    # Print a sample of fields that can be used for filtering
    filter_fields = [f["field"] for f in field_info if f["filterable"]]
    print(f"\nFields that can be used for filtering ({len(filter_fields)}):")
    for i, field in enumerate(filter_fields[:20]):  # Show first 20
        print(f"  {field}")
    if len(filter_fields) > 20:
        print(f"  ... and {len(filter_fields) - 20} more (see CSV file)")
    
    # Print date-related fields specifically
    date_fields = [f["field"] for f in field_info if "date" in f["field"].lower() and f["filterable"]]
    print(f"\nDate-related fields that can be used for filtering:")
    for field in date_fields:
        print(f"  {field}")
        
else:
    print("Failed to retrieve fields information.")

print("\nScript completed.") 