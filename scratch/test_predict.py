import requests
import os

url = "http://127.0.0.1:5000/predict"
image_path = r"c:\Users\Jayasri t\OneDrive\Desktop\lungs_disease\static\uploads\NORMAL1266.jpg"

if os.path.exists(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
else:
    print("Image not found")
