import requests

url = 'http://127.0.0.1:9696/predict'

dataset_id = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
data_path = kagglehub.dataset_download(dataset_id)

test_df = pd.read_csv(os.path.join(data_path, 'Test.csv'))

sample_rel_path = test_df.iloc[0]['Path'] 
test_img_path = os.path.join(data_path, sample_rel_path)

print(f"Sending image to Docker: {test_img_path}")

try:
    with open(test_img_path, 'rb') as f:
        response = requests.post(url, files={'file': f})
    
    print("✅ Prediction Result:")
    print(response.json())
except Exception as e:
    print(f"❌ Still failing: {e}")
    print("Check !docker logs traffic_service again to see the error.")