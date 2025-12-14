import requests
from time import sleep


url = "http://localhost:9696/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client).json()

while True:
    response = requests.post(url, json=client).json()
    print(response)
    sleep(0.1)