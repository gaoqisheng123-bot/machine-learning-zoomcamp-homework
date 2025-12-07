import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from urllib import request

model_name = "hair_classifier_empty.onnx"
session = ort.InferenceSession(model_name)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize using NEAREST as specified in the previous homework context
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
    # Scale to 0-1
    x = np.array(img, dtype=np.float32) / 255.0
    # Transpose to (Channels, Height, Width)
    x = np.transpose(x, (2, 0, 1))
    # Add batch dimension -> (1, 3, 200, 200)
    x = np.expand_dims(x, axis=0)
    return x

def predict(url):
    img = download_image(url)
    target_size = (200, 200) # Based on the model input shape
    img = prepare_image(img, target_size)
    x = preprocess(img)

    # Run inference
    pred = session.run([output_name], {input_name: x})
    return pred[0]

def lambda_handler(event, context):
    url = event.get('url')
    result = predict(url)
    return result.tolist()
