import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('traffic_sign_model.h5')

classes = {0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'}

@app.route('/')
def index():
    return render_template_string('<h1>Traffic Sign AI</h1><form method=post enctype=multipart/form-data action="/predict"><input type=file name=file><input type=submit value="Classify"></form>')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file uploaded'}), 400
    
    img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    res = model.predict(img_array)
    idx = int(np.argmax(res))
    
    return jsonify({
        'class_id': idx,
        'class_name': classes[idx], 
        'confidence': f"{float(np.max(res))*100:.2f}%"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)