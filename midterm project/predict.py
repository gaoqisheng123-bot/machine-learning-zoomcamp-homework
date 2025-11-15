import pickle
from flask import Flask, request, jsonify

model_file = 'heart_disease_model.pkl'

with open(model_file, 'rb') as f_in:
    artifacts = pickle.load(f_in)

dv = artifacts['vectorizer']
model = artifacts['model']

app = Flask('heart-disease-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    patient_data = request.get_json()

    X = dv.transform([patient_data])

    prediction_probability = model.predict_proba(X)[0, 1]

    has_heart_disease = prediction_probability >= 0.5

    result = {
        'heart_disease_probability': float(prediction_probability),
        'has_heart_disease': bool(has_heart_disease)
    }

    return jsonify(result)

if __name__ == '__main__':
    print("Starting the prediction service on port 9696...")
    app.run(debug=True, host='0.0.0.0', port=9696)