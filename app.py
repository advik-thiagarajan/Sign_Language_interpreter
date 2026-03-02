import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_bundle = None

def load_model():
    global model_bundle
    try:
        with open('isl_model.pkl', 'rb') as f:
            model_bundle = pickle.load(f)
        print("✅ Optimized ISL Model Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.route('/predict_isl', methods=['POST'])
def predict_isl():
    if model_bundle is None: return jsonify({"prediction": "Model Offline"})
    
    try:
        data = request.json['landmarks']
        input_data = np.array(data).reshape(1, -1)
        
        model = model_bundle['model']
        expected_features = model_bundle['features']
        
        # Match input size to model requirements
        if input_data.shape[1] < expected_features:
            padded = np.zeros((1, expected_features))
            padded[0, :input_data.shape[1]] = input_data[0]
            input_data = padded
        elif input_data.shape[1] > expected_features:
            input_data = input_data[:, :expected_features]

        # Use probability to filter out "lazy" guesses
        probs = model.predict_proba(input_data)
        confidence = np.max(probs)

        # If the model isn't at least 50% sure, it's likely a bad reading
        if confidence < 0.5:
            return jsonify({"prediction": "Scanning..."})

        prediction = model.predict(input_data)
        return jsonify({"prediction": str(prediction[0])})
    except:
        return jsonify({"prediction": "..."})

if __name__ == '__main__':
    load_model()
    app.run(port=5000)