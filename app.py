from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('crop_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[feature]) for feature in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        prediction = model.predict([np.array(data)])
        return render_template('index.html', result=f"Recommended Crop: {prediction[0]}")
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
