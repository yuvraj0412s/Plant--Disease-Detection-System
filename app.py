from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('models/Plant_Disease_Detection_Model_01.keras')

# Load the disease metadata
with open('plant_disease.json') as f:
    disease_info = json.load(f)

# Create a name-to-info lookup dictionary
disease_lookup = {item['name']: item for item in disease_info}

# Preprocess uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(160, 160))  # ✅ fixed to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "Empty file uploaded", 400

    uploads_dir = os.path.join('static', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    filepath = os.path.join(uploads_dir, file.filename)
    file.save(filepath)

    # Preprocess and predict
    img = preprocess_image(filepath)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction[0])

    class_labels = [item['name'] for item in disease_info]
    predicted_label = class_labels[predicted_index]
    result = disease_lookup.get(predicted_label, {})

    return render_template('result.html',
                           filename=f'uploads/{file.filename}',
                           disease=predicted_label,
                           cause=result.get('cause', 'Information not available'),
                           cure=result.get('cure', 'Information not available'),
                           symptoms=result.get('symptoms', 'Not specified'),
                           treatment=result.get('treatment', 'Not specified'))

# For viewing uploaded image
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
