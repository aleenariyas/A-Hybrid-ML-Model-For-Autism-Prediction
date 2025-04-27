from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from joblib import load
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained VGG19 model
base_model = VGG19(weights='imagenet', include_top=False)

# Load the SVM classifier
svm_classifier = load('svm_classifier_model.joblib')

# Define the path to the static folder
STATIC_FOLDER = 'static'

# Ensure the STATIC_FOLDER directory exists
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Set the app's upload folder to the defined path
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Define the path to the Excel sheet
EXCEL_FILE = 'predictions.xlsx'

# Function to preprocess and resize image
def preprocess_and_resize_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict autism
def predict_autism(image_path):
    processed_image = preprocess_and_resize_image(image_path)
    features = base_model.predict(processed_image)
    features_flattened = features.reshape(1, -1)
    prediction = svm_classifier.predict(features_flattened)
    return prediction[0]

# Function to save prediction result to Excel
def save_to_excel(filename, data):
    try:
        existing_data = pd.read_excel(filename)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['Image Filename', 'Prediction Result'])
    
    new_data = pd.DataFrame(data, index=[0])  # Convert dictionary to DataFrame
    existing_data = pd.concat([existing_data, new_data], ignore_index=True)
    existing_data.to_excel(filename, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        # Save the uploaded file to the STATIC_FOLDER directory
        file_path = os.path.join(app.config['STATIC_FOLDER'], filename)
        file.save(file_path)
        prediction = predict_autism(file_path)
        if prediction == 0:
            result = "Non-autistic"
        else:
            result = "Autistic"
        # Save the prediction result to Excel
        save_to_excel(EXCEL_FILE, {'Image Filename': filename, 'Prediction Result': result})
        # Pass the image filename and prediction result to the template
        return render_template('result.html', image_filename=filename, prediction_result=result)

# Function to save prediction result to Excel
def save_to_excel(filename, data):
    # Create a new DataFrame if the file doesn't exist or is empty
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        existing_data = pd.DataFrame(columns=['Date', 'Time', 'Image Filename', 'Prediction Result'])
    else:
        existing_data = pd.read_excel(filename)
    
    # Add current date and time to the data
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    # Create a new DataFrame for the new row
    new_row = pd.DataFrame({'Date': [date_str], 'Time': [time_str]})
    new_row = new_row.assign(**data)  # Merge new data with date and time
    
    # Concatenate the new row with existing data
    existing_data = pd.concat([existing_data, new_row], ignore_index=True)
    
    # Save the DataFrame to Excel
    existing_data.to_excel(filename, index=False)



if __name__ == '__main__':
    app.run(debug=True)
