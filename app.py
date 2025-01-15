import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'model.keras'  # Path to your .keras model file
cnn = load_model(model_path)

# Create 'uploads' directory if it doesn't exist
uploads_dir = './uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction_text='No file uploaded.')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction_text='No selected file.')

    try:
        # Save the uploaded image to the 'uploads' directory
        img_path = os.path.join(uploads_dir, file.filename)
        file.save(img_path)

        # Load and preprocess the image
        test_image = image.load_img(img_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        result = cnn.predict(test_image)
        if result[0][0] == 1:
            output = 'Dog'
        else:
            output = 'Cat'

        # Provide image URL (note that we're passing the filename for Flask to serve)
        img_url = file.filename

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

    return render_template('index.html', prediction_text=f'The image is a: {output}', img_url=img_url)

# Route to serve the uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(uploads_dir, filename)

if __name__ == "__main__":
    app.run(debug=True)
