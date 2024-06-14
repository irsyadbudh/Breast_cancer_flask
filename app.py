import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template
import os

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model_path = os.path.join(os.getcwd(), 'Brest CNN 2.h5')
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(50, 50))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    img_file = request.files['image']
    # Save the file to the upload folder
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
    img_file.save(img_path)
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    # Predict using the loaded model
    prediction = model.predict(processed_img)
    # Get the predicted class label
    predicted_class = np.argmax(prediction, axis=1)[0]
    # Map the predicted class to a meaningful label
    class_names = ['No Breast Cancer', 'Breast Cancer']
    result = class_names[predicted_class]
    # Return the result as JSON
    return jsonify({'result': result, 'image': img_path})

# Route to render the prediction HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
