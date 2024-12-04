from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the pre-trained model structure (assuming it was trained on MNIST)
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[28, 28, 1]),  # Changed to match grayscale images
    keras.layers.Flatten(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile the model to match the training configuration
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

# Load pre-trained weights
weights_path = 'model.h5'
if os.path.exists(weights_path):
    try:
        model.load_weights(weights_path)
        print(f"Model weights loaded successfully from {weights_path}")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
else:
    raise FileNotFoundError(f"Model weights not found at '{weights_path}'. Ensure the file exists.")

# Class labels for Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Handle the uploaded image file
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Process the image
        try:
            # Convert the uploaded file to a grayscale image
            image = Image.open(file).convert('L')
            # Resize the image to 28x28 pixels (as required by the model)
            image = image.resize((28, 28))
            
            # Normalize the image data
            img_array = np.array(image) / 255.0
            
            # Reshape to match the input shape expected by the model
            img_array = img_array.reshape(1, 28, 28, 1)  # Add channel dimension for grayscale images
            
            # Predict the class
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_labels[predicted_class]

            # Generate a response and pass the labels correctly
            return render_template(
                'result.html',
                label=predicted_label,
                probabilities=predictions[0].tolist(),  # Convert numpy array to list for proper rendering in template
                class_labels=class_labels
            )
        except Exception as e:
            return jsonify({'error': f"Error processing the image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
