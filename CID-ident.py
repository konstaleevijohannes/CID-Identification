# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Model
from keras.layers import Layer
from keras.utils import to_categorical
from keras import layers, models, optimizers
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
num_classes = 8

y = to_categorical(y, num_classes=num_classes)
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

app = Flask(__name__)

UPLOAD_FOLDER = 'C:/Users/KonstaLemettinen/Desktop/CID-identification/user-pics'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load and preprocess images
def load_and_preprocess_data(data_folder):
    images = []
    labels = []

    for label, folder in enumerate(os.listdir(data_folder)):
        for file in os.listdir(os.path.join(data_folder, folder)):
            img_path = os.path.join(data_folder, folder, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Resize images to a common size
            img = img / 255.0  # Normalize pixel values to the range [0, 1]
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Load and preprocess training data
training_folder = 'C:/Users/KonstaLemettinen/Desktop/CID-identification/training-material/'
X, y = load_and_preprocess_data(training_folder)


# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

# Define a simple convolutional neural network (CNN) model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))  # For binary classification, use 'sigmoid'; for multiclass, use 'softmax'

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Function to predict if a user-uploaded image has similar symptoms
def predict_image(symptom_image_path):
    img = cv2.imread(symptom_image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    return prediction

# Function to check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
       
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict using the machine learning model
            prediction_result = predict_image(filepath)

            return render_template('result.html', prediction_result=prediction_result)
    
    return render_template('index.html', message='Upload a picture')

if __name__ == '__main__':
    app.run(debug=True)
