## Gender Recognition Model
#### This repository contains a Convolutional Neural Network (CNN) model for real-time gender recognition from facial images captured through a camera. The model is designed to predict whether a person in the camera feed is male or female.

* **Capture Camera Feed:** Access the camera feed using OpenCV.
* **Preprocess Images:** Preprocess the camera images to match the input dimensions required by the model.
* **Model Prediction:** Feed the preprocessed images into the trained model and obtain predictions for gender.
* **Display Results:** Display the camera feed with gender predictions overlayed in real-time.

## Model Architecture
The CNN model architecture is similar to the one described earlier, with slight modifications to handle real-time input from a camera.

* **Input Layer:** This layer accepts real-time facial images captured from the camera, with specified dimensions.

* **Convolutional Layers:** A stack of convolutional layers with increasing filters and decreasing spatial dimensions captures features from the camera images.

* **Max Pooling Layers:** These layers perform downsampling to reduce spatial dimensions.

* **Flatten Layer:** Converts the multi-dimensional feature maps into a flattened vector.

* **Dense Layers:** Fully connected layers further process the features and make predictions.

* **Output Layer:** The output layer predicts the gender based on the real-time camera feed.

## Dataset
The model was trained on a dataset containing labeled facial images of both male and female individuals. The dataset was divided into training and testing sets to ensure the model's performance is accurately assessed.
