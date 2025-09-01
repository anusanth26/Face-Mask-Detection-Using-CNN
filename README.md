**Face Mask Detection Using CNN**

This project demonstrates a Convolutional Neural Network (CNN) model trained to detect whether individuals are wearing face masks. The model is built using TensorFlow/Keras and can be deployed as a web application using Streamlit.

**Technologies Used**

Deep Learning Framework: TensorFlow/Keras

Programming Language: Python

Web Application Framework: Streamlit

Image Processing: OpenCV

Model Format: .h5 (Keras HDF5 format)

**Dataset**

The model is trained on a dataset containing images of individuals with and without face masks. The dataset is divided into training and testing sets, with images resized to 128x128 pixels.

**Model Architecture**

The CNN model comprises several convolutional layers followed by dense layers, culminating in a binary classification output indicating whether a mask is present.

**Training Process**

Data Preprocessing: Images were resized to 128x128 pixels and normalized.

Model Training: The model was trained for 20 epochs using the Adam optimizer and binary cross-entropy loss function.
