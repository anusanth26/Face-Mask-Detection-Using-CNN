import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("face_mask_detection_model.h5")

st.title("Face Mask Detection using CNN")

st.write("This application detects whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN) model.")

st.write("To use the application, upload an image of a person, and the model will predict if the person is wearing a mask or not.")

# st.write("The model is trained on a dataset of images with and without masks, and it uses image preprocessing techniques to prepare the images for prediction.")

    
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    if st.button("Predict"):
        img = Image.open(uploaded_file)
        
        st.image(img,caption = "Uploaded Image",use_container_width=True)
        img = img.resize((128,128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array/255.0
        
        prediction = model.predict(img_array)
        
        if prediction[0][0] > 0.50:
            st.error("No mask detected")
        else:
            st.success("Mask detected")