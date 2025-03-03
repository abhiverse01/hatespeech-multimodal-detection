import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your pre-trained model
model = tf.keras.models.load_model('path_to_your_model.h5')

def predict(image):
    # Preprocess the image to match the input shape of the model
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    return prediction

st.title("Hate Speech Detection in Memes")
st.write("Upload an image to detect hate speech in memes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    prediction = predict(image)
    
    if prediction[0] > 0.5:
        st.write("The meme contains hate speech.")
    else:
        st.write("The meme does not contain hate speech.")