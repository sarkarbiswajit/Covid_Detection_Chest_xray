import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
loaded_model = load_model('my_cnn_model.h5')

# Define the class labels
class_labels = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the input size of the model (64x64)
    resized_image = cv2.resize(gray_image, (64, 64))
    # Reshape the image to add the channel dimension (1 for grayscale)
    input_image = resized_image.reshape((1, 64, 64, 1))
    # Normalize the pixel values
    input_image = input_image.astype('float32') / 255
    return input_image

# Function to make prediction using the model
def predict(image):
    input_image = preprocess_image(image)
    prediction = loaded_model.predict(input_image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    return predicted_label

# Streamlit app
def main():
    st.title("COVID-19 Chest X-ray Classification")
    st.write("Upload an image and get predictions from the trained model.")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        uploaded_image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make prediction using the model
        predicted_label = predict(uploaded_image)

        # Print the predicted label
        st.write("Predicted Label:", predicted_label)

if __name__ == "__main__":
    main()
