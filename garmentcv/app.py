import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from PIL import Image, ImageOps

# Load saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fashion_custom_cnn.h5')

# Load test data for demonstration
@st.cache
def load_test_data():
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    return x_test, y_test

labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

st.title('Fashion MNIST CNN Demo')
model = load_model()

# Input method selection
choice = st.radio('Select input method:', ('Random', 'Index', 'Upload'))

x_test, y_test = load_test_data()

if choice == 'Upload':
    uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        # Open and preprocess the image
        img = Image.open(uploaded_file)
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        st.image(img, width=200, caption='Uploaded Image')

        # Predict
        input_img = img_array.reshape(1, 28, 28, 1)
        pred = model.predict(input_img)
        pred_label = labels[np.argmax(pred)]
        st.write(f"**Prediction:** {pred_label}")
    else:
        st.write('Please upload an image to classify.')

else:
    if choice == 'Random':
        idx = np.random.randint(len(x_test))
    else:
        idx = st.slider('Test sample index', 0, len(x_test)-1, 0)

    img = x_test[idx]
    st.image(img, width=200, caption=f"True: {labels[y_test[idx]]}")

    # Prepare and predict
    input_img = img.reshape(1,28,28,1).astype('float32') / 255.0
    pred = model.predict(input_img)
    pred_label = labels[np.argmax(pred)]
    st.write(f"**Prediction:** {pred_label}")