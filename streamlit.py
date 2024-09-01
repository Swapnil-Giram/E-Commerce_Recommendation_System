import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Adjust the path to the 'venv' directory
embeddings_path = 'venv/embeddings.pkl'
filenames_path = 'venv/filenames.pkl'

# Load feature list and filenames
feature_list = np.array(pickle.load(open(embeddings_path, 'rb')))

# Load filenames with correct path
filenames = [os.path.join('venv', 'images', file) for file in os.listdir('venv/images')]

# Initialize the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')


def feature_extraction(img, model):
    img = img.resize((224, 224))  # Resize the image to match model input
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    try:
        # Open the uploaded image file
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image)

        # Extract features from the uploaded image
        features = feature_extraction(uploaded_image, model)

        # Recommend similar images
        indices = recommend(features, feature_list)

        # Display recommendations
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])

    except Exception as e:
        st.error(f"An error occurred during feature extraction or recommendation: {e}")
