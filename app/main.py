import os
import json
from PIL import Image


import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

def set_background(image_url):
    if image_url:
        st.markdown(
            f"""
            <style>
            .reportview-container {{
                background: url("{image_url}") no-repeat center center fixed;
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
def main():
 background_image_url = "https://cff2.earth.com/uploads/2020/05/07174714/shutterstock_718414630.jpg"
 set_background(background_image_url)
#Streamlit app

# Title and Subtitle

# Applying styling to the title and subtitle
st.markdown(
    "<h1 style='text-align: center; color: #008000;'>🌱 Plant Disease Classifier</h1>",
    unsafe_allow_html=True
)
st.write('')
st.markdown(
    "<p style='text-align: center; color: #138808; font-size: 16px;'>Upload an image and click the 'Classify' button to predict the disease.</p>",
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()

# Sidebar
# Applying color to the sidebar title
st.sidebar.markdown(
    "<h2 style='color: #138808;'>ℹ️ About</h2>",
    unsafe_allow_html=True
)
st.write('')
st.sidebar.info('This app classifies plant diseases using a convolutional neural network (CNN). '
                'Upload an image of a plant leaf and click the "Classify" button.')

st.sidebar.markdown(
    "<h2 style='color: #138808;'>🛈 How to Use</h2>",
    unsafe_allow_html=True
)
st.write('')
st.sidebar.write("1. Upload an image of a plant leaf using the file uploader on the main page.\n"
                 "2. Click the 'Classify' button to predict the disease affecting the plant.\n"
                 "3. The prediction result will be displayed below the image.")

# Main content
st.write('')
st.write('')
st.write('')
# Frontend design for the image uploader
uploaded_image = st.file_uploader("📷 Upload an image of a plant leaf (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")

st.write('')
st.write('')

# Check if an image is uploaded
if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)

    # Resize the image
    resized_img = image.resize((256, 256))

    # Display the resized image with a caption
    st.image(resized_img, caption='Uploaded Image', use_column_width=True)

st.write('')
st.write('')

# Classify button
if st.button('🔍 Classify'):
    if uploaded_image is None:
        st.warning("Please upload an image first.")
    else:
        # Placeholder for classification result
        with st.spinner('Classifying...'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)

        # Display result
        if prediction is not None:
            st.write('\n')  # Add space before result for better visualization
            st.subheader('Prediction Result')
            st.success(f'The Predicted Disease is: {prediction}')
        else:
            st.error("Failed to predict. Please try again.")


  # Provide additional information about the predicted class
        if prediction == 'Apple___Apple_scab':
            st.subheader('WHAT IS APPLE SCAB?')
            st.write('Apple scab is a common disease of plants in the rose family (Rosaceae)'
                     ' that is caused by the ascomycete fungus Venturia inaequalis.'
                     ' While this disease affects several plant genera, including Sorbus,'
                     ' Cotoneaster, and Pyrus, it is most commonly associated with the infection'
                     ' of Malus trees, including species of flowering crabapple, as well as cultivated apple.The first symptoms of this disease are found in the foliage, blossoms, and developing fruits of affected trees, which develop dark, irregularly-shaped lesions upon infection.Although apple scab rarely kills its host, infection typically leads to fruit deformation and premature leaf and fruit drop, which enhance the susceptibility of the host plant to abiotic stress and secondary infection.The reduction of fruit quality and yield may result in crop losses of up to 70%, posing a significant threat to the profitability of apple producers.To reduce scab-related yield losses, growers often combine preventive practices, including sanitation and resistance breeding, with reactive measures, such as targeted fungicide or biocontrol treatments, to prevent the incidence and spread of apple scab in their crops.')
            st.subheader('FUNGICIDES TO PREVENT APPLE SCAB:')
            st.write('1.Mancozeb ')
            st.write('2.Captan  3.Chlorothalonil  4.Dithiocarbamates (e.g., Ziram)  5.Dithiocarbamates (e.g., Ziram)  6.Phosphorous acid-based products (e.g., Fosetyl-Al)  7.Strobilurins (e.g., Azoxystrobin, Pyraclostrobin)')
        elif prediction == 'Apple___Black_rot':
            st.subheader('WHAT IS APPLE BLACK ROT?')
            st.write('Apple black rot, also known as black rot of apple, is a fungal disease caused by the pathogen Botryosphaeria obtusa (formerly known as Botryosphaeria dothidea). It primarily affects apple trees and can cause significant damage to fruit production if not managed properly. The disease typically appears as circular, sunken lesions on the fruit, which darken and expand as the infection progresses. These lesions may also produce spores, contributing to the spread of the disease. In addition to the fruit, black rot can also affect leaves, twigs, and branches, causing cankers, leaf spots, and dieback.')

            st.write('The fungus responsible for black rot thrives in warm, humid conditions, making it more prevalent in regions with such climates. Management strategies for black rot include cultural practices such as pruning to improve airflow within the canopy, sanitation to remove infected plant material, and fungicide applications to protect healthy tissue. Proper orchard management and timely application of control measures are essential for minimizing the impact of black rot on apple production.')
            st.subheader('FUNGICIDES TO PREVENT APPLE ROT:')
            st.write('1.Captan  2.Mancozeb  3.Thiophanate-methyl  4.Boscalid  5.Pyraclostrobin  6.Azoxystrobin  7.Fludioxonil 8.Myclobutanil')