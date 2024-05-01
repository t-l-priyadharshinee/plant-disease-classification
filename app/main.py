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
            st.write('Apple black rot is a fungal disease caused by the pathogen Botryosphaeria obtusa. It primarily affects apple trees and can lead to significant damage to the fruit and foliage. The disease is characterized by the appearance of black, sunken lesions on the fruit, often starting at the blossom end and expanding over time. These lesions can also develop on the leaves, causing them to wither and die.'

                    'Apple black rot thrives in warm, humid conditions, and the fungus can overwinter in infected plant debris or in the bark of trees. Spores spread through rain or irrigation water, as well as through wind, insects, or contaminated equipment.'

                    'To manage apple black rot, orchard management practices such as pruning to improve airflow and reduce moisture around the trees, sanitation to remove infected plant material, and fungicide applications are often employed. Additionally, selecting apple varieties that are resistant to black rot can help prevent the disease from taking hold in orchards.')
            st.subheader('FUNGICIDES TO PREVENT APPLE ROT:')
            st.write('1. Captan  2. Mancozeb  3. Thiophanate-methyl  4. Myclobutanil  5. Pyraclostrobin  6. Boscalid')
        elif prediction == 'Apple___Cedar_apple_rust':
            st.subheader('WHAT IS CEDAR APPLE RUST?')
            st.write('Cedar apple rust is a fungal disease that affects apple and cedar trees. It is caused by the fungus Gymnosporangium juniperi-virginianae. The disease requires both apple or crabapple trees and Eastern red cedar or related junipers to complete its life cycle. '

'The fungus produces distinctive orange lesions on apple leaves, fruit, and sometimes on cedar trees. These lesions release spores that are carried by wind to infect cedar trees, where they create swollen, woody galls. These galls then release spores that infect apple trees, completing the cycle.'

'Cedar apple rust can cause significant damage to apple trees, including reduced fruit yield and quality. Control measures often involve planting resistant apple varieties, removing nearby cedar trees, and applying fungicides when necessary.')
            st.subheader('FUNGICIDES TO PREVENT CEDAR APPLE RUST:')
            st.write('1.Captan  2.Myclobutanil  3.Thiophanate-methyl  4.Mancozeb  5.Chlorothalonil  6.Tebuconazole  7.Propiconazole')
        elif prediction == 'Apple___healthy':
            st.subheader('HOW DO WE SAY THAT APPLE TREE IS HEALTHY?')
            st.write('A healthy apple tree is characterized by vibrant foliage, sturdy branches, and prolific fruit production. Its leaves are a rich green color, free from discoloration or signs of disease. The branches should be supple yet strong, capable of bearing the weight of numerous apples without bending or breaking. The trunk of the tree should be straight and robust, providing strong support for its structure. A healthy apple tree typically produces an abundance of fruit, with apples that are plump, firm, and free from blemishes or insect damage. Additionally, a healthy tree demonstrates resilience to environmental stresses, showing vigorous growth throughout the growing season. Regular pruning, adequate sunlight, proper watering, and timely pest management are essential practices in maintaining the health and vitality of an apple tree.')
            st.subheader('HOW TO MAINTAIN THE APPLE TREE HEALTHY:')
            st.write(
                '1. Pruning  2. Regular watering  3. Adequate sunlight  4. Fertilization  5. Pest control  6. Disease management  7. Mulching  8. Proper spacing  9. Monitoring for signs of stress or disease  10. Training branches for optimal structure')
        elif prediction == 'Cherry_(including_sour)___Powdery_mildew':
            st.subheader('WHAT IS CHERRY POWDERY MILDEW?')
            st.write('Cherry powdery mildew is a fungal disease that affects cherry trees, including sour cherry varieties. Powdery mildew is a common fungal infection characterized by a white, powdery growth on the surfaces of leaves, stems, and sometimes fruit. It is caused by various species of the fungus Podosphaera, with specific species affecting different plant hosts.'

'Cherry powdery mildew can weaken the tree, reduce fruit quality and yield, and in severe cases, even lead to defoliation. The fungus typically thrives in warm, dry conditions with high humidity, making it a common problem in many cherry-growing regions.'

'Management of cherry powdery mildew often involves cultural practices such as proper spacing of trees to improve air circulation, pruning to remove infected plant parts, and application of fungicides when necessary. Additionally, selecting resistant cherry varieties can help reduce the likelihood of powdery mildew infection.')
            st.subheader('FUNGICIDES TO PREVENT CHERRY POWDERY MILDEW:')
            st.write(
                '1.Sulfur-based fungicides  2.Copper-based fungicides  3.Triazole fungicides (e.g., tebuconazole)  4.Sterol biosynthesis inhibitors (e.g., myclobutanil)  5.Strobilurin fungicides (e.g., azoxystrobin)  6.Thiophanate-methyl  7.Potassium bicarbonate-based fungicides')
        elif prediction == 'Blueberry___healthy':
            st.subheader('HOW DO WE SAY THAT BLUEBERRRY SHRUB IS HEALTHY?')
            st.write('Blueberry shrubs are renowned not only for their delicious taste but also for their numerous health benefits. These vibrant blue gems are packed with antioxidants, vitamins, and minerals that contribute to overall well-being. When we say a blueberry shrub is healthy and free from disease, we are acknowledging its robust growth, lush foliage, and abundant fruiting without any signs of infection or pest infestation. A healthy blueberry shrub boasts vibrant leaves, sturdy stems, and plump berries, signaling optimal conditions and care. Additionally, disease-free status ensures that the shrub can thrive without the hindrance of common ailments such as fungal infections or bacterial diseases. With proper attention to soil quality, watering, and pruning, a disease-free blueberry shrub can continue to flourish, offering a bounty of nutritious berries for enjoyment and sustenance.')
            st.subheader('HOW TO MAINTAIN THE BLUEBERRRY SHRUB HEALTHY:')
            st.write(
                '1. Regular pruning  2. Adequate watering  3. Proper soil pH management  4. Mulching  5. Fertilization  6. Pest control  7. Disease monitoring  8. Sunlight exposure management  9. Proper spacing  10. Winter protection')
        elif prediction == 'Cherry_(including_sour)___healthy':
            st.subheader('HOW DO WE SAY THAT CHERRY TREE IS HEALTHY?')
            st.write('A healthy cherry tree is a testament to vitality and care. It stands tall, its branches adorned with lush foliage, promising a bountiful harvest of succulent fruit. A vibrant green canopy, free from withering leaves or discoloration, signifies robust growth and optimal photosynthesis. The bark, smooth and unblemished, protects the inner layers from harm while providing structural support. Strong, well-established roots anchor the tree securely in the earth, enabling it to withstand winds and adverse weather conditions. Vigorous blossoms in the spring transform into clusters of cherries, their rich hues a testament to the tree health and vigor. Disease-resistant and resilient to pests, a healthy cherry tree thrives in its environment, providing shade, beauty, and nourishment for generations to come.')
            st.subheader('HOW TO MAINTAIN THE CHERRY TREE HEALTHY:')
            st.write(
                '1. Regular Pruning  2. Adequate Watering  3. Proper Fertilization  4. Pest Control  5. Disease Prevention  6. Mulching  7. Sunlight Exposure  8. Soil pH Management  9. Air Circulation  10. Winter Protection')
        elif prediction == 'Cherry_(including_sour)___Powdery_mildew':
            st.subheader('WHAT IS CHERRY POWDERY MILDEW?')
            st.write(
                'Cherry powdery mildew is a fungal disease that affects cherry trees, including sour cherry varieties. Powdery mildew is a common fungal infection characterized by a white, powdery growth on the surfaces of leaves, stems, and sometimes fruit. It is caused by various species of the fungus Podosphaera, with specific species affecting different plant hosts.'

                'Cherry powdery mildew can weaken the tree, reduce fruit quality and yield, and in severe cases, even lead to defoliation. The fungus typically thrives in warm, dry conditions with high humidity, making it a common problem in many cherry-growing regions.'

                'Management of cherry powdery mildew often involves cultural practices such as proper spacing of trees to improve air circulation, pruning to remove infected plant parts, and application of fungicides when necessary. Additionally, selecting resistant cherry varieties can help reduce the likelihood of powdery mildew infection.')
            st.subheader('FUNGICIDES TO PREVENT CHERRY POWDERY MILDEW:')
            st.write(
                '1.Sulfur-based fungicides  2.Copper-based fungicides  3.Triazole fungicides (e.g., tebuconazole)  4.Sterol biosynthesis inhibitors')
        elif prediction == 'Corn_(maize)___Common_rust_':
            st.subheader('WHAT IS COMMON RUST IN CORN(MAIZE)?')
            st.write('Common rust, scientifically known as Puccinia sorghi, is a fungal disease that affects corn, commonly known as maize. This disease manifests as small, round, reddish-brown pustules on the leaves, stems, and ears of infected corn plants. These pustules contain spores that can spread the disease to other nearby plants. Common rust can weaken the affected corn plants, reducing yield and quality if left unmanaged. It thrives in warm, humid conditions and is often managed through cultural practices, fungicide applications, and planting resistant corn varieties.')
            st.subheader('FUNGICIDES TO PREVENT COMMON RUST IN CORN(MAIZE):')
            st.write(
                '1. Azoxystrobin  2. Flutriafol  3. Trifloxystrobin  4. Propiconazole  5. Thiophanate-methyl  6. Tebuconazole  7. Pyraclostrobin  8. Mancozeb  9. Copper-based fungicides  10. Chlorothalonil')
        elif prediction == 'Corn_(maize)___Northern_Leaf_Blight':
            st.subheader('WHAT IS NORTHERN LEAF BLIGHT IN CORN?')
            st.write(
                ' Northern Leaf Blight (NLB), caused by the fungus Exserohilum turcicum, is a significant foliar disease that affects corn (maize) plants. It primarily occurs in cooler climates and can lead to significant yield losses if left uncontrolled. NLB manifests as long, cigar-shaped lesions on corn leaves, initially gray-green and later turning tan or brown with irregular borders. Severe infections can lead to premature death of leaves, reducing the photosynthetic capacity of the plant and impacting yield. Management strategies include planting resistant varieties, crop rotation, fungicide applications, and maintaining optimal plant health through proper fertilization and irrigation.')
            st.subheader('FUNGICIDES TO PREVENT NORTHERN LEAF BLIGHT IN CORN:')
            st.write(
                '1. Azoxystrobin  2. Flutriafol  3. Trifloxystrobin  4. Propiconazole  5. Thiophanate-methyl  6. Tebuconazole  7. Pyraclostrobin  8. Mancozeb  9. Chlorothalonil  10. Boscalid')
        elif prediction == 'Corn_(maize)___healthy':
            st.subheader('HOW DO WE SAY THAT CORN PLANT IS HEALTHY?')
            st.write('Assessing the health of a corn (maize) plant involves observing various indicators that reflect its vigor and vitality. A healthy corn plant typically exhibits several characteristics. Firstly, its foliage appears vibrant and lush, with leaves that are a deep green color and free from discoloration or spots. The stalk of a healthy corn plant is sturdy and erect, providing adequate support for the growing ears. Additionally, the growth is robust and uniform, with each stalk displaying similar height and development. Healthy corn plants also demonstrate strong root systems, anchoring them securely in the soil and facilitating efficient nutrient uptake. Furthermore, the presence of well-formed ears, filled with plump and evenly spaced kernels, indicates reproductive health and successful pollination. Regular monitoring for signs of pests and diseases is crucial, as a healthy corn plant is typically resistant to common threats. Overall, by assessing factors such as leaf color, stalk strength, uniform growth, ear development, root health, and pest resistance, one can confidently determine the health of a corn plant and intervene promptly if any issues arise to ensure optimal growth and yield.')
            st.subheader('HOW TO MAINTAIN THE CORN PLANT HEALTHY:')
            st.write(
                '1.Crop rotation  2.Adequate watering  3.Fertilization management  4.Pest monitoring and control  5.Disease prevention measures 6.Weed management  7.Soil conservation practices  8.Timely planting and harvesting  9.Genetic selection of disease-resistant varieties  10.Proper spacing between plants')
        elif prediction == 'Grape___Black_rot':
            st.subheader('WHAT IS GRAPE BLACK ROT?')
            st.write('Black rot, caused by the fungus Guignardia bidwellii, is a common and devastating disease that affects grapevines. It is particularly problematic in warm and humid climates. Black rot can infect all green parts of the grapevine, including leaves, shoots, tendrils, and fruit.'

'Symptoms of black rot typically appear as small, brown lesions on leaves, which eventually enlarge and develop a characteristic black center surrounded by a reddish-brown margin. Infected fruit may also exhibit similar lesions, which can lead to fruit rot and shriveling. In severe cases, black rot can cause defoliation, reduced yield, and even death of the vine.'

'Management of black rot often involves a combination of cultural practices and fungicide applications. These practices may include pruning to improve air circulation, removing and destroying infected plant material, applying fungicides preventatively, and promoting vine health through proper irrigation and nutrient management. Additionally, planting resistant grapevine varieties can help mitigate the impact of black rot.')
            st.subheader('FUNGICIDES TO PREVENT GRAPE BLACK ROT:')
            st.write(
                '1.Captan  2.Mancozeb  3.Myclobutanil  4.Thiophanate-methyl  5.Azoxystrobin  6.Boscalid  7.Pyraclostrobin  8.Tebuconazole  9.Fludioxonil  10.Copper-based fungicides')
        elif prediction == 'Grape___Esca_(Black_Measles)':
            st.subheader('WHAT IS GRAPE BLACK MEASLES?')
            st.write(
                'Esca, also known as "Black Measles," is a destructive fungal disease that affects grapevines. It is caused by a complex of fungi, including Phaeomoniella chlamydospora and Phaeoacremonium spp., among others. Esca can affect all grapevine tissues, including leaves, shoots, and fruit, and is particularly problematic in older vineyards.'

'Symptoms of Esca typically appear as foliar symptoms such as tiger-striped discoloration, known as "tiger-stripe leaf symptom," or the "tache noir" symptom, which includes dark brown to black spots on leaves. The disease can also cause wilting and necrosis of shoots and cankers on the woody parts of the vine. In severe cases, Esca can lead to vine decline and death.'

'Management of Esca is challenging and often involves cultural practices such as pruning to remove infected wood, reducing vine stress through proper irrigation and nutrition, and planting disease-resistant grapevine varieties. However, there are no curative treatments available for Esca once symptoms are visible. Some preventative measures include trunk injections or fungicide applications, although their efficacy can vary.'

'Overall, managing Esca requires an integrated approach that focuses on preventing infection and minimizing vine stress to prolong the lifespan and productivity of affected grapevines.')
            st.subheader('FUNGICIDES TO PREVENT GRAPE BLACK MEASLES:')
            st.write(
                '1.Thiophanate-methyl  2.Boscalid  3.Fluopyram  4.Tebuconazole  5.Myclobutanil  6.Propiconazole  7.Azoxystrobin  8.Trifloxystrobin  9.Fludioxonil  10.Copper-based fungicides')
        elif prediction == 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
            st.subheader('WHAT IS GRAPE LEAF BLIGHT?')
            st.write('Grape leaf blight, also known as Isariopsis leaf spot, is a fungal disease caused by the pathogen Isariopsis (=Botrytis) leaf spot (Isariopsis (=Botryotinia) viticola). It affects grapevines and can lead to significant damage, particularly in warm and humid growing conditions.'

'Symptoms of grape leaf blight typically appear as small, circular to irregularly shaped lesions on grape leaves. These lesions may start as yellow or brown spots and can eventually enlarge and develop a tan to grayish center with a reddish-brown margin. Severe infections can cause defoliation, weakening the vine and reducing fruit quality and yield.'

'Management of grape leaf blight often involves a combination of cultural practices and fungicide applications. Cultural practices may include pruning to improve air circulation, removing and destroying infected plant material, and promoting vine health through proper irrigation and nutrition. Fungicides may be applied preventatively or curatively, depending on the severity of the disease and local conditions. Commonly used fungicides for grape leaf blight management include those containing active ingredients such as captan, mancozeb, azoxystrobin, and fludioxonil, among others.'

'As with any disease management strategy, essential to monitor vineyards regularly, practice integrated pest management, and adhere to recommended application rates and timings for fungicide treatments. Additionally, selecting disease-resistant grape varieties can help reduce the impact of grape leaf blight in vineyards.')
            st.subheader('FUNGICIDES TO PREVENT GRAPE LEAF BLIGHT:')
            st.write(
                '1.Captan  2.Mancozeb  3.Chlorothalonil  4.Azoxystrobin  5.Fludioxonil  6.Thiophanate-methyl  7.Boscalid  8.Pyraclostrobin  9.Trifloxystrobin  10.Propiconazole')
        elif prediction == 'Grape___healthy':
            st.subheader('HOW DO WE SAY THAT GRAPE PLANT IS HEALTHY?')
            st.write('Grapes are unique among fruit-bearing plants, as they grow on woody perennial vines rather than trees, shrubs, or traditional herbaceous plants. These vines, known as grapevines, can be trained onto trellises or other support structures to manage their growth and optimize fruit production. While they have a somewhat shrub-like appearance due to their branching structure close to the ground, they are not classified as shrubs. Instead, they are best described as climbing vines or vine plants. Grapes belong to the Vitaceae family and are widely cultivated for their fruit, which is used for making wine, juice, raisins, and fresh consumption. Their ability to climb and intertwine with their support structures allows them to reach for sunlight and maximize photosynthesis, contributing to the abundant growth of their luscious fruit.')
            st.subheader('HOW TO MAINTAIN THE GRAPE PLANT HEALTHY:')
            st.write(
                '1. Pruning  2. Adequate Watering  3. Fertilization Management  4. Pest Monitoring and Control  5. Disease Prevention Measures  6. Canopy Management  7. Weed Control  8. Soil Management 9. Sunlight Exposure  10. Integrated Pest Management')
        elif prediction == 'Orange___Haunglongbing_(Citrus_greening)':
            st.subheader('WHAT IS ORANGE HUANGLONGBING?')
            st.write(
               'Huanglongbing(HLB), commonly known as citrus greening disease, is a devastating bacterial disease that affects citrus trees, including oranges. It is caused by the bacterium Candidatus Liberibacter asiaticus and is primarily transmitted by the Asian citrus psyllid, a tiny insect that feeds on citrus trees.'

'Symptoms of citrus greening disease vary but often include yellowing of leaves, mottled or blotchy discoloration, and asymmetrical blotching of leaves. Infected trees may also produce small, lopsided, and bitter-tasting fruit that fails to ripen properly. Additionally, HLB can cause stunted growth, premature fruit drop, and eventual decline and death of infected trees.'

'Management of citrus greening disease is challenging and typically involves a combination of cultural, biological, and chemical methods. These may include controlling psyllid populations through insecticides, removing and destroying infected trees to prevent further spread, planting disease-resistant varieties, and promoting overall tree health through proper irrigation, fertilization, and pruning practices. Additionally, ongoing research into resistant varieties and disease management strategies is critical for combating citrus greening and preserving the citrus industry.')
            st.subheader('HOW TO PREVENT ORANGE HUANGLONGBING:')
            st.write(
                '1. Psyllid Control  2. Disease-free Planting Material  3. Monitoring and Early Detection  4. Tree Removal and Destruction of Infected Trees  5. Use of Disease-resistant Varieties  6. Pruning Practices  7. Soil and Nutrient Management  8. Biological Control Agents  9. Integrated Pest Management  10. Public Awareness and Education')
        elif prediction == 'Peach___Bacterial_spot':
            st.subheader('WHAT IS PEACH BACTERIAL SPOT?')
            st.write(
                 'Peach bacterial spot is a common and destructive disease caused by the bacterium Xanthomonas arboricola pv. pruni. It affects peach trees and other stone fruit trees, such as nectarines and plums. Bacterial spot can cause significant damage to leaves, fruit, and shoots, leading to reduced yield and fruit quality.'

'Symptoms of peach bacterial spot typically appear as small, water-soaked lesions on leaves, which later turn brown or black and develop a shot-hole appearance. Infected fruit may exhibit dark, sunken lesions with raised margins, rendering them unmarketable. In severe cases, bacterial spot can cause defoliation, shoot dieback, and fruit rot.'

'Management of peach bacterial spot often involves a combination of cultural practices and chemical control methods. These may include pruning to improve air circulation, avoiding overhead irrigation, removing and destroying infected plant material, and applying copper-based fungicides or bactericides preventatively or curatively. Additionally, planting disease-resistant peach varieties can help reduce the impact of bacterial spot in orchards.')
            st.subheader('BACTERICIDES TO PREVENT PEACH BACTERIAL SPOT:')
            st.write(
                '1.Copper-based products (e.g., copper hydroxide, copper sulfate)  2.Streptomycin sulfate  3.Oxytetracycline hydrochloride')
        elif prediction == 'Pepper,_bell___Bacterial_spot':
            st.subheader('WHAT IS BELL PEPPER BACTERIAL SPOT?')
            st.write('Bell pepper Bacterial spot is a common and potentially devastating disease affecting bell peppers (Capsicum annuum) and other pepper varieties. It is caused by the bacterium Xanthomonas campestris pv. vesicatoria. This disease manifests as dark, water-soaked lesions on leaves, stems, and fruit, which later develop into small, raised spots with a necrotic center. Infected leaves may eventually yellow and drop prematurely, leading to defoliation and reduced yield. Bacterial spot can spread rapidly under warm, humid conditions, especially during periods of rainfall or overhead irrigation. Management strategies for bacterial spot in bell peppers typically involve planting disease-resistant varieties, practicing crop rotation, using pathogen-free seeds and transplants, implementing proper sanitation measures, and applying copper-based bactericides or other approved bactericides preventatively. Regular monitoring and early intervention are essential to minimize the impact of bacterial spot and maintain pepper crop health.')
            st.subheader('BACTERICIDES TO PREVENT BELL PEPPER BACTERIAL SPOT:')
            st.write(
                '1.Copper-based products (e.g., copper hydroxide, copper sulfate)  2.Streptomycin sulfate  3.Oxytetracycline hydrochloride')
        elif prediction == 'Peach___healthy':
            st.subheader('HOW DO WE SAY THAT PEACH TREE IS HEALTHY?')
            st.write('A healthy peach tree exhibits several key characteristics that indicate its vitality and well-being. Firstly, its foliage appears vibrant and lush, with leaves that are a rich green color and free from discoloration, spots, or signs of wilting. The branches of a healthy peach tree are sturdy and well-formed, without any signs of damage or disease. Additionally, the tree shows vigorous growth, with new shoots and branches emerging regularly and reaching outwards in a balanced manner. A healthy peach tree also produces an abundance of blossoms in the spring, followed by well-formed fruit that matures to the appropriate size and color. Furthermore, the root system is robust, anchoring it securely in the soil and facilitating efficient uptake of nutrients and water. Regular monitoring for signs of pests and diseases is crucial, as a healthy peach tree is typically more resistant to common threats. Overall, by assessing factors such as leaf color, branch structure, growth pattern, fruit development, and pest resistance, one can confidently determine the health of a peach tree and intervene promptly if any issues arise to ensure its continued vigor and productivity.')
            st.subheader('HOW TO MAINTAIN THE PEACH TREE HEALTHY:')
            st.write(
                '1. Pruning  2. Adequate Watering  3. Fertilization Management  4. Pest Monitoring and Control  5. Disease Prevention Measures  6. Weed Control  7. Sunlight Exposure  8. Soil Management  9. Mulching  10. Training and Support Structures')
        elif prediction == 'Pepper,_bell___healthy':
             st.subheader('HOW DO WE SAY THAT BELL PEPPER PLANT IS HEALTHY?')
             st.write('A healthy bell pepper plant displays vibrant green foliage, free from discoloration or wilting. Its stems stand tall and firm, supporting an abundant array of leaves and burgeoning fruit. Upon closer inspection, one may observe a robust root system, firmly anchoring the plant in the soil. The leaves themselves are lush and glossy, indicative of ample moisture and nutrient uptake. Furthermore, the plant exhibits vigorous growth, with new shoots and flowers appearing regularly. Overall, the vitality of the bell pepper plant is evident in its flourishing appearance, promising a bountiful harvest to come.')
             st.subheader('HOW TO MAINTAIN THE BELL PEPPER PLANT HEALTHY:')
             st.write(
                 '1. Regular watering  2. proper sunlight exposure  3. nutrient-rich soil  4. pruning  5. pest control  6.adequate spacing')
        elif prediction == 'Potato___Early_blight':
            st.subheader('WHAT IS POTATO EARLY BLIGHT?')
            st.write(
                'Potato early blight is a fungal disease caused by Alternaria solani, affecting potato plants worldwide. It typically manifests as dark lesions on the leaves, starting as small spots and expanding into larger, irregular shapes. These lesions can cause defoliation, reducing the ability to photosynthesize and impacting yield. Early blight thrives in warm, humid conditions, spreading through spores carried by wind or water. Management strategies include crop rotation, fungicide application, and selecting resistant potato varieties. However, despite these efforts, early blight remains a significant challenge for potato growers, necessitating vigilant monitoring and proactive measures to minimize its impact on crop production.')
            st.subheader('BACTERICIDES TO PREVENT POTATO EARLY BLIGHT:')
            st.write(
                '1.Copper-based products (e.g., copper hydroxide, copper sulfate)  2.Streptomycin sulfate  3.Mancozeb')
        elif prediction == 'Potato___Late_blight':
            st.subheader('WHAT IS POTATO LATE BLIGHT?')
            st.write(
                'Bell potato late blight, caused by the oomycete pathogen Phytophthora infestans, is a devastating disease affecting potato plants. It is characterized by rapidly spreading lesions on leaves, stems, and tubers, often leading to complete crop loss if left unchecked. This disease thrives in cool, moist conditions, spreading through airborne spores and can cause significant economic losses in potato farming. Historical significance includes the Irish Potato Famine in the mid-19th century. Management strategies typically involve cultural practices, fungicide application, and planting resistant potato varieties.')
            st.subheader('BACTERICIDES TO PREVENT POTATO LATE BLIGHT:')
            st.write(
                '1.Copper-based products (e.g., copper hydroxide, copper sulfate)  2.Streptomycin sulfate  3.Mancozeb')
        elif prediction == 'Potato___healthy':
            st.subheader('HOW DO WE SAY THAT POTATO PLANT IS HEALTHY?')
            st.write(
                'A healthy potato plant exhibits several key characteristics that indicate its vigor and vitality. Firstly, its foliage appears lush and vibrant, with deep green leaves that are free from discoloration, wilting, or abnormal spots. The stems should be sturdy and upright, supporting the weight of the foliage without bending or breaking easily. Additionally, a healthy potato plant displays vigorous growth, with new shoots emerging regularly and steadily increasing in size. The roots of a healthy potato plant are well-developed and firm, anchoring the plant securely in the soil and facilitating efficient nutrient uptake. Furthermore, a healthy potato plant is typically resistant to common pests and diseases, showing no signs of infestation or infection. Overall, the presence of robust foliage, vigorous growth, strong stems, healthy roots, and resistance to pests and diseases are indicators of a thriving potato plant. Regular monitoring and proper care, including adequate watering, fertilization, and disease management, are essential to maintain the health and productivity of potato plants throughout the growing season.')
            st.subheader('HOW TO MAINTAIN THE POTATO PLANT HEALTHY:')
            st.write(
                '1. Proper soil preparation and drainage 2. Adequate watering 3. Regular fertilization4. Crop rotation 5. Weed control 6. Disease and pest management 7. Timely harvesting')
        elif prediction == 'Squash___Powdery_mildew':
            st.subheader('WHAT IS SQUASH POWDERY MILDEW?')
            st.write(
                'Squash powdery mildew is a fungal disease that affects squash plants, including various types such as zucchini, pumpkins, and butternut squash. It is caused by the fungus Podosphaera xanthii (formerly known as Sphaerotheca fuliginea). This disease manifests as white powdery patches on the leaves, stems, and sometimes fruit of infected plants. These patches can gradually spread and coalesce, covering large areas of the plant and impairing photosynthesis, which can reduce yield and plant vigor. Squash powdery mildew thrives in warm, humid conditions and can spread rapidly, especially in crowded plantings or where air circulation is poor. Management strategies include selecting resistant varieties, practicing crop rotation, applying fungicides, and maintaining proper spacing between plants to improve air circulation.')
            st.subheader('BACTERICIDES TO PREVENT SQUASH POWDERY MILDEW:')
            st.write(
                '1.Bacillus subtilis-based products 2.Copper-based products (e.g., copper hydroxide, copper sulfate)  3.Streptomycin sulfate ')
        elif prediction == 'Raspberry___healthy':
            st.subheader('HOW DO WE SAY THAT RASPBERRY PLANT IS HEALTHY?')
            st.write(
                'A healthy raspberry plant is easily identified by several key indicators. Firstly, its growth is robust and vigorous, with sturdy canes standing upright and producing abundant foliage. The leaves of a healthy raspberry plant are a rich, deep green, reflecting optimal chlorophyll levels and strong photosynthetic activity. Importantly, there are no signs of disease present on the foliage or stems; the plant appears clean and free from common ailments like powdery mildew or leaf spot. Additionally, a healthy raspberry plant bears plentiful fruit that is plump, juicy, and flavorful, indicating its ability to support reproductive growth. Underneath the soil, the plant boasts a well-developed root system, anchoring it firmly and facilitating efficient nutrient uptake. Furthermore, a healthy raspberry plant demonstrates resilience to environmental stresses, showing minimal wilting or damage even under adverse conditions such as drought or extreme temperatures. Overall, by observing these signs of robust growth, disease resistance, abundant fruit production, and stress tolerance, growers can confidently determine the health and vitality of their raspberry plants. Regular monitoring and appropriate care practices ensure the continued well-being and productivity of these valuable fruiting specimens.')
            st.subheader('HOW TO MAINTAIN THE RASPBERRY PLANT HEALTHY:')
            st.write(
                '1. Proper irrigation 2. Regular fertilization 3. Pruning and training 4. Disease and pest management 5. Weed control 6. Mulching 7. Adequate spacing and air circulation 8. Winter protection')
        elif prediction == 'Soybean___healthy':
            st.subheader('HOW DO WE SAY THAT SOYABEAN PLANT IS HEALTHY?')
            st.write('A healthy soybean plant can be recognized by several key characteristics. Firstly, its foliage appears lush and vibrant, with deep green leaves that are free from discoloration, wilting, or abnormal spots. The stems should be sturdy and upright, supporting the weight of the foliage without bending or breaking easily. Additionally, a healthy soybean plant displays vigorous growth, with new shoots emerging regularly and steadily increasing in size. The roots of a healthy soybean plant are well-developed and firm, anchoring the plant securely in the soil and facilitating efficient nutrient uptake. Furthermore, a healthy soybean plant is typically resistant to common pests and diseases, showing no signs of infestation or infection. Overall, the presence of robust foliage, vigorous growth, strong stems, healthy roots, and resistance to pests and diseases are indicators of a thriving soybean plant. Regular monitoring and proper care, including adequate watering, fertilization, and disease management, are essential to maintain the health and productivity of soybean plants throughout the growing season.')
            st.subheader('HOW TO MAINTAIN THE SOYABEAN PLANT HEALTHY:')
            st.write(
                '1. Crop rotation 2. Proper soil preparation 3. Adequate watering 4. Balanced fertilization 5. Weed control 6. Disease monitoring and management 7. Pest monitoring and management 8. Timely harvesting')
        elif prediction == 'Strawberry___Leaf_scorch':
            st.subheader('WHAT IS STRAWBERRY LEAF SCORCH?')
            st.write(
                'Strawberry leaf scorch is a bacterial disease caused by the bacterium Xylella fastidiosa. It primarily affects strawberry plants, causing symptoms such as yellowing and browning of leaf margins, leaf wilting, and necrosis. The disease is transmitted by sharpshooter insects, such as leafhoppers and spittlebugs, which feed on the  sap, carrying the bacteria from infected to healthy plants. Strawberry leaf scorch can lead to reduced yield and plant vigor, and severe infections may result in plant death. Management strategies include planting disease-resistant varieties, controlling insect vectors, and removing infected plants to prevent the spread of the disease.')
            st.subheader('BACTERICIDES TO PREVENT STRAWBERRY LEAF SCORCH:')
            st.write(
                '1.Oxytetracycline, another antibiotic sometimes used for disease control in strawberries. 2.Copper-based products (e.g., copper hydroxide, copper sulfate)  3.Streptomycin sulfate ')
        elif prediction == 'Strawberry___healthy':
            st.subheader('HOW DO WE SAY THAT STRAWBERRY PLANT IS HEALTHY?')
            st.write('A healthy strawberry plant displays several key characteristics that indicate its well-being. Firstly, its foliage appears vibrant and lush, with deep green leaves that are free from discoloration, wilting, or abnormal spots. The stems should be sturdy and erect, supporting the weight of the foliage without bending or breaking easily. Additionally, a healthy strawberry plant exhibits vigorous growth, with runners extending and new leaves emerging regularly. The roots of a healthy strawberry plant are firm and well-developed, anchoring the plant securely in the soil and facilitating efficient nutrient uptake. Furthermore, a healthy strawberry plant is typically resistant to common pests and diseases, showing no signs of infestation or infection. Overall, the presence of robust foliage, vigorous growth, strong stems, healthy roots, and resistance to pests and diseases are indicators of a thriving strawberry plant. Regular monitoring and proper care, including adequate watering, fertilization, and disease management, are essential to maintain the health and productivity of strawberry plants throughout the growing season.')
            st.subheader('HOW TO MAINTAIN THE STRAWBERRY PLANT HEALTHY:')
            st.write(
                '1. Proper site selection 2. Adequate sunlight 3. Well-draining soil 4. Regular watering 5. Mulching 6. Fertilization 7. Weed control 8. Disease monitoring and management 9. Pest monitoring and management 10. Proper pruning 11. Adequate spacing between plants 12. Removal of runners if necessary 13. Timely harvesting')
        elif prediction == 'Tomato___Bacterial_spot':
            st.subheader('WHAT IS TOMATO BACTERIAL SPOT?')
            st.write('Tomato bacterial spot is a plant disease caused by the bacterium Xanthomonas campestris pv. vesicatoria. It primarily affects tomatoes but can also infect other members of the Solanaceae family, such as peppers and eggplants. The disease manifests as dark, water-soaked lesions on the leaves, stems, and fruit of affected plants. These lesions may have a raised, scabby appearance and can eventually lead to defoliation, fruit rot, and yield loss. Tomato bacterial spot is favored by warm, humid conditions and can spread rapidly through splashing water, contaminated tools, or infected plant debris. Management strategies include planting disease-resistant varieties, practicing crop rotation, using pathogen-free seeds and transplants, applying copper-based bactericides, and maintaining proper sanitation practices.')
            st.subheader('BACTERICIDES TO PREVENT TOMATO BACTERIAL SPOT:')
            st.write(
                '1.Copper-based bactericides, such as copper sulfate or copper hydroxide. 2.Streptomycin, an antibiotic. 3.Oxytetracycline, another antibiotic sometimes used for disease control in tomatoes. 4.Fixed copper formulations.')
        elif prediction == 'Tomato___Early_blight':
            st.subheader('WHAT IS TOMATO EARLY BLIGHT?')
            st.write(
                'Tomato early blight is a fungal disease caused by the fungus Alternaria solani. It affects tomato plants, typically appearing as dark brown or black lesions with concentric rings on the lower leaves of the plant. These lesions can expand and cause defoliation, reducing the  ability to photosynthesize and impacting yield. Tomato early blight is favored by warm, humid conditions and can spread rapidly through splashing water, contaminated tools, or infected plant debris. Management strategies include planting disease-resistant varieties, practicing crop rotation, using fungicides, maintaining proper spacing between plants for good air circulation, and removing infected plant debris to reduce the spread.')
            st.subheader('BACTERICIDES TO PREVENT TOMATO EARLY BLIGHT:')
            st.write(
                '1.Copper-based bactericides, such as copper sulfate or copper hydroxide. 2.Streptomycin, an antibiotic. 3.Oxytetracycline, an')
        elif prediction == 'Tomato___Late_blight':
            st.subheader('WHAT IS TOMATO LATE BLIGHT?')
            st.write('Tomato late blight, caused by the oomycete pathogen Phytophthora infestans, is a devastating disease affecting tomato plants. It is characterized by dark, water-soaked lesions on the leaves, stems, and fruit, often accompanied by a white, fuzzy growth on the undersides of the leaves. These lesions rapidly expand, leading to tissue collapse and plant death if left untreated. Tomato late blight thrives in cool, wet conditions and can spread rapidly, especially during periods of high humidity. Management strategies include planting disease-resistant varieties, applying fungicides preventively, maintaining proper spacing between plants for good air circulation, and removing infected plant debris to reduce the  spread.')
            st.subheader('BACTERICIDES TO PREVENT TOMATO LATE BLIGHT:')
            st.write('1.Copper-based bactericides, such as copper sulfate or copper hydroxide. 2.Streptomycin, an antibiotic. 3.Fixed copper formulations.')
        elif prediction == 'Tomato___Leaf_Mold':
            st.subheader('WHAT IS TOMATO LEAF MOLD?')
            st.write('Tomato leaf mold, caused by the fungus Fulvia fulva (formerly known as Cladosporium fulvum), is a common fungal disease affecting tomato plants. It manifests as fuzzy, olive-green to yellowish patches on the upper surface of the leaves, often accompanied by a velvety grayish mold growth on the lower leaf surface. Tomato leaf mold thrives in warm, humid conditions and can spread rapidly, particularly in greenhouses or areas with poor air circulation. Infected leaves may become distorted, yellow, or necrotic, ultimately leading to defoliation if left untreated. Management strategies include selecting resistant tomato varieties, providing proper ventilation to reduce humidity, spacing plants adequately for air circulation, avoiding overhead watering, and applying fungicides preventively in severe cases.')
            st.subheader('FUNGICIDES TO PREVENT TOMATO LEAF MOLD:')
            st.write('1.Chlorothalonil 2.Mancozeb 3.Copper-based fungicides (although primarily effective against bacterial diseases, they may have some activity against certain fungal pathogens)')
        elif prediction == 'Tomato___Septoria_leaf_spot':
            st.subheader('WHAT IS TOMATO SEPTORIA LEAF SPOT?')
            st.write(
                 'Tomato septoria leaf spot is a fungal disease caused by the fungus Septoria lycopersici. It is one of the most common and destructive diseases affecting tomato plants. Tomato septoria leaf spot typically appears as small, circular lesions with dark brown centers and lighter-colored borders on the lower leaves of the plant. As the disease progresses, the lesions may enlarge and coalesce, leading to extensive leaf damage and defoliation. Tomato septoria leaf spot thrives in warm, humid conditions and can spread rapidly, particularly when foliage remains wet for extended periods. Management strategies include planting disease-resistant varieties, practicing crop rotation, maintaining proper spacing between plants for good air circulation, removing infected plant debris, and applying fungicides preventively in severe cases.')
            st.subheader('FUNGICIDES TO PREVENT TOMATO SEPTORIA LEAF SPOT:')
            st.write('1.Chlorothalonil 2.Mancozeb 3.Azoxystrobin 4.Boscalid 5.Copper-based fungicides (although primarily effective against bacterial diseases, they may have some activity against certain fungal pathogens')
        elif prediction == 'Tomato___Spider_mites Two-spotted_spider_mite':
            st.subheader('WHAT IS TOMATO TWO-SPOTTED SPIDER MITE?')
            st.write('The two-spotted spider mite (Tetranychus urticae) is a common pest that infests various plants, including tomatoes. Despite its name, this pest is not an insect but rather a tiny arachnid related to spiders. Two-spotted spider mites are very small, usually less than 1/50th of an inch, making them difficult to see with the naked eye. They have a characteristic two dark spots on their backs.'

'These mites feed on the undersides of tomato leaves by piercing plant cells and sucking out the juices, causing stippling or yellowing of the leaves. As infestations progress, leaves may become discolored, dry, and eventually drop prematurely, leading to reduced plant vigor and yield.'

'Two-spotted spider mites thrive in hot, dry conditions and can reproduce rapidly under these conditions, making them particularly troublesome in greenhouses or during periods of drought. Management strategies include increasing humidity, using biological control agents such as predatory mites or insects, applying insecticidal soaps or oils, and employing cultural practices like removing infested plant material.')
            st.subheader('MITICIDES TO PREVENT TOMATO TWO-SPOTTED SPIDER MITE:')
            st.write('1.Abamectin 2.Bifenazate 3.Hexythiazox 4.Spiromesifen 5.Fenazaquin')
        elif prediction == 'Tomato___Target_Spot':
            st.subheader('WHAT IS TOMATO TARGET SPOT?')
            st.write(
'Tomato target spot, also known as tomato target leaf spot or early blight, is a fungal disease caused by the fungus Corynespora cassiicola. It affects tomato plants, typically appearing as dark, concentric rings or target-like lesions on the leaves, stems, and fruit of infected plants. These lesions may start as small, dark spots that gradually expand in size and coalesce, leading to tissue necrosis and leaf yellowing. Target spot can cause defoliation, reducing the ability to photosynthesize and impacting yield. The disease is favored by warm, humid conditions and can spread rapidly, especially through splashing water or contaminated tools. Management strategies include planting disease-resistant varieties, practicing crop rotation, maintaining proper spacing between plants for good air circulation, removing infected plant debris, and applying fungicides preventively in severe cases.')
            st.subheader('FUNGICIDES TO PREVENT TOMATO TARGET SPOT:')
            st.write('1.Chlorothalonil 2.Mancozeb 3.Azoxystrobin 4.Boscalid 5.Copper-based fungicides ,although primarily effective against bacterial diseases, they may have some activity against certain fungal pathogens')
        elif prediction == 'Tomato___healthy':
            st.subheader('HOW DO WE SAY THAT TOMATO PLANT IS HEALTHY?')
            st.write('A healthy tomato plant exhibits several key characteristics that indicate its vitality and vigor. Firstly, its foliage appears lush and vibrant, with deep green leaves that are free from discoloration, wilting, or abnormal spots. The stems should be sturdy and upright, supporting the weight of the foliage without bending or breaking easily. Additionally, a healthy tomato plant displays vigorous growth, with strong stems and branches, and abundant foliage. The presence of flowers and developing fruit is another sign of a healthy plant, indicating its ability to support reproductive growth. Furthermore, a healthy tomato plant is typically resistant to common pests and diseases, showing no signs of infestation or infection. Regular monitoring and proper care, including adequate watering, fertilization, and disease management, are essential to maintain the health and productivity of tomato plants throughout the growing season. By observing these signs of robust growth, resilience, and reproductive capacity, growers can confidently determine the health and vitality of their tomato plants.')
            st.subheader('HOW TO MAINTAIN THE TOMATO PLANT HEALTHY:')
            st.write('1. Proper watering 2. Adequate sunlight 3. Regular fertilization 4. Disease monitoring and management 5. Pest monitoring and management 6. Proper spacing 7. Mulching 8. Proper pruning 9. Support structures 10. Crop rotation')
        elif prediction == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
            st.subheader('WHAT IS TOMATO YELLOW LEAF CURL VIRUS?')
            st.write('Tomato yellow leaf curl virus (TYLCV) is a highly destructive viral disease that affects tomato plants. It is transmitted primarily by the sweet potato whitefly (Bemisia tabaci) in a persistent, circulative manner, meaning the virus can persistently infect and be transmitted by the vector throughout its lifespan. TYLCV is characterized by symptoms such as yellowing and upward curling of the leaves, stunted growth, and reduced fruit production. Infected plants may exhibit leaf curling, chlorosis (yellowing), and a general decline in vigor. TYLCV infection can severely impact tomato production, leading to significant yield losses.'

'Management strategies for TYLCV include:'

'1. Planting virus-resistant tomato varieties.'
'2. Implementing rigorous control measures for the whitefly vector, such as insecticide application or biological control methods.'
'3. Removing and destroying infected plants to prevent further spread.'
'4. Implementing cultural practices to reduce whitefly populations, such as reflective mulches or companion planting with repellent plants.'
'5. Avoiding the introduction of infected plant material into new areas.'
'6. Using virus-free seedlings or certified virus-free planting material.'

'Preventing TYLCV is challenging due to the persistent nature of the vector and the lack of effective curative treatments once plants are infected. Therefore, a combination of preventative measures is typically recommended to manage and minimize the impact of this devastating viral disease on tomato crops.')
            st.subheader('INSECTICIDES TO PREVENT TOMATO YELLOW LEAF CURL VIRUS:')
            st.write('1.Neonicotinoids  2.Pyrethroids  3.Insect growth regulators 4.Spinosad 5.Horticultural oils')
        elif prediction == 'Tomato___Tomato_mosaic_virus':
            st.subheader('WHAT IS TOMATO MOSAIC VIRUS?')
            st.write('Tomato mosaic virus (ToMV) is a highly contagious viral disease that affects tomato plants worldwide. It belongs to the Tobamovirus genus and is transmitted primarily through mechanical means, such as contaminated tools, hands, or through sap contact during pruning or handling infected plants. ToMV infects all parts of the tomato plant, causing characteristic symptoms such as mosaic patterns of light and dark green on the leaves, leaf distortion, stunting, and reduced fruit yield and quality. The virus can persist in infected plant debris and soil for extended periods, making it difficult to eradicate once established in an area. Management strategies for Tomato mosaic virus include planting virus-free seedlings, practicing strict sanitation measures to prevent virus spread, controlling insect vectors, and removing infected plants promptly. Additionally, using resistant tomato varieties can help mitigate the impact of ToMV in tomato crops.')
            st.subheader('HOW TO PREVENT TOMATO MOSAIC VIRUS:')
            st.write('To prevent the spread of Tomato Mosaic Virus, focus on the following measures:'

'1.Planting virus-free seedlings or certified virus-free planting material.'
'2.Practicing strict sanitation measures to prevent virus spread, including disinfecting tools and equipment between uses and avoiding contact with infected plants.'
'3.Controlling insect vectors, such as aphids, which can transmit other viruses but not ToMV.'
'4.Removing and destroying infected plants promptly to prevent further spread of the virus.')