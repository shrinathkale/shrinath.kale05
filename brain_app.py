import tensorflow as tf
import streamlit as st

st.markdown("<h1 style='color: lightgreen;'>BRAIN TUMOR PREDICTION SYSTEM ON THE MRI SCAN</h1>", unsafe_allow_html=True)
st.write("Key Points for Using App are Given in the Sidebar")
img1_path = "image1.jpeg"
img2_path = "image2.jpeg"
st.image(img1_path)
st.sidebar.image(img2_path)

st.sidebar.markdown('''
            ##   **KEY POINTS**
            ###  * Upload the Photograph of MRI Scan of Brain and Press on the Show Image Button for getting Uploaded Image
            ###  * Upload the image in the JPG/JPEG/PNG Format
            ###  * Limit of the image uploading is 200MB
            ###  * Press on the Predict Button Which Shows the Graph of Brain Tumor Prediction Based on the MRI Scan
            ''')

model = tf.keras.models.load_model("trained_model_mri.keras")

def predict(img_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Define the class labels
    labels = ['Healthy Brain', 'Tumor Brain']
    
    plt.figure(figsize=(12, 12))
    plt.style.use('fivethirtyeight')
    
    # Load and preprocess the image
    img = Image.open(img_path)
    resized_img = img.resize((224, 224))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    # Make predictions
    predictions = model.predict(img)
    probs = list(predictions[0])
    
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    
    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt='%.2f')
    
    # Render the plot in Streamlit
    st.pyplot(plt)

image = st.file_uploader("Choose an image.....", type=["JPG","JPEG","PNG"])
if(st.button("Show image")):
    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.markdown("<medium style='color: orange;'>Image Not Uploaded</h1>", unsafe_allow_html=True)
if(st.button("Predict")):
    if image is not None:
        st.write("The Prediction is.....")
        predict(image)
    else:
        st.markdown("<medium style='color: orange;'>Image Not Uploaded</h1>", unsafe_allow_html=True)