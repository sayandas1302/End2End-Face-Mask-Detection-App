import streamlit as st
import tensorflow as tf
import cv2 as cv
import numpy as np

# calling the pretrained model
model = tf.keras.models.load_model('./artifacts/model.h5')

# function to check if whether a face is masked or not
def predict_masked_face(img_path):
    img_arr = cv.imread(img_path)
    img_resized = cv.resize(img_arr, (224, 224))
    img_scalled = img_resized/255
    y_pred = np.argmax(model.predict(img_scalled.reshape(1, 224, 224, 3)))
    return y_pred 

# prediction message dictionary
message = {
    0:'Mask Detected',
    1:'Mask not Detected'
}

# app layout
#===========

# background 
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://sytoss-live-10d81101576e4742896edfa4fb-3e648de.aldryn-media.com/filer_public/2a/f6/2af65da1-f6b1-4fa5-93ee-283bf5e7d285/face_mask_detection-min.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
     
# title
st.markdown(f'''
# Face Mask Detector App
This app detects wheather a face is masked or not, from the image of the face
''')

# image uploader
st.markdown(f'''
#### Enter the image of your face''')
uploaded_image = st.file_uploader('', type=['png', 'jpg', 'jpeg', 'webp'])

# button
col1, col2, col3, col4, col5 = st.columns(5)

with col3:
   check_button =  st.button('Result')

# app functionality
#==================

if check_button:
    # saving the uploaded image
    with open('./temp/uploadedImage.jpg', 'wb') as file:
        file.write(uploaded_image.getbuffer())

    # prediction
    img_path =  './temp/uploadedImage.jpg'
    result = predict_masked_face(img_path)
    
    # showing the result 
    result_container = st.container()

    with result_container:
        # place to show the results of the prediction regarding the face mask detection
        col1, col2 = st.columns([2,3])
        
        with col1:
            # functionality to show the inserted image 
            img_to_show = cv.resize(cv.imread('./temp/uploadedImage.jpg'), (200,200))
            st.image(img_to_show, caption='Image Inserted', channels='BGR')

        with col2:
            # functionality to show the prediction result
            st.markdown(f''' 
            ## {message[result]}''')