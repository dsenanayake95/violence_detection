import streamlit as st 
from streamlit_player import st_player
import requests
import tempfile 
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf 
from violence_detection.cropper import capture_frames


def hide_streamlit_widgets():
    """
    hides widgets that are displayed by streamlit when running
    """
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.sidebar.markdown(f"""
    # Navigation menu
    """)

direction = st.sidebar.radio('Select a page', ('About the project', 'Meet the team', 'Try the model'))

#########################################
# Title and introduction to the project #
######################################### 

if direction == 'About the project':
    st.markdown("""# Violence Detection
## Can we detect violence in video?
""")
    
    col1,col2,col3,col4 = st.columns(4)
    
    non_violent1 = Image.open('/Users/dehajasenanayake/code/violence_detection/raw_data/frames/non_violence/NV_21.mp4_frame3.jpg')
    col1.image(non_violent1, caption='Non violent', use_column_width=True)
    
    non_violent2 = Image.open('/Users/dehajasenanayake/code/violence_detection/raw_data/frames/non_violence/NV_145.mp4_frame2.jpg')
    col2.image(non_violent2, caption='Non violent', use_column_width=True)
    
    non_violent3 = Image.open('/Users/dehajasenanayake/code/violence_detection/raw_data/frames/non_violence/NV_207.mp4_frame2.jpg')
    col3.image(non_violent3, caption='Non violent', use_column_width=True)
    
    violent1 = Image.open('/Users/dehajasenanayake/code/violence_detection/raw_data/frames/violence/V_9.mp4_frame4.jpg')
    col4.image(violent1, caption='Violent', use_column_width=True)
    


    if st.button('The Problem?'):
        print('button clicked!')
        st.write('Currently, the most common way to identify violent behaviour in video \
                 is using "human monitors". The extended exposure to violence in videos \
                     can cause harm to the mental health of these individuals. In addition, \
                         monitors may not be able to identify violence as it is happening \
                             meaning fewer opportunities to intervene.')
        
    if st.button('The Solution?'):
        print('button clicked!')
        st.write('We use transfer-learning and a CNN-RNN model to identify violent \
            behaviour in videos. Our output is the probability of violent behaviour throughout \
                the video. This approach means a reduction in the need for human monitors \
                    meaning a reduction in the negative impact on their mental health and \
                        potentially the earlier identification of intervention.')
    
#########################################
#           Meet the team               #
######################################### 

elif direction == 'Meet the team':
    col1,col2,col3 = st.columns(3)
    
    col1.subheader("Gift Opar")
    gift_photo = Image.open('/Users/dehajasenanayake/Documents/BREAD/recipe+for+monster+eye+halloween+cupcakes.jpeg')
    col1.image(gift_photo, use_column_width=True)
    col1.write("Insert text here")

    col2.subheader("Lukas (Tu) Pham")
    lukas_photo = Image.open('/Users/dehajasenanayake/Documents/BREAD/recipe+for+monster+eye+halloween+cupcakes.jpeg')
    col2.image(lukas_photo, use_column_width=True)
    col2.write("Insert text here")
    
    col3.subheader("Dehaja Senanayake")
    dehaja_photo = Image.open('/Users/dehajasenanayake/Documents/BREAD/recipe+for+monster+eye+halloween+cupcakes.jpeg')
    col3.image(dehaja_photo, use_column_width=True)
    col3.write("Dehaja is studying for a Masters in Environmental Technology.")
    
    
#########################################
#           Try the model               #
######################################### 




#########################################
#           Upload a video              #
######################################### 

elif direction == 'Try the model':
# save model - tf.keras.models.save_model(model, 'MY_MODEL')
    @st.cache
    def load_model():
        model = tf.keras.models.load_model('PATH_FOR_MY_MODEL')
        return model 
    model = load_model()
    st.write('model has been loaded')
    
    def upload_video():
        upload = st.empty()

        with upload:
            video = st.file_uploader('Upload Video file (mpeg/mp4 format)')
            if video is not None:
                tfile  = tempfile.NamedTemporaryFile(delete = True)
                tfile.write(video.read())
        return video 
    
    def predict_on_uploaded_video():
        video = upload_video()
        cap = cv2.VideoCapture(video)
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # I think this is where we would predict?
                
                cv2.imshow('frame', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()
                
                
    def cropped_frames(): # use CV2 and pillow (loops to the end of the video)
        video = upload_video()
        rectangles = capture_frames(video)
        return rectangles 
    
    def predict():
        for frame in cropped_frames:
            prediction = model.predict(frame)
            return frame
        
        

          
###
###Code to play a YouTube video
###
    #title = st.text_input('YouTube URL', 'Insert URL here') 
    #if st.button('Is there violence in the video?'):
        #st_player(title)

    
#workflow
# upload video 
# cropper creates frames
# run prediction on each cropped images
# 
# return video with bounding boxes and probabilities 

#webrtc - output videos on streamlit 

