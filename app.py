import streamlit as st 
from streamlit_player import st_player
import requests
import tempfile 
from PIL import Image
import numpy as np
import pandas as pd


@st.cache
def upload_model():
    pass 

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

elif direction == 'Try the model':
    
    title = st.text_input('YouTube URL', 'Insert URL here') 
    if st.button('Is there violence in the video?'):
        st_player(title)

    st.write("OR")

#########################################
#           Upload a video              #
######################################### 

    upload = st.empty()
    #start_button = st.empty()
    #stop_button = st.empty()

    with upload:
        f = st.file_uploader('Upload Video file (mpeg/mp4 format)')
        if f is not None:
            tfile  = tempfile.NamedTemporaryFile(delete = True)
            tfile.write(f.read())



    