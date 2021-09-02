import streamlit as st 
from streamlit_player import st_player
import requests


import numpy as np
import pandas as pd


st.sidebar.markdown(f"""
    # Navigation menu
    """)

direction = st.radio('Select a page', ('About the project', 'Meet the team', 'Try the model'))

st.write(direction)

#########################################
# Title and introduction to the project #
######################################### 

if direction == 'About the project':
    st.markdown("""# Violence Detection
## Detecting violence in uploaded videos

Check it out yourself below. 
""")

    if st.button('About the project'):
        print('button clicked!')
        st.write('This project has looked at identifing the probability of violent behaviour \
             in a video, in real time. Currently, the most common way to identify this behavour \
                 is using "human monitors". The extended exposure to violence in videos \
                     can cause harm to the mental health of these individuals.')
    else:
        st.write('Why is detecting violence important?')
    
#########################################
#           Meet the team               #
######################################### 

elif direction == 'Meet the team':
    col = st.columns(3)
    
    col[0].header("Gift Opar")
    col[1].header("Lukas (Tu) Pham")
    col[2].header("Dehaja Senanayake")
    

#else: direction == 'Try the model':
    
    #title = st.text_input('Youtube URL', 'Insert URL here')
    
    #st_player(title)
    
#########################################
#     Insert the URL onto the page      #
######################################### 



#########################################
#           Upload a video              #
######################################### 

#upload = st.empty()
#start_button = st.empty()
#stop_button = st.empty()

#with upload:
    #f = st.file_uploader('Upload Video file (mpeg/mp4 format)', key = state.upload_key)
    #if f is not None:
        #tfile  = tempfile.NamedTemporaryFile(delete = True)
        #tfile.write(f.read())


