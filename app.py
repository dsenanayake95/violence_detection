import streamlit as st
from streamlit_player import st_player
import requests
import tempfile
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
from tensorflow.lite.python.interpreter import Interpreter

# Get path to model
PATH_FOR_MY_MODEL = 'violence_detection/models/VGG19_lr_0.0002_model_v3-0.7082'

# Get path to current working directory
CWD_PATH = os.getcwd()

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
    st.title('Violence Detection')
    st.subheader('Can we detect violence in video?')

    '''Currently, the most common way to identify violent behaviour \
            in video is using "human monitors". The extended exposure to violence in videos \
                  can cause harm to the mental health of these individuals. In addition, \
                  monitors may not be able to identify violence as it is happening \
                          meaning fewer opportunities to intervene.'''

    if st.button('Example'):

        col1, col2 = st.columns(2)

        col1.subheader('Group of Men on a Field')
        non_violent = Image.open(os.path.join(CWD_PATH, 'images_frontend/non_violent_sample.jpg'))
        col1.image(non_violent, caption='Probability of Violence: 30%', use_column_width=True)

        col2.subheader('Man About to Punch Another Person')
        violent = Image.open(os.path.join(CWD_PATH, 'images_frontend/violent_sample.jpg'))
        col2.image(violent, caption='Probability of Violence: 100%', use_column_width=True)

    '''We use transfer-learning and a CNN-RNN model to identify violent \
        behaviour in videos. Our output is the probability of violent behaviour throughout \
            the video. This approach means a reduction in the need for human monitors \
                meaning a reduction in the negative impact on their mental health and \
                    potentially the earlier identification of intervention.'''

#########################################
#           Upload a video              #
#########################################

elif direction == 'Try the model':
    prediction_values = []

    model = tf.keras.models.load_model(PATH_FOR_MY_MODEL)

    st.write('model has been loaded')

    upload = st.empty()
    frames = 0

    MODEL_DIR = 'coco_mobilenet'
    MODEL_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, MODEL_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_DIR, LABELMAP_NAME)

    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    with upload:
        video = st.file_uploader('Upload Video file (mpeg/mp4 format)')
        if video is not None:
            st.write("video uploaded")
            tfile = tempfile.NamedTemporaryFile(delete=True)
            tfile.write(video.read())

            vf = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            # Get the dimensions of the video used for rectangle creation
            imW = vf.get(3)  # float `width`
            imH = vf.get(4)  # float `height`

            while vf.isOpened():
                ret, frame = vf.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)

                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                # Perform the actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Retrieve detection results
                # Bounding box coordinates of detected objects
                boxes = interpreter.get_tensor(output_details[0]['index'])[0]

                # Class index of detected objects
                classes = interpreter.get_tensor(output_details[1]['index'])[0]

                # Confidence of detected objects
                scores = interpreter.get_tensor(output_details[2]['index'])[0]

                # Locate indexes for persons classes only
                if 0 in classes:
                    idx_list = [idx for idx, val in enumerate(classes) if val == 0]

                    # Reassign bounding boxes only to detected people
                    boxes = [boxes[i] for i in idx_list]

                    # Loop over all detections and draw detection box if confidence is above minimum threshold
                    for i in range(len(scores)):
                        if ((scores[i] > 0.70) and (scores[i] <= 1.0)):

                            # Get bounding box coordinates and draw box for all people detected
                            if len(boxes) > 0:
                                # Find the top-most top
                                top = min([i[0] for i in boxes])
                                # Find the left-most left
                                left = min([i[1] for i in boxes])
                                # Find the bottom-most bottom
                                bottom = max([i[2] for i in boxes])
                                # Find the right-most right
                                right = max([i[3] for i in boxes])

                                # Convert bounding lines into coordinates
                                # Interpreter can return coordinates that are outside of image dimensions,
                                # Need to force them to be within image using max() and min()
                                ymin = int(max(1, (top * imH)))
                                xmin = int(max(1, (left * imW)))
                                ymax = int(min(imH, (bottom * imH)))
                                xmax = int(min(imW, (right * imW)))

                                # Save cropped area into a variable for each frame
                                rectangle = frame_rgb[ymin:ymax, xmin:xmax]

                                #########################################
                                #         Predict on the video          #
                                #########################################

                                if rectangle is not None:

                                    cv2.rectangle(frame_rgb, (xmin, ymin),
                                                  (xmin + 290, ymin + 50),
                                                  (0, 0, 0), -1)

                                    prediction = model.predict(
                                        np.expand_dims(tf.image.resize(
                                            (rectangle), [224, 224]),
                                                      axis=0) / 255.0)

                                    if prediction is not None:
                                        prediction_values.append(prediction[0][0]*100)
                                    else:
                                        prediction_values.append(0)

                                    if len(prediction_values) < 3:
                                        cv2.rectangle(frame_rgb, (xmin, ymin),
                                                      (xmax, ymax),
                                                      (10, 255, 0), 2)
                                    elif prediction_values[-1] >= 80 and prediction_values[-2] >= 80 and prediction_values[-3] >= 80:
                                        cv2.rectangle(frame_rgb, (xmin, ymin),
                                                          (xmax, ymax),
                                                          (255, 0, 0), 2)
                                    else:
                                        cv2.rectangle(frame_rgb, (xmin, ymin),
                                                          (xmax, ymax),
                                                          (10, 255, 0), 2)

                            cv2.putText(
                                frame_rgb,
                                f"violence:{round(prediction[0][0]*100)}%",
                                (xmin + 20, ymin + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 255, 10), 4)

                stframe.image(frame_rgb)


    #########################################
    #           Button to refresh           #
    #########################################

    # col1, col2, col3 = st.columns(3)

    # if col1.button('Play again'):
    #     st.experimental_rerun()

    # col2.write('OR')

    # if col3.button('Upload another video'): #### or loop back to main function here
    #     video = st.file_uploader('Upload Video file (mpeg/mp4 format)')
    #     if video is not None:
    #         tfile  = tempfile.NamedTemporaryFile(delete = True)
    #         tfile.write(video.read())
