# Import packages
import os
import cv2
import numpy as np
import sys
import time
import fnmatch
import matplotlib.pyplot as plt
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

MODEL_DIR = 'coco_mobilenet'
MODEL_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = float(0.70)

print(os.getcwd())
print(__file__)

# Get path to current working directory
# Only works in Jupyter notebook
CWD_PATH = os.getcwd()
# CWD_PATH = os.path.dirname(__file__)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, MODEL_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_DIR, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

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

# Path to video dataset
# TODO: Change the file path to videos / frames of interest
startpath_nonviolent = 'raw_data/videos_dataset/Real Life Violence Dataset/NonViolence/'

PATH_NONVIOLENT = os.path.join(CWD_PATH, startpath_nonviolent)

# Create a list with all the file names
vids_list = os.listdir(startpath_nonviolent)

# Establish a pattern to detect only videos
pattern = "*.mp4"

# Create a list of names of every file in the dataset
filename_list = []
for file in vids_list:
    if fnmatch.fnmatch(file, pattern):
        filename_list.append(file)

# Initialize video
for video in filename_list[0:1000]:

    capture = cv2.VideoCapture(
        f'{PATH_NONVIOLENT}{video}'
    )

    # Create window
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

    # Get the dimensions of the video used for rectangle creation
    imW = capture.get(3)  # float `width`
    imH = capture.get(4)  # float `height`

    while capture.isOpened():
        # Capture the video frame
        ret, frame = capture.read()

        if not ret:
            break

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
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
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

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

                        # Build a rectangle
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                      (10, 255, 0), 2)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            break

# cleanup
capture.release()
cv2.destroyAllWindows()
