import streamlit as st
import youtube_dl
import cv2
import os
import re
import math
import torch
import pathlib
from super_gradients.training import models
from super_gradients.common.object_names import Models

st.set_page_config(
    page_icon='🎯',
    page_title="Judge - accurate scoring",
    layout='wide'
)

st.title('🎯 Judge - accurate scoring')

st.header('Video Upload')
uploaded_file = st.file_uploader("Upload your own video...")

video_upload = False

if uploaded_file is not None:
    video_title = uploaded_file.name
    with open(f'videos/{video_title}.mp4') as file:
        file.write(uploaded_file.read())
        video_upload = True


youtube_url = st.text_input('Or, enter a Youtube link')

if st.button('Send video', type='primary'):
    st.text('Starting download..')
    st.text(youtube_url)
    with youtube_dl.YoutubeDL({'outtmpl': 'videos/%(title)s.%(ext)s', 'verbose': True}) as ydl:
        ydl.download([youtube_url])
    st.text('Video download successfully !')

    info_dict = ydl.extract_info(youtube_url, download=False)
    video_title = info_dict.get('title', None)
    video_upload = True


if video_upload:
    with st.spinner('***Creating frame from video..***'):

        if not os.path.exists('frames'):
            os.makedirs('frames')

        if not os.path.exists(f'frames/{video_title}'):
            os.makedirs(f'frames/{video_title}')

        cam = cv2.VideoCapture(f'videos/{video_title}.mp4')

        currentframe = 0

        while True:
            ret, frame = cam.read()

            if ret:
                if currentframe % 300 == 0:
                    name = f'./frames/{video_title}/' + str(currentframe) + '.jpg'

                    cv2.imwrite(name, frame)

                currentframe += 1
            else:
                break

        cam.release()
        st.text('Generating frame done, you can run the model.')

st.header('Model inference')
if not video_upload:
    if st.button('Launch inference'):
        st.text('Please upload a video or a link.')
else:
    if st.button('Launch inference', type='primary'):
        yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
        def make_prediction(input_file, action, confidence=0.55):
            """
            Make a prediction using the fixed model and device, and either show or save the result.

            Args:
            - input_file (str): Path to the input file.
            - action (str): Either 'show' or 'save'.
            - confidence (float, optional): Confidence threshold. Defaults to 0.75.

            Returns:
            - None

            Raises:
            - ValueError: If the action is not 'show' or 'save'.
            """
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            if action == "show":
                yolo_nas_pose.to(device).predict(input_file, conf=confidence).show()
                model_pred = yolo_nas_pose.predict(input_file, conf=confidence)[0].prediction.poses
            elif action == "save":
                output_file = pathlib.Path(input_file).stem + "-detections" + pathlib.Path(input_file).suffix
                yolo_nas_pose.to(device).predict(input_file, conf=confidence).save(output_file)
                print(f"Prediction saved to {output_file}")
            else:
                raise ValueError("Action must be either 'show' or 'save'.")

            return model_pred

        frames = os.listdir(f'frames')
        frames.sort(key=lambda x: int(re.search('[0-9]+', x).group(0)))
        for frame in frames:
            print(f'Prediction for frame : {frame}')
            make_prediction(f'frames/{frame}', 'show')

        pred = make_prediction('frames/16800.jpg', 'show')
        pred[0]

        def euclidian(point1, point2):
            return math.sqrt(pow(point2[0] - point1[0], 2) + pow(point2[1] - point1[1], 2))
        point_list = [[4, 6], # Oreille gauche, épaule gauche
              [6, 12], # épaule gauche, hanche gauche
              [12, 14], # hanche gauche, genoux gauche
              [14, 16]] # genoux gauche, pied gauche
        length = sum([euclidian(pred[0][point1], pred[0][point2]) for point1, point2 in point_list])
        length
