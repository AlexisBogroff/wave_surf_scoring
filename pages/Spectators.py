import streamlit as st
import youtube_dl
import cv2
import os

st.set_page_config(
    page_icon='🏄',
    page_title="Judge - accurate scoring",
    layout='wide'
)

st.title('🏄 Spectators - direct video')

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
        st.write('***Creating frame from video..***')

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
        pass
        ### MODEL HERE ###