import os
import time
import random
import cv2
from flask import Flask,  Response, stream_with_context, jsonify, request
from pytube import YouTube
import json
import re

app = Flask(__name__)


@app.route('/analyze_video')
def analyze_video():
    """Analyze the video from the provided URL and return the scores of the frames."""
    data = request.get_json()
    print(data)
    # data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    url = data['url']

    # Download the video from the URL
    video_title = download_video(url)

    if video_title is None:
        return jsonify({'error': 'Failed to download the video from the provided YouTube URL'}), 400

    def generate_frames():
        # Extract frames from the video
        frames_list = extract_frames(video_title)

        frame_scores = {}
        for i,frame in enumerate(frames_list):
            frame_scores[i] = get_score(frame)
            yield f"{json.dumps({i: get_score(frame)})}\n\n"

    return Response(stream_with_context(generate_frames()), content_type='text/event-stream')


def download_video(url):
    """Download the video from the provided URL and return the title of the video."""
    yt = YouTube(url)
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first()

    if video:
        # Define the folder to save the video
        destination_folder = "videos"
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Download the video
        video.download(output_path=destination_folder)

        # return the title of the video
        return re.sub(r'[^\w\s]', '', yt.title)
    else:
        return None

def extract_frames(video_title):
    """Extract frames from the video and save them to the 'frames' folder. 
    Return the list of the extracted frames."""

    if not os.path.exists('frames'):
        os.makedirs('frames')

    if not os.path.exists(f'frames/{video_title}'):
        os.makedirs(f'frames/{video_title}')

    cam = cv2.VideoCapture(f'videos/{video_title}.mp4')

    currentframe = 0
    frames_list = []

    while True:
        ret, frame = cam.read()
        if ret:
            if currentframe % 300 == 0:
                name = f'./frames/{video_title}/' + str(currentframe) + '.jpg'
                frames_list.append(currentframe)
                cv2.imwrite(name, frame)

            currentframe += 1
        else:
            break

    cam.release()

    # Return list of the extracted frames
    return frames_list

def get_score(frame):
    """Return the score of the frame."""

    #@TODO: Replace with the model
    time.sleep(0.2)
    return random.random()

if __name__ == '__main__':
    app.run(debug=True)
