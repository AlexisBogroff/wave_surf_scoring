import json
import streamlit as st
import requests

st.set_page_config(
    page_icon='🎯',
    page_title="Judge - accurate scoring",
    layout='wide'
)

st.title('🎯 Judge - accurate scoring')

youtube_url = st.text_input('Enter a Youtube link')

if st.button('Send video', type='primary'):
    st.text('Starting download..')

    with st.spinner('Calcul du score...'):
        placeholder = st.empty()
        response = requests.get('http://127.0.0.1:5000/analyze_video', json={'url': youtube_url}, stream=True)
        if response.status_code == 200:
            try:
                for line in response.iter_lines():
                    decoded_line = line.decode('utf-8')
                    if decoded_line:
                        data = json.loads(decoded_line)
                        placeholder.text(f"Frame: {list(data.keys())[0]}, Score: {list(data.values())[0]}")

            except KeyboardInterrupt:
                st.text('Stopped.')
        else:
            st.text('Failed to get a response.')

    st.text('Finish predication !')
