from utils import get_detection_folder, check_folders, get_yolov5_detection_folder
import redirect as rd

from pathlib import Path
import streamlit as st
from PIL import Image
import subprocess
import os
import sys

# This will check if we have all the folders to save our files for inference
check_folders()

if __name__ == '__main__':
    
    st.title('Gall Detection Streamlit App')

    source = ("YOLOv5", "YOLOv8")
    source_index = st.sidebar.selectbox("Select Model type", range(
        len(source)), format_func=lambda x: source[x])
    
    uploaded_file = st.sidebar.file_uploader(
        "Load File", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='Loading...'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)
            picture = picture.save(f'data/images/{uploaded_file.name}')
            source = f'data/images/{uploaded_file.name}'
    else:
        is_valid = False
    

    if is_valid:
        print('valid')
        if st.button('Detect'):
            if source_index == 1:
                print('v8')
                with rd.stderr(format='markdown', to=st.sidebar), st.spinner('Wait for it...'):
                    print(subprocess.run(['yolo', 'task=detect', 'mode=predict', 'model=model\gall_detectv8.pt', 'conf=0.25', 'source={}'.format(source)],capture_output=True, universal_newlines=True).stderr)

                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))

            else:
                print('v5')
                command = [f"{sys.executable}", "yolov5/detect.py", "--weights", "model/gall_detectv5.pt", "--source", f"{source}"]
                with rd.stderr(format='markdown', to=st.sidebar), st.spinner('Wait for it...'):
                    print(subprocess.run(command ,capture_output=True, universal_newlines=True).stderr)

                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_yolov5_detection_folder()):
                        st.image(str(Path(f'{get_yolov5_detection_folder()}') / img))

