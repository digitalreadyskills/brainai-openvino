# 필요 라이브러리 불러오기
import streamlit as st
import openvino as ov
import cv2
import numpy as np
import io
import tempfile
from PIL import Image 
import moviepy.editor as mpy
from camera_input_live import camera_input_live

# Streamlit 환경 설정
st.set_page_config(
    page_title="Hello Text Detection",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",)

st.title("Hello Text Dection :sun_with_face:")
st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])
st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100

# *****************************************************************************
# OpenVINO AI 모델 불러 로딩
core = ov.Core()
model = core.read_model(model="model/horizontal-text-detection-0001.xml")
compiled_model = core.compile_model(model = model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output("boxes")

# 새로 입력된 데이터 전처리
def preprocess(uploaded_image_cv, input_layer):
    N, C, H, W = input_layer.shape
    resized_image = cv2.resize(uploaded_image_cv, (W, H))
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    return input_image, resized_image

# AI 추론
def predict_image(uploaded_image_cv, conf_threshold):
    input_image, resized_image = preprocess(uploaded_image_cv, input_layer)
    boxes = compiled_model([input_image])[output_layer]
    boxes = boxes[~np.all(boxes == 0, axis=1)]
    return boxes, resized_image

# 추론 결과 후처리
def convert_result_to_image(bgr_image, resized_image, boxes, conf_threshold, conf_labels=True):
    # Define colors for boxes and descriptions.
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}
    # Fetch the image shapes to calculate a ratio.
    (real_y, real_x), (resized_y, resized_x) = (
        bgr_image.shape[:2],
        resized_image.shape[:2],
    )
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    # Convert the base image from BGR to RGB format.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # Iterate through non-zero boxes.
    for box in boxes:
        # Pick a confidence factor from the last place in an array.
        conf = box[-1]
        if conf > conf_threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio.
            # If the bounding box is found at the top of the image,
            # position the upper box bar little lower to make it visible on the image.
            (x_min, y_min, x_max, y_max) = [
                (int(max(corner_position * ratio_y, 10)) if idx % 2 else int(corner_position * ratio_x)) for idx, corner_position in enumerate(box[:-1])
            ]
            # Draw a box based on the position, parameters in rectangle function are: image, start_point, end_point, color, thickness.
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)
            # Add text to the image based on position and confidence.
            # Parameters in text function are: image, text, bottom-left_corner_textfield, font, font_scale, color, thickness, line_type.
            if conf_labels:
                rgb_image = cv2.putText(
                    rgb_image,
                    f"{conf:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )
    return rgb_image
# *************************

# IMAGE 선택시
input = None 
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))
    if input is not None:
        uploaded_image = Image.open(input)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        boxes, resized_image = predict_image(uploaded_image_cv, conf_threshold = conf_threshold)
        result_image = convert_result_to_image(uploaded_image_cv, resized_image, boxes, conf_threshold, conf_labels=False)
        st.image(result_image, channels = "RGB")
        st.markdown(f"<h4 style='color: blue;'><strong>The result of running the AI inference on an image.</strong></h4>", unsafe_allow_html=True)
    else: 
        st.image("data/intel_rnb.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image." )

# VIDEO 선택시
def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    fps = camera.get(cv2.CAP_PROP_FPS)
    temp_file_2 = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
    video_row=[]
    # frame
    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0
    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()
        if ret:
            try:
                boxes, resized_image = predict_image(frame, conf_threshold)
                visualized_image = convert_result_to_image(frame, resized_image, boxes, conf_threshold, conf_labels=False)
            except:
                visualized_image = frame
            st_frame.image(visualized_image, channels = "BGR")
            video_row.append(cv2.cvtColor(visualized_image,cv2.COLOR_BGR2RGB))
            frame_count +=1 
            progress_bar.progress(frame_count/total_frames, text=None)
    
        else:
            progress_bar.empty()
            camera.release()
            st_frame.empty()
            break
        clip = mpy.ImageSequenceClip(video_row, fps = fps)
        clip.write_videofile(temp_file_2.name)
        st.video(temp_file_2.name)
        
temporary_location = None
if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input_file = st.sidebar.file_uploader("Choose a video.", type=("mp4"))
    if input_file is not None:
        # 파일을 임시 경로에 저장
        g = io.BytesIO(input_file.read())
        temporary_location = "upload.mp4"
        with open(temporary_location, "wb") as out:
            out.write(g.read())
        out.close()
    # 업로드된 비디오 파일이 있는 경우 비디오 재생
    if temporary_location is not None:
        play_video(temporary_location)
    else:
        st.video("data/sample_video.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on a video.")
        
# WEBCAM 선택시
if source_radio == "WEBCAM":
    input = camera_input_live()
    uploaded_image = Image.open(input)
    uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    boxes, resized_image = predict_image(uploaded_image_cv, conf_threshold)
    visualized_image = convert_result_to_image(uploaded_image_cv, resized_image, boxes, conf_threshold, conf_labels=False)
    st.image(visualized_image, channels = "RGB")
