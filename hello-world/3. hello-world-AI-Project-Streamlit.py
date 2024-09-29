import streamlit as st
import cv2
import numpy as np
import PIL
from pathlib import Path
import openvino as ov

st.set_page_config(
    page_title="Hello Image Classification",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",)

st.title("Hello Image Classification :sun_with_face:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100

core = ov.Core()
model = core.read_model(model='models/v3-small_224_1.0_float.xml')
compiled_model = core.compile_model(model = model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def preprocess(image, input_layer):
    input_image = cv2.resize(src=image, dsize=(224, 224))
    input_image = np.expand_dims(input_image, 0)
    return input_image 

def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    imagenet_filename = Path('data/imagenet_2012.txt')
    imagenet_classes = imagenet_filename.read_text().splitlines()
    imagenet_classes = ["background"] + imagenet_classes
    imagenet_classes = imagenet_classes[result_index]
    return imagenet_classes


input = None 
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))
    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        imagenet_classes = predict_image(uploaded_image_cv, conf_threshold = conf_threshold)
        st.image(uploaded_image_cv, channels = "BGR")
        st.markdown(f"<h2 style='color: red;'>{''.join(imagenet_classes)}</h2><h4 style='color: blue;'><strong>The result of running the AI inference on an image:</strong></h4>", unsafe_allow_html=True)
    else: 
        st.image("data/coco.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image." )


