import copy
import cv2
import datetime
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from vega_datasets import data
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


@st.cache(persist=True)
def load_data(path):
    data=pd.read_csv(path)
    return data

def show_visualization():

    #load in data
    fb_mask_original=load_data("fb_mask.csv")
    fb_sympton_original=load_data("fb_sympton.csv")
    fb_sympton=copy.deepcopy(fb_sympton_original)
    fb_mask=copy.deepcopy(fb_mask_original)
    fb_mask['time_value']= pd.to_datetime(fb_mask['time_value'], format='%Y/%m/%d')
    fb_sympton['time_value']= pd.to_datetime(fb_sympton['time_value'], format='%Y/%m/%d')
    fb_mask.rename(columns={'value':'mask_percentage'}, inplace=True)
    fb_sympton.rename(columns={'value':'sympton_percentage'}, inplace=True)

    fb_all=fb_mask.merge(fb_sympton, on=['time_value','geo_value'])
    fb_all=fb_all[['geo_value','time_value','mask_percentage','sympton_percentage']]
    fb_all = fb_all[fb_all['time_value']>'2020-09-08']

    states=fb_all.geo_value.str.upper().unique()

    #first plot: correlation between wearing mask and having symptons
    st.title("Let`s see the correlation between wearing mask and having symptons.")

    state_choice = st.sidebar.multiselect(
        "Which state are you interested in?",
        states.tolist(), default=['AK','AL','AR','AZ','CA','CO']
    )

    date_range = st.sidebar.date_input("Which range of date are you interested in? Choose between %s and %s"% (min(fb_all['time_value']).strftime('%Y/%m/%d'),  max(fb_all['time_value']).strftime('%Y/%m/%d')), [min(fb_all['time_value']), max(fb_all['time_value'])])

    fb_temp = fb_all[fb_all['geo_value'].str.upper().isin(state_choice)]

    if len(date_range)==2:
        fb_selected = fb_temp[(fb_temp['time_value']>=pd.to_datetime(date_range[0])) & (fb_temp['time_value']<=pd.to_datetime(date_range[1]))]
    else:
        fb_selected = fb_temp[(fb_temp['time_value']>=pd.to_datetime(date_range[0]))]

    scatter_chart = alt.Chart(fb_selected).mark_circle().encode(
        x=alt.X('mask_percentage', scale=alt.Scale(zero=False), axis=alt.Axis(title='percentage of wearing masks')), 
        y=alt.Y('sympton_percentage', scale=alt.Scale(zero=False), axis=alt.Axis(title='percentage of having covid symptons'))
    )
    scatter_chart + scatter_chart.transform_regression('mask_percentage', 'sympton_percentage').mark_line()



    map_data = fb_all[fb_all['time_value']==pd.to_datetime(date_range[0])].copy()
    ids = [2,1,5,4,6,8,9,11,10,12,13,15,19,16,17,18,20,21,22,25,24,23,26,27,29,28,30,37,38,31,33,34,35,32,
           36,39,40,41,42,44,45,46,47,48,49,51,50,53,55,54,56] 
    map_data['id'] = ids

    states = alt.topo_feature(data.us_10m.url, 'states')
    variable_list = ['mask_percentage','sympton_percentage']

    chart = alt.Chart(states).mark_geoshape().encode(
        alt.Color(alt.repeat('row'), type='quantitative')
    ).transform_lookup(
        lookup='id',
        from_=alt.LookupData(map_data, 'id', variable_list)
    ).properties(
        width=500,
        height=300
    ).project(
        type='albersUsa'
    ).repeat(
        row=variable_list
    ).resolve_scale(
        color='independent'
    )

    st.write(chart)


def mask_detection(image, conf=0.5):

    FACE_MODEL_PATH = "./model/face_detector.caffemodel"
    FACE_PROTO_PATH = "./model/deploy.prototxt"
    MASK_MODEL_PATH =  "./model/mask_detector.model"
    IMG_RESIZE = 300

    # record output values
    faces = []
    height, width = image.shape[:2]

    # load models
    model = load_model(MASK_MODEL_PATH)
    net = cv2.dnn.readNet(FACE_PROTO_PATH, FACE_MODEL_PATH)
    net.setInput(cv2.dnn.blobFromImage(image, size=(IMG_RESIZE, IMG_RESIZE)))

    detections = net.forward()
    for i in range(detections.shape[2]):

        # filter out non-faces
        if detections[0, 0, i, 2] > conf:

            # compute face bounding box
            x1, y1, x2, y2 = detections[0, 0, i, 3:7]
            x1, y1, x2, y2 = max(0, int(x1*width)), max(0, int(y1*height)), min(width-1, int(x2*width)), min(height-1, int(y2*height))

            # transform face image for mask detector
            face = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = np.expand_dims(preprocess_input(img_to_array(face)), axis=0)

            # mask detection for faces
            mask, noMask = model.predict(face)[0]
            prob = max(mask, noMask)*100
            label, color = "No Mask", (0, 0, 255) # red
            if mask > noMask: 
                label, color = "Mask", (0, 255, 0) # green

            # store face info for scoring step
            face_info = {}
            face_info["cord"] = (x1, y1, x2, y2)
            face_info["prob"] = prob * (2*(mask > noMask)-1)
            faces.append(face_info)

            # draw result on image
            text = "%s: %.1f%%"%(label, prob)
            font_scale, font, thickness = (abs(x1-x2))/(160), cv2.FONT_HERSHEY_DUPLEX, 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_location = ((x1+x2)//2 - text_size[0]//2, y2 + int(text_size[1] * 1.5))
            cv2.putText(image, text, text_location, font, font_scale, color, thickness)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image, faces

def process_mask_image():
    
    IMAGE_PATH = "./images/"

    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title("Face Mask Detection")
    st.sidebar.text('Mask Detection Options:')

    select_image = None
    upload_image = None
    
    # select sample images
    sample_images = sorted([f for f in os.listdir(IMAGE_PATH) if f.lower().endswith('jpg') or f.lower().endswith('png') or f.lower().endswith('jpeg')])
    try_sample = st.sidebar.checkbox('Try Sample Images')
    if try_sample:
        select_image = st.sidebar.selectbox("Or Try With Sample Images", sample_images)
    else:
        # file upload
        # TODO: INTEGRATE FILE UPLOAD WITH IMAGE SELECTION
        upload_image = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])


    # select confidence level
    conf = st.sidebar.slider('Face Detection Threshold (Higher => Less Faces)', 0.15, 0.85, 0.40)
    show_mask_result = st.sidebar.checkbox('Show Mask Detection Result')

    # process image
    opencv_image = None
    if not try_sample and upload_image is not None:
        # convert the file to an opencv image.
        opencv_image = cv2.cvtColor(np.array(Image.open(upload_image)), cv2.COLOR_BGR2RGB)
    elif try_sample:
        # use selected sample
        opencv_image = cv2.imread(os.path.join(IMAGE_PATH, select_image))
    else:
        st.write("Select or Upload an Image to Begin")

    results = []
    if show_mask_result and opencv_image is not None:
        result_img, results = mask_detection(opencv_image, conf)
        st.image(result_img, channels="BGR", caption='face mask detection result', use_column_width=True)
    elif opencv_image is not None:
        saved_image = st.image(opencv_image, channels="BGR", caption='selected image', use_column_width=True)
    return results

# TODO: IMPLEMENT SCORING MODEL
def calculate_score(results):
    show_eval = st.sidebar.checkbox('Show Safety Level Evaluation')
    if show_eval:
        st.title("Safety Level Evaluation")
        st.write(results)


if __name__ == '__main__':
    show_visualization()
    results = process_mask_image()
    calculate_score(results)
