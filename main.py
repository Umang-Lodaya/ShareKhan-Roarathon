from matplotlib import markers
from nbformat import write
import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
import pandas_datareader as data
import datetime
import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from streamlit_folium import folium_static
import folium
import matplotlib as plt
import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
import os
from collections import deque
from pathlib import Path
from typing import List
from streamlit_player import st_player
import json
import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client
import cv2
import time
from PIL import Image
from io import BytesIO
tf.gfile = tf.io.gfile
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
# from object_detection.protos import string_int_label_map_pb2
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import openai
openai.api_key = "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"
import base64
from donationApp import bookDonation, bloodDonation, foodDonation, firstpage, loginPage
import sqlite3
from streamlit_card import card

st.set_page_config(
   page_title="Stonks Exchange",
   page_icon="./logo.png",
   layout="wide",
   initial_sidebar_state="expanded",
)


st.sidebar.title('Helper-site')
rad1 =st.sidebar.radio("Navigation",["Home","Speech-to-text","Text-to-Speech","Summarization","ASL", "Community Page","E-commerce","Map","Help", "Profile", "About-Us"])
# cpsbox = st.sidebar.selectbox("Community Page", ["Write a blog", "Read"])
if rad1 == "Home":
    st.title("capABLE - Empowering Differently-Abled Lives") 
    temp = 32
    w = 16
    hum = 88
    if temp > 35:
        st.warning('Heatwave warning', icon="🔥")
    if w > 15:
        st.warning('Fast winds', icon="🌪")
    if hum > 95:
        st.warning('Heavy rainfall', icon="⛈")
    if hum > 97 and w > 17:
        st.warning('Thunderstorm', icon="🌪")
# st.set_page_config


    # st.warning("This is a warning!")


    col1, col2, col3 = st.columns(3)

    with col1:
        card1 = card(
        title="Title 1",
        text="Some description"
        #   image="http://placekitten.com/200/300",
        #   url="https://github.com/gamcoh/st-card" 
        )
    
    with col2: 
        card2 = card(
        title="Title 2",
        text="Some description"
        #   image="http://placekitten.com/200/300",
        #   url="https://github.com/gamcoh/st-card"
        )
    
    with col3:
        card3 = card(
        title="Title 3",
        text="Some description"
        #   image="http://placekitten.com/200/300",
        #   url="https://github.com/gamcoh/st-card"
        )
        
    # st.image("image_url.jpg", use_column_width=True, caption="Image Caption")

    st.markdown("""
    <style>
        /* Define a custom paragraph style */
        .custom-paragraph {
            text-align: center;
            font-size: 18px;
            line-height: 1.4;
            margin: 10px 0;
            color: white;
        }
        .bold-text{
            font-weight: bold;
            font-size: 20px;
            line-height: 1.4;
            margin: 10px 0;
            text-align: center;
        }
        .custom-list {
        list-style-type: none;
        padding: 0;
    }

    .list-item {
        margin-bottom: 10px;
    }

    input[type="checkbox"] {
        margin-right: 5px;
    }

    .right_to_right{
        text-align: right;
    }
    .button-1 {
    background-color: purple;
    border-radius: 8px;
    border-style: none;
    box-sizing: border-box;
    color: #FFFFFF;
    cursor: pointer;
    display: inline-block;
    font-family: "Haas Grot Text R Web", "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 14px;
    font-weight: 500;
    height: 40px;
    line-height: 20px;
    list-style: none;
    margin: 0;
    outline: none;
    padding: 10px 16px;
    position: relative;
    text-align: center;
    text-decoration: none;
    transition: color 100ms;
    vertical-align: baseline;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
    }

    .button-1:hover,
    .button-1:focus {
    background-color: lightpurple;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create paragraphs with the custom style
    st.markdown('<p class="custom-paragraph">At [NGO Name], we believe that every individual, regardless of their abilities or disabilities, deserves an equal chance at a fulfilling and empowered life. We are committed to breaking down barriers, advocating for inclusivity, and providing a helping hand to those who need it most.</p>', unsafe_allow_html=True)
    st.markdown('<p class="bold-text">Our Mission</p>',unsafe_allow_html=True)
    st.markdown('<p class="custom-paragraph">Our mission is simple yet profound: to empower differently-abled individuals to lead independent, meaningful lives filled with dignity and purpose. We achieve this through a range of programs and services designed to address the unique needs and aspirations of each person we serve.</p>', unsafe_allow_html=True)
    st.markdown('<p class="bold-text">Our Impact</p>',unsafe_allow_html=True)




    # Define the HTML content with CSS classes
    html_content = """
    <ul class="custom-list">
        <li class="list-item"><input type="checkbox" checked> Lives Transformed: Over the years, we've touched the lives of thousands of differently-abled individuals and their families, helping them overcome obstacles and achieve their dreams.</li>
        <li class="list-item"><input type="checkbox" checked> Community Engagement: We actively engage with local communities, businesses, and educational institutions to foster a more inclusive society.</li>
        <li class="list-item"><input type="checkbox" checked> Empowering Through Education: Our educational initiatives have opened doors to learning and personal growth for countless individuals.</li>
        <li class="list-item"><input type="checkbox" checked> Assistive Technologies: We provide access to cutting-edge assistive technologies, enhancing independence and self-reliance.</li>
    </ul>
    """

    # Add the HTML content to the Streamlit app
    st.markdown(html_content, unsafe_allow_html=True)
    st.write("<hr>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    st.markdown('<p class="bold-text">How can you help?</p>',unsafe_allow_html=True)
        # st.subheader("How can you help?")
        # st.markdown('<div style="text-align: right; font-weight: bold; line-height: 1.4; font-size: 25px; color: purple;">How can you help us?</div>', unsafe_allow_html=True)
    st.markdown("Click to know more: ")
    if st.toggle("Donate"):
        st.markdown("<h1 style='text-align: left; color: greenyellow;'>Donation</h1>", unsafe_allow_html=True)

        st.markdown("<h1 style='text-align: left; color: greenyellow;'>Welcome to the Donation page.</h1>", unsafe_allow_html=True)



        st.subheader("Let us come together and donate something for the needy.")
        # menu = ["None", "Book Donation", "Blood Donation","Food Donation","Organization Login/Register"]

        choice = st.radio("Navigation", ["Food Donation", "Blood Donation", "Book Donation"])
        if choice=="Book Donation" :
            bookDonation.bookdonate()
        if choice=="Blood Donation" :
            conn = sqlite3.connect('data1.db',check_same_thread=False)
            cur = conn.cursor()
            def bloodDonate() :
                st.markdown("<h1 style='text-align: left; color: aquamarine;'>Blood Donation</h1>", unsafe_allow_html=True)
                st.title("Welcome to the food donation page, your donated food can bring hope in someones life of survival.\nCome let us donate food for needy one.\nYou don't have to walk and donate it you just have to register yourself and we will pick the food from your house address that will be provided.")
                st.write("Here if you are willing to donate food\n you have to register yourself.")
                with st.form(key="Register for food donation"):
                    nameblood = st.text_input("Enter your name : ")
                    bloodaddress = st.text_input("Enter your address please : ")
                    blood_phone = st.text_input("Enter your phone  Number : ")
                    
                    bloodsubmission = st.form_submit_button(label="Submit", on_click = st.balloons)
                    if bloodsubmission:
                        addData(nameblood,bloodaddress,blood_phone)
                        st.balloons
                        st.success("Congratulations!!!")
                        
                    else:
                        st.info("Please submit the form.")
            def addData(a,b,c):
                
                cur.execute("""CREATE TABLE IF NOT EXISTS food(NAME TEXT(50), ADDRESS TEXT(50), PHONE_NO  TEXT(15)); """) 
                cur.execute("INSERT INTO food VALUES (?,?,?)",(a,b,c))
                conn.commit()
                conn.close()
                st.success("Successfully inserted")
            bloodDonate()


        if choice=="Food Donation" :            
            conn = sqlite3.connect('data1.db',check_same_thread=False)
            cur = conn.cursor()
            def foodDonate():
                st.markdown("<h1 style='text-align: left; color: peachpuff;'>Food Donation</h1>", unsafe_allow_html=True)
                st.title("Welcome to the food donation page, your donated food can bring hope in someones life of survival.\nCome let us donate food for needy one.\nYou don't have to walk and donate it you just have to register yourself and we will pick the food from your house address that will be provided.")
                st.write("Here if you are willing to donate food\n you have to register yourself.")
                with st.form(key="Register for food donation"):
                    namefood = st.text_input("Enter your name : ")
                    foodaddress = st.text_input("Enter your address please : ")
                    food_phone = st.text_input("Enter your phone  Number : ")
                    
                    foodsubmission = st.form_submit_button(label="Submit", on_click = st.balloons)
                    if foodsubmission:
                        addData(namefood,foodaddress,food_phone)
                        st.balloons
                        st.success("Congratulations!!!")
                        
                    else:
                        st.info("Please submit the form.")

            def addData(a,b,c):
                
                cur.execute("""CREATE TABLE IF NOT EXISTS food(NAME TEXT(50), ADDRESS TEXT(50), PHONE_NO  TEXT(15)); """) 
                cur.execute("INSERT INTO food VALUES (?,?,?)",(a,b,c))
                conn.commit()
                conn.close()
                st.success("Successfully inserted")
            foodDonate()
        if choice == "None" :
            firstpage.firstpage()
        if choice== "Organization Login/Register":
            loginPage.loginPages()
            



if rad1 == "Speech-to-text":
    import streamlit as st
    import numpy as np
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
# from streamlit_webrtc import VideoTransformerBase, VideoTransformerContext

    from pydub import AudioSegment
    import queue, pydub, tempfile, openai, os, time
    from bokeh.models.widgets import Button
    from bokeh.models import CustomJS
    from streamlit_bokeh_events import streamlit_bokeh_events
    import speech_recognition as sr #(SpeechRecognition in pypi) 

    def transcribe_speech():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            with st.spinner("Listening..."):
                audio = r.listen(source)
                st.write("Transcribing...")
                try:
                    text = r.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    st.write("Could not understand audio")
                except sr.RequestError as e:
                    st.write("Could not request results from Google Speech Recognition service; {0}".format(e))

    ready_button = st.button("TALK TO ME", key='ready_button')

    if ready_button:
        text = transcribe_speech()
        if text:
            st.write(f"You said: {text}")

if rad1 == "Text-to-Speech":
    from gtts import gTTS
    from io import BytesIO
    st.title("Text-To-Speech")
    text1 = st.text_area("Enter text:",height=None,max_chars=None,key=2,help="Enter your text here")
    sound_file = BytesIO()
    if st.button('Listen'):
        if text1 == "":
            st.warning('Please **enter text** for translation')

        else:
            tts = gTTS(text1, lang='en')
            tts.write_to_fp(sound_file)

            st.audio(sound_file)
    else:
        pass

    from google_trans_new import google_translator # pip install google_trans_new==1.1.9
    import gtts # pip install gtts
    from deep_translator import GoogleTranslator
    Languages = {'afrikaans':'af','albanian':'sq','amharic':'am','arabic':'ar','armenian':'hy','azerbaijani':'az','basque':'eu','belarusian':'be','bengali':'bn','bosnian':'bs','bulgarian':'bg','catalan':'ca','cebuano':'ceb','chichewa':'ny','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','corsican':'co','croatian':'hr','czech':'cs','danish':'da','dutch':'nl','english':'en','esperanto':'eo','estonian':'et','filipino':'tl','finnish':'fi','french':'fr','frisian':'fy','galician':'gl','georgian':'ka','german':'de','greek':'el','gujarati':'gu','haitian creole':'ht','hausa':'ha','hawaiian':'haw','hebrew':'iw','hebrew':'he','hindi':'hi','hmong':'hmn','hungarian':'hu','icelandic':'is','igbo':'ig','indonesian':'id','irish':'ga','italian':'it','japanese':'ja','javanese':'jw','kannada':'kn','kazakh':'kk','khmer':'km','korean':'ko','kurdish (kurmanji)':'ku','kyrgyz':'ky','lao':'lo','latin':'la','latvian':'lv','lithuanian':'lt','luxembourgish':'lb','macedonian':'mk','malagasy':'mg','malay':'ms','malayalam':'ml','maltese':'mt','maori':'mi','marathi':'mr','mongolian':'mn','myanmar (burmese)':'my','nepali':'ne','norwegian':'no','odia':'or','pashto':'ps','persian':'fa','polish':'pl','portuguese':'pt','punjabi':'pa','romanian':'ro','russian':'ru','samoan':'sm','scots gaelic':'gd','serbian':'sr','sesotho':'st','shona':'sn','sindhi':'sd','sinhala':'si','slovak':'sk','slovenian':'sl','somali':'so','spanish':'es','sundanese':'su','swahili':'sw','swedish':'sv','tajik':'tg','tamil':'ta','telugu':'te','thai':'th','turkish':'tr','turkmen':'tk','ukrainian':'uk','urdu':'ur','uyghur':'ug','uzbek':'uz','vietnamese':'vi','welsh':'cy','xhosa':'xh','yiddish':'yi','yoruba':'yo','zulu':'zu'}
    
    st.title("Language Translator")
    text = st.text_area("Enter text:",height=None,max_chars=None,key=1,help="Enter your text here")

    option1 = st.selectbox('Input language',
                        ('english', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch',  'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'))

    option2 = st.selectbox('Output language',
                        ('malayalam', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'))

    value1 = Languages[option1]
    value2 = Languages[option2]
    translator = GoogleTranslator(source=value1, target=value2)
    if st.button('Translate Sentence'):
        if text == "":
            st.warning('Please **enter text** for translation')

        else:
            translate = translator.translate(text)
            st.info(str(translate))

            converted_audio = gtts.gTTS(translate, lang='en')
            converted_audio.save("translated.mp3")
            audio_file = open('translated.mp3','rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio')
            st.write("To **download the audio file**, click the kebab menu on the audio bar.")
            st.success("Translation is **successfully** completed!")
    else:
        pass

if rad1 == 'Summarization':
    st.title("Learn")

    url = 'https://youtu.be/rmVRLeJRkl4'

    file_highlights = 'o6faqyp448-35de-44c6-b1ff-1e6799699b5b_highlights.json'
    file_chapters = 'o6faqyp448-35de-44c6-b1ff-1e6799699b5b_chapters.json'

    placeholder = st.empty()
    with placeholder.container():
        st_player(url, playing=False, muted=True)
        
    mode = st.selectbox("Summary Mode", ("Chapters", "Highlights"))

    def get_btn_text(start_ms):
        seconds = int((start_ms / 1000) % 60)
        minutes = int((start_ms / (1000 * 60)) % 60)
        hours = int((start_ms / (1000 * 60 * 60)) % 24)
        btn_txt = ''
        if hours > 0:
            btn_txt += f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        else:
            btn_txt += f'{minutes:02d}:{seconds:02d}'
        return btn_txt


    def add_btn(start_ms, key):
        start_s = start_ms / 1000
        if st.button(get_btn_text(start_ms), key):
            url_time = url + '&t=' + str(start_s) + 's'
            with placeholder.container():
                st_player(url_time, playing=True, muted=False)
            

    if mode == "Highlights":
        pass
        # with open(file_highlights, 'r') as f:
        #     data = json.load(f)
        # results = data['results']
        
        # cols = st.columns(3)
        # n_buttons = 0
        # for res_idx, res in enumerate(results):
        #     text = res['text']
        #     timestamps = res['timestamps']
        #     col_idx = res_idx % 3
        #     with cols[col_idx]:
        #         st.write(text)
        #         for t in timestamps:
        #             start_ms = t['start']
        #             add_btn(start_ms, n_buttons)
        #             n_buttons += 1
    else:
        with open(file_chapters, 'r') as f:
            chapters = json.load(f)
        for chapter in chapters:
            start_ms = chapter['start']
            add_btn(start_ms, None)
            txt = chapter['summary']
            st.write(txt)

if rad1 == "ASL":

# Set page configs.
    # st.set_page_config(page_title="Sign Language Detection", layout="centered")

    #-----------------------------------------------------------------------------

    # # Path of the pre-trained TF model
    MODEL_DIR = r"./trained_model/saved_model"

    # # Path of the LabelMap file
    PATH_TO_LABELS = r"./trained_model/label_map.pbtxt"

    # Decision Threshold
    MIN_THRESH = float(0.60)

    @st.cache(allow_output_mutation=True)
    def load_model():
        print('Loading model...', end='')
        start_time = time.time()
        # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
        detect_fn = tf.saved_model.load(MODEL_DIR)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))
        return detect_fn

    # load model
    detect_fn = load_model()

    # -------------Header Section------------------------------------------------

    title = '<p style="text-align: center;font-size: 50px;font-weight: 350;font-family:Cursive "> Sign Language Detection </p>'
    st.markdown(title, unsafe_allow_html=True)

    # -------------Sidebar Section------------------------------------------------

    # with st.markdown:

    title = '<p style="font-size: 25px;font-weight: 550;">Sign Language Detection</p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose the mode for detection
    mode = st.radio("Choose Detection Mode", ('Image Upload',
                                                'Webcam Image Capture',
                                                'Webcam Realtime'), index=0)
    if mode == 'Image Upload':
        detection_mode = mode
    elif mode == 'Video Upload':
        detection_mode = mode
    elif mode == "Webcam Image Capture":
        detection_mode = mode
    elif mode == 'Webcam Realtime':
        detection_mode = mode

    # -------------Image Upload Section------------------------------------------------

    if detection_mode == "Image Upload":
        
        st.markdown("&nbsp; Upload your Image below and our ML model will "
                    "detect signs inside the Image", unsafe_allow_html=True)

        # Example Image
        # st.image(image="./imgs/collage.jpg")
        st.markdown("</br>", unsafe_allow_html=True)

        # Upload the Image
        content_image = st.file_uploader(
            "Upload Content Image (PNG & JPG images only)", type=['png', 'jpg', 'jpeg'])

        st.markdown("</br>", unsafe_allow_html=True)
        st.warning('NOTE : You need atleast Intel i3 with 8GB memory for proper functioning of this application. ' +
                ' All Images are resized to 640x640')

        if content_image is not None:

            with st.spinner("Scanning the Image...will take few secs"):

                content_image = Image.open(content_image)

                content_image = np.array(content_image)

                # Resize image to 640x640
                content_image = cv2.resize(content_image, (640,640))

                # ---------------Detection Phase-------------------------

                # LOAD LABEL MAP DATA FOR PLOTTINg
                category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                                use_display_name=True)

                # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                input_tensor = tf.convert_to_tensor(content_image)

                # The model expects a batch of images, so add an axis with `tf.newaxis`.
                input_tensor = input_tensor[tf.newaxis, ...]

                # Detect Objects
                detections = detect_fn(input_tensor)

                # All outputs are batches tensors.
                # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                # We're only interested in the first num_detections.
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                image_with_detections = content_image.copy()

                # Detected classes
                detected_classes = detections['detection_classes']
                scores = detections['detection_scores']

                print("Detected Classes : ", detected_classes)
                print("Scores : ", scores)

                # ---------------Drawing Phase-------------------------

                classes = {1: "food",
                        2: "yes",
                        3: "no",
                        4: "hello",
                        5: "thank_you"}
            
                responses= {
                    "hello":"Hi, Nice to meet you !",
                    "yes":"Great !",
                    "no":"It's okay, Alright !",
                    "thank_you":"Welcome !",
                    "food":"Wait...",
                }
                
                response_imgs = {
                    "hello":"./imgs/nice to meet you.png",
                    "yes":"./imgs/great.png",
                    "no":"./imgs/its okay.png",
                    "thank_you":"./imgs/welcome.png",
                    "food":"./imgs/wait.png",
                }
            

                # Find indexes with scores greater than the MIN_THRESH
                score_indexes = [idx for idx, element in enumerate(scores) if element > MIN_THRESH]
                detected_classes = [detected_classes[idx] for idx in score_indexes]
                # Replace numbers with class names
                detected_classes = [classes.get(i) for i in detected_classes]

                if len(detected_classes)!=0:
                
                    # Draw the bounding boxes with probability score
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes'],
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=MIN_THRESH,
                        agnostic_mode=False)

                    print('Done')

                    if image_with_detections is not None:
                        # some baloons
                        st.balloons()

                    col1, col2 = st.columns(2)
                    with col1:
                        # Display the output
                        st.image(image_with_detections)
                    with col2:
                        st.markdown("</br>", unsafe_allow_html=True)
                        st.markdown(f"<h5> Detected : {detected_classes[0]} </h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5> Response : {responses[detected_classes[0]]} </h5>", unsafe_allow_html=True)
                        
                        # Display the response image
                        st.image(image=response_imgs[detected_classes[0]])
                        
                        st.markdown(
                            "<b> Your Image is Ready ! Click below to download it. </b> ", unsafe_allow_html=True)

                        # convert to pillow image
                        img = Image.fromarray(image_with_detections)
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        st.download_button(
                            label="Download image",
                            data=buffered.getvalue(),
                            file_name="output.png",
                            mime="image/png")
            
                else:
                    st.markdown(f"<h5> No Signs Found inside Image..Try another Image ! </h5>", unsafe_allow_html=True)      
            
    # -------------Webcam Image Capture Section------------------------------------------------

    if detection_mode == "Webcam Image Capture":

        st.info("NOTE : In order to use this mode, you need to give webcam access.")

        img_file_buffer = st.camera_input("Capture an Image from Webcam", disabled=False, key=1,
                                        help="Make sure you have given webcam permission to the site")

        if img_file_buffer is not None:

            with st.spinner("Detecting Signs ..."):
                
                # To read image file buffer as a PIL Image:
                img = Image.open(img_file_buffer)

                # To convert PIL Image to numpy array:
                img = np.array(img)
                
                # Resize image to 640x640
                content_image = cv2.resize(img, (640,640))

                # ---------------Detection Phase-------------------------

                # LOAD LABEL MAP DATA FOR PLOTTINg
                category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                                use_display_name=True)

                # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                input_tensor = tf.convert_to_tensor(content_image)

                # The model expects a batch of images, so add an axis with `tf.newaxis`.
                input_tensor = input_tensor[tf.newaxis, ...]

                # Detect Objects
                detections = detect_fn(input_tensor)

                # All outputs are batches tensors.
                # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                # We're only interested in the first num_detections.
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                image_with_detections = content_image.copy()

                # Detected classes
                detected_classes = detections['detection_classes']
                scores = detections['detection_scores']

                print("Detected Classes : ", detected_classes)
                print("Scores : ", scores)

                # ---------------Drawing Phase-------------------------

                classes = {1: "food",
                        2: "yes",
                        3: "no",
                        4: "hello",
                        5: "thank_you"}
            
                responses= {
                    "hello":"Hi, Nice to meet you !",
                    "yes":"Great !",
                    "no":"It's okay, Alright !",
                    "thank_you":"Welcome !",
                    "food":"Wait...",
                }
                
                response_imgs = {
                    "hello":"./imgs/nice to meet you.png",
                    "yes":"./imgs/great.png",
                    "no":"./imgs/its okay.png",
                    "thank_you":"./imgs/welcome.png",
                    "food":"./imgs/wait.png",
                }
            

                # Find indexes with scores greater than the MIN_THRESH
                score_indexes = [idx for idx, element in enumerate(scores) if element > MIN_THRESH]
                detected_classes = [detected_classes[idx] for idx in score_indexes]
                # Replace numbers with class names
                detected_classes = [classes.get(i) for i in detected_classes]

                if len(detected_classes)!=0:
                
                    # Draw the bounding boxes with probability score
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes'],
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=MIN_THRESH,
                        agnostic_mode=False)

                    print('Done')

                    if image_with_detections is not None:
                        # some baloons
                        st.balloons()

                    col1, col2 = st.columns(2)
                    with col1:
                        # Display the output
                        st.image(image_with_detections)
                    with col2:
                        st.markdown("</br>", unsafe_allow_html=True)
                        st.markdown(f"<h5> Detected : {detected_classes[0]} </h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5> Response : {responses[detected_classes[0]]} </h5>", unsafe_allow_html=True)
                        
                        # Display the response image
                        st.image(image=response_imgs[detected_classes[0]])
                        
                        st.markdown(
                            "<b> Your Image is Ready ! Click below to download it. </b> ", unsafe_allow_html=True)

                        # convert to pillow image
                        img = Image.fromarray(image_with_detections)
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        st.download_button(
                            label="Download image",
                            data=buffered.getvalue(),
                            file_name="output.png",
                            mime="image/png")
            
                else:
                    st.markdown(f"<h5> No Signs Found inside Image..Try another Image ! </h5>", unsafe_allow_html=True)    

    # -------------Webcam Realtime Section------------------------------------------------

    if detection_mode == "Webcam Realtime":

        st.warning("NOTE : In order to use this mode, you need to give webcam access. "
                "After clicking 'Start' , it takes about 10-20 seconds to ready the webcam.")

        spinner_message = "Wait a sec, getting some things done..."

        with st.spinner(spinner_message):

            class VideoProcessor:

                def recv(self, frame):
                    # convert to numpy array
                    
                    frame = frame.to_ndarray(format="bgr24")

                    # Resize image to 640x640
                    content_image = cv2.resize(frame, (640,640))

                    # ---------------Detection Phase-------------------------

                    # LOAD LABEL MAP DATA FOR PLOTTINg
                    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                                use_display_name=True)

                    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                    input_tensor = tf.convert_to_tensor(content_image)

                    # The model expects a batch of images, so add an axis with `tf.newaxis`.
                    input_tensor = input_tensor[tf.newaxis, ...]

                    # Detect Objects
                    detections = detect_fn(input_tensor)

                    # All outputs are batches tensors.
                    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                    # We're only interested in the first num_detections.
                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                for key, value in detections.items()}
                    detections['num_detections'] = num_detections

                    # detection_classes should be ints.
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                    image_with_detections = content_image.copy()

                    # Detected classes
                    detected_classes = detections['detection_classes']
                    scores = detections['detection_scores']

                    print("Detected Classes : ", detected_classes)
                    print("Scores : ", scores)
                    
                    # ---------------Drawing Phase-------------------------

                    classes = {1: "food",
                            2: "yes",
                            3: "no",
                            4: "hello",
                            5: "thank_you"}
                    
                    # Find indexes with scores greater than the MIN_THRESH
                    score_indexes = [idx for idx, element in enumerate(scores) if element > MIN_THRESH]
                    detected_classes = [detected_classes[idx] for idx in score_indexes]
                    # Replace numbers with class names
                    detected_classes = [classes.get(i) for i in detected_classes]

                    if len(detected_classes)!=0:
                
                        # Draw the bounding boxes with probability score
                        viz_utils.visualize_boxes_and_labels_on_image_array(
                            image_with_detections,
                            detections['detection_boxes'],
                            detections['detection_classes'],
                            detections['detection_scores'],
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=5,
                            min_score_thresh=MIN_THRESH,
                            agnostic_mode=False)

                        print('Done')

                    frame = av.VideoFrame.from_ndarray(image_with_detections, format="bgr24")

                    return frame

            webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                            rtc_configuration=RTCConfiguration(
                                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))


if rad1 == "Community Page":
    # Set up a directory for saving the blog posts and images
    if not os.path.exists("blog_posts"):
        os.makedirs("blog_posts")
    if not os.path.exists("blog_images"):
        os.makedirs("blog_images")

    # Define the sidebar for creating new blog posts
    def create_main():
        st.title("Create New Blog Post")
        post_title = st.text_input("Title")
        post_tags = st.text_input("Tags (comma-separated)")
        post_content = st.text_area("Content", height=500)
        post_formatting = st.selectbox("Formatting", ["Markdown", "Plain Text"])
        post_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
        if st.button("Create"):
            # Save the blog post and any uploaded images
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            post_filename = f"{post_title}.md"
            with open(f"blog_posts/{post_filename}", "w") as f:
                f.write(f"Title: {post_title}\n")
                f.write(f"Tags: {post_tags}\n")
                f.write(f"Timestamp: {timestamp}\n")
                if post_image is not None:
                    image_filename = f"{timestamp}-{post_image.name}"
                    with open(f"blog_images/{image_filename}", "wb") as img_file:
                        img_file.write(post_image.getbuffer())
                    f.write(f"Image: {image_filename}\n")
                f.write("\n")
                f.write(post_content)
            st.success("Blog post created!")

    # Define the main content area for viewing existing blog posts
    def create_main_content():
        col1, col2 = st.columns(2)
        col1.title("Swatantra Lekh")
        post_files = os.listdir("blog_posts")
        if not post_files:
            st.info("No blog posts yet.")
        for post_file in post_files:
            with open(f"blog_posts/{post_file}", "r") as f:
                post_content = f.read()
                post_lines = post_content.split("\n")
                post_title = post_lines[0][7:]
                post_tags = post_lines[1][6:].split(",")
                post_timestamp = post_lines[2][11:]
                place_img = r'place_img.png'
                post_image = None
                if len(post_lines) > 3 and post_lines[3].startswith("Image: "):
                    post_image = post_lines[3][7:]
                post_content = "\n".join(post_lines[4:])
                col1.write(f"## {post_title}")
                col1.write(f"*Tags:* {' '.join(['`'+tag.strip()+'`' for tag in post_tags])}")
                col1.write(f"*Date/Time:* {post_timestamp}")
                if post_image is not None:
                    col2.image(f"{place_img}", use_column_width = True)
                if post_image is None:
                    col2.image(f"{place_img}", use_column_width = True)
                if post_file.endswith(".md"):
                    col1.markdown(post_content)
                else:
                    col1.write(post_content)
            col1.download_button("Download blog", post_content)

    # Run the app  
    create_main() 
    create_main_content()
    
if rad1=="E-commerce":
    import webbrowser
    url = 'https://64fc1edc5e5dc333107bbe0d--fanciful-axolotl-a090d0.netlify.app/'
    webbrowser.open_new_tab(url)



if rad1 == "Map":
    st.header("Mobility friendly resteraunts")
    df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [19.07, 72.87],
    columns=['lat', 'lon'])

    st.map(df)

if rad1 == "Help":
    from streamlit_chat import message
    message("My message") 
    message("Hello bot!", is_user=True)
    

if rad1 == "Profile":
        # Security
    #passlib,hashlib,bcrypt,scrypt
    import hashlib
    def make_hashes(password):
        return hashlib.sha256(str.encode(password)).hexdigest()

    def check_hashes(password,hashed_text):
        if make_hashes(password) == hashed_text:
            return hashed_text
        return False
    # DB Management
    import sqlite3 
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    # DB  Functions
    def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


    def add_userdata(username,password):
        c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
        conn.commit()

    def login_user(username,password):
        c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
        data = c.fetchall()
        return data


    def view_all_users():
        c.execute('SELECT * FROM userstable')
        data = c.fetchall()
        return data



    def main():
        """Simple Login App"""

        st.title("Simple Login App")

        menu = ["Login","SignUp"]
        choice = st.selectbox("Menu",menu)

        if choice == "Login":
            st.subheader("Login Section")

            username = st.text_input("User Name")
            password = st.text_input("Password",type='password')
            if st.checkbox("Login"):
                # if password == '12345':
                create_usertable()
                hashed_pswd = make_hashes(password)

                result = login_user(username,check_hashes(password,hashed_pswd))
                if result:

                    st.success("Logged In as {}".format(username))

                    # task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
                    # if task == "Add Post":
                    #     st.subheader("Add Your Post")

                    # elif task == "Analytics":
                    #     st.subheader("Analytics")
                    # elif task == "Profiles":
                    #     st.subheader("User Profiles")
                    #     user_result = view_all_users()
                    #     clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                    #     st.dataframe(clean_db)
                else:
                    st.warning("Incorrect Username/Password")





        elif choice == "SignUp":
            st.subheader("Create New Account")
            new_user = st.text_input("Username")
            new_password = st.text_input("Password",type='password')

            if st.button("Signup"):
                create_usertable()
                add_userdata(new_user,make_hashes(new_password))
                st.success("You have successfully created a valid Account")
                st.info("Go to Login Menu to login")



    if __name__ == '__main__':
        main()
    # st.title("Your Profile")

    # col1 , col2 = st.columns(2)

    # rad2 =st.radio("Profile",["Sign-Up","Sign-In"])


    # if rad2 == "Sign-Up":

    #     st.title("Registration Form")



    #     col1 , col2 = st.columns(2)

    #     fname = col1.text_input("First Name",value = "first name")

    #     lname = col2.text_input("Second Name")

    #     col3 , col4 = st.columns([3,1])

    #     email = col3.text_input("Email ID")

    #     phone = col4.text_input("Mob number")

    #     col5 ,col6 ,col7  = st.columns(3)

    #     username = col5.text_input("Username")

    #     password =col6.text_input("Password", type = "password")

    #     col7.text_input("Repeat Password" , type = "password")

    #     but1,but2,but3 = st.columns([1,4,1])

    #     agree  = but1.checkbox("I Agree")

    #     if but3.button("Submit"):
    #         if agree:  
    #             st.subheader("Additional Details")

    #             address = st.text_area("Tell Us Something About You")
    #             st.write(address)

    #             st.date_input("Enter your birth-date")

    #             v1 = st.radio("Gender",["Male","Female","Others"],index = 1)

    #             st.write(v1)

    #             st.slider("age",min_value = 18,max_value=60,value = 30,step = 2)

    #             img = st.file_uploader("Upload your profile picture")
    #             if img is not None:
    #                 st.image(img)

    #         else:
    #             st.warning("Please Check the T&C box")

    # if rad2 == "Sign-In":
    #     col1 , col2 = st.columns(2)

    #     username = col1.text_input("Username")

    #     password =col2.text_input("Password", type = "password")

    #     but1,but2,but3 = st.columns([1,4,1])

    #     agree  = but1.checkbox("I Agree")

    #     if but3.button("Submit"):
            
    #         if agree:  
    #             st.subheader("Additional Details")

    #             address = st.text_area("Tell Us Something About You")
    #             st.write(address)

    #             st.date_input("Enter your birth-date")

    #             v1 = st.radio("Gender",["Male","Female","Others"],index = 1)

    #             st.write(v1)

    #             st.slider("age",min_value = 18,max_value=60,value = 30,step = 2)

    #             img = st.file_uploader("Upload your profile picture")
    #             if img is not None:
    #                 st.image(img)
    #         else:
    #             st.warning("Please Check the T&C box")

if rad1 == "About-Us": 
    st.title("Stonks Exchange")

    st.subheader('Locate Us')
    m = folium.Map(location=[18.930131167156954, 72.83363330215157], zoom_start=16)

    # add marker for Bombay Stock Exhcange
    tooltip = "Bombay Stock Exhcange"
    folium.Marker(
        [18.930131167156954, 72.83363330215157], popup="Bombay Stock Exhcange", tooltip=tooltip
    ).add_to(m)

    # call to render Folium map in Streamlit
    folium_static(m)