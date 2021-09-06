import streamlit as st
from fer import FER, Video
from PIL import Image
import matplotlib.image as mpimg
import pandas as pd

st.header("**Emotionify** Facial Emotion Detection")
uploaded_file = st.file_uploader(label="Upload your image to analyze the emotion", type=['png', 'jpg', 'jpeg'])
detector = FER(mtcnn=True)
if uploaded_file is not None:
    print(uploaded_file)
    if 'image' in uploaded_file.type:
        img = mpimg.imread(uploaded_file)

        st.header("You're **{}**".format(detector.top_emotion(img)[0]))
        emotions = detector.detect_emotions(img)[0]['emotions']
        
        st.bar_chart(pd.DataFrame.from_dict(emotions, orient='index'))
        st.image(img)
    
    if 'video' in uploaded_file.type:

        video = Video(uploaded_file)

        # Output list of dictionaries
        raw_data = video.analyze(detector, display=False)

        # Convert to pandas for analysis
        df = video.to_pandas(raw_data)
        df = video.get_first_face(df)
        df = video.get_emotions(df)
        st.video(uploaded_file)

        # Plot emotions
        st.write(df.plot())
        # plt.show()
        
st.write("Stay Tuned! **Moodify** is in progress 💕")
