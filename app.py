import streamlit as st
import pandas as pd

# Load the dataset (ensure 'url' column exists with video links)
data = pd.read_csv('ted_talks.csv')

st.title("TED Talks Recommendation System")

# Dropdown with placeholder
talk = st.selectbox("Select a TED Talk", data['title'], placeholder="Search here")

# Get the URL for the selected talk
if talk:
    video_url = data[data['title'] == talk]['url'].values[0]

    # Button to redirect to the video link
    if st.button("Go to Video"):
        st.markdown(f"[Watch here]({video_url})", unsafe_allow_html=True)
