!pip install -r requirements.txt
import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

CLIENT_ID = "7491ccadae8849a1aef06e054776e16f"
CLIENT_SECRET = "a37973bfe4944e5eb7b1bef93b5637ba"

# Load the dataset and similarity matrix from pickle files
df_sample = pd.read_pickle('df_sample.pkl')
similarity = pd.read_pickle('https://drive.google.com/file/d/1Hwos6UKxh5-AjOGdhyc5NXbkf3zZqDYW/view?usp=drive_link')

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(song, df_sample, similarity):
    # Print the indices for debugging
    print(f"DataFrame indices: {df_sample.index}")

    # Check if the song exists in the DataFrame
    if song not in df_sample['name'].values:
        st.error(f"The selected song '{song}' is not found in the dataset.")
        return [], [], []

    # Get the index only if the song exists
    index = df_sample[df_sample['name'] == song].index
    print(f"Selected index: {index}")
    
    if index.empty:
        st.error(f"The index for the selected song '{song}' is not found.")
        return [], [], []

    index = index[0]
    
    # Check if the index is within the valid range
    if index >= len(similarity) or index < 0:
        st.error(f"The index {index} is out of range.")
        return [], [], []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_music_names = []
    recommended_music_posters = []
    recommended_music_url = []

    for i in range(min(10, len(distances))):  # Ensure we don't go out of bounds
        # Print the current index for debugging
        print(f"Current index: {distances[i][0]}")

        # Check if the current index is within the valid range
        if distances[i][0] >= len(df_sample.index) or distances[i][0] < 0:
            st.error(f"The current index {distances[i][0]} is out of range.")
            return [], [], []

        # fetch the album cover URL
        artist = df_sample.iloc[distances[i][0]].artist
        recommended_music_posters.append(get_song_album_cover_url(df_sample['name'].iloc[distances[i][0]], artist))
        recommended_music_names.append(df_sample['name'].iloc[distances[i][0]])
        recommended_music_url.append(df_sample['spotify_preview_url'].iloc[distances[i][0]])

    return recommended_music_names, recommended_music_posters, recommended_music_url

st.header('1 Million Songs')

music_list = df_sample['name'].values
selected_song = st.selectbox("Type or select a song from the dropdown", music_list)
# Remove trailing spaces from the selected song
selected_song = selected_song.rstrip()

if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters, recommended_music_url = recommend(selected_song, df_sample, similarity)
    
    col1, col2, col3, col4, col5 = st.columns(5)

    for i in range(min(10, len(recommended_music_names))):  # Ensure we don't go out of bounds
        row_number = i % 2  # Calculate the row number based on the pattern
        with col1 if i % 5 == 0 else col2 if i % 5 == 1 else col3 if i % 5 == 2 else col4 if i % 5 == 3 else col5:
            st.text(recommended_music_names[i])
            st.image(recommended_music_posters[i])
            st.markdown(f"[Play Preview]({recommended_music_url[i]})", unsafe_allow_html=True)
