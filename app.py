import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import ast
import os
import base64
from requests import post, get
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Load models 
model1_path = os.path.join("models", "my_model1.h5")
model2_path = os.path.join("models", "my_model2.h5")
model1 = load_model(model1_path)
model2 = load_model(model2_path)


# Highlight rows
def highlight_rows(row):
     # Initialize an empty list to store the styles
    styles = ['' for _ in row.index]

            # Check if 'Prediction_Model1' column exists and highlight
    if 'Prediction_Model1' in row.index:
        if row['Prediction_Model1'] >= 0.5:
            styles[row.index.get_loc('Prediction_Model1')] = 'background-color: green'
        else:
            styles[row.index.get_loc('Prediction_Model1')] = 'background-color: red'

            # Check if 'Mood_Definition' column exists and apply color based on its value
    if 'Mood_Definition' in row.index and pd.notna(row['Mood_Definition']):
        mood_colors = {
            0: 'background-color: lightblue',    # Light blue for 0
            1: 'background-color: skyblue',      # Sky blue for 1
            2: 'background-color: royalblue',    # Royal blue for 2
            3: 'background-color: navy'          # Navy blue for 3
        }
        mood_color = mood_colors.get(row['Mood_Definition'], '')
        styles[row.index.get_loc('Mood_Definition')] = mood_color

    return styles


# Spotify API Authentication

def get_playlist_id_from_url(url):
    """
    Extract the Spotify playlist ID from a given URL.
    Handles two formats of URLs:
    1. https://open.spotify.com/playlist/ID
    2. https://open.spotify.com/playlist/ID?additional_parameters
    """
    base_url = "https://open.spotify.com/playlist/"
    if base_url in url:
        # Extract the part of the URL after the base URL
        id_part = url.split(base_url)[-1]
        # If there are query parameters, split by '?' and take the first part
        playlist_id = id_part.split('?')[0]
        return playlist_id
    else:
        return None  # or raise an error/return an empty string based on your preference
def get_token():
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

# Spotify API Calls
def get_playlist_tracks(token, playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = get_auth_header(token)
    response = get(url, headers=headers)
    json_response = json.loads(response.content)
    tracks = json_response['items']
    return tracks

def get_audio_features(token, track_id):
    url = f"https://api.spotify.com/v1/audio-features/{track_id}"
    headers = get_auth_header(token)
    response = get(url, headers=headers)
    audio_features = json.loads(response.content)
    return audio_features

def get_artist_genres(token, artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = get_auth_header(token)
    response = get(url, headers=headers)
    artist_info = json.loads(response.content)
    return artist_info.get('genres', [])

def get_playlist_songs_attributes(token, playlist_id):
    tracks = get_playlist_tracks(token, playlist_id)
    track_details = []
    for track in tracks:
        track_data = track['track']
        if track_data:  # Ensure there is track data
            track_id = track_data.get('id')
            track_name = track_data.get('name')
            track_artist = track_data['artists'][0].get('name') if track_data['artists'] else ''
            track_artist_id = track_data['artists'][0].get('id') if track_data['artists'] else ''
            track_album = track_data['album'].get('name') if track_data.get('album') else ''
            track_release_date = track_data['album'].get('release_date') if track_data.get('album') else ''
            track_popularity = track_data.get('popularity')
            track_duration = track_data.get('duration_ms')
            track_external_url = track_data.get('external_urls', {}).get('spotify')

            artist_genres = get_artist_genres(token, track_artist_id) if track_artist_id else []

            audio_features = get_audio_features(token, track_id) if track_id else {}

            track_info = {
                'id': track_id,
                'name': track_name,
                'artist': track_artist,
                'album': track_album,
                'release_date': track_release_date,
                'popularity': track_popularity,
                'duration_ms': track_duration,
                'spotify_url': track_external_url,
                'genres': artist_genres,
                **audio_features
            }

            track_details.append(track_info)

    df = pd.DataFrame(track_details)
    return df
def preprocess_data(df):
    df['tempo'] = df['tempo'].apply(lambda x: x/2 if x > 135 else x)
    # Convert dates to decades
    def categorize_decade(date_str):

        # Initialize the decade categories with 0
        decades = {
            'decade_1960s_and_before': 0.0,
            'decade_1970s': 0.0,
            'decade_1980s': 0.0,
            'decade_1990s': 0.0,
            'decade_2000s': 0.0,
            'decade_2010s': 0.0,
            'decade_2020s_and_after': 0.0
        }

        # Handle missing or invalid dates
        if pd.isna(date_str) or not date_str.strip():
            return decades

        # Extract the year from the date string
        try:
            # If only the year is present
            year = int(date_str[:4])
        except ValueError:
            # Invalid date format
            return decades

        # Categorize based on the year
        if year <= 1969:
            decades['decade_1960s_and_before'] = 1.0
        elif year <= 1979:
            decades['decade_1970s'] = 1.0
        elif year <= 1989:
            decades['decade_1980s'] = 1.0
        elif year <= 1999:
            decades['decade_1990s'] = 1.0
        elif year <= 2009:
            decades['decade_2000s'] = 1.0
        elif year <= 2019:
            decades['decade_2010s'] = 1.0
        else:
            decades['decade_2020s_and_after'] = 1.0

        return decades

    # Apply the function to the 'release_date' column
    decade_categories = df['release_date'].apply(categorize_decade)
    decade_df = pd.DataFrame(decade_categories.tolist())

    # Combine the original data with the new decade columns
    df = pd.concat([df, decade_df], axis=1)

    #drop the release date column
    df = df.drop('release_date', axis=1)

    for i in range(12):
        column_name = f'key_{i}'
        df[column_name] = df['key'].apply(lambda x: 1.0 if x == i else 0.0)
    df = df.drop('key', axis=1)


    # Genre categorization
    # Define your wanted and unwanted genres lists
    wanted_0 = ['bossa nova', 'soul', 'vocal jazz', 'classic soul', 'indie soul', 'mpb', 'chillhop', 'funk', 'samba', 'brazilian jazz', 'jazz', 'cool jazz', 'samba-jazz', 'neo soul', 'jazz guitar', 'bebop', 'bboy', 'violao', 'southern soul', 'soul jazz', 'contemporary vocal jazz', 'quiet storm', 'latin jazz', 'downtempo', 'coverchill', 'tropicalia', 'italian adult pop', 'italian lounge', 'jazz pop', 'jazz saxophone', 'nu jazz', 'classic italian pop', 'indie jazz', 'motown', 'electronica', 'trip hop', 'jazz funk', 'samba-rock', 'jazz cover', 'bossa nova cover', 'brazilian boogie', 'nova mpb', 'souldies', 'velha guarda', 'jazz blues', 'manguebeat', 'torch song', 'lounge', 'chanson', 'memphis soul',  'psychedelic soul', 'alternative r&b', 'art pop', 'lo-fi beats', 'hip-hop experimental', 'italian jazz', 'psicodelia brasileira', 'british soul', 'jazz boom bap', 'nouvelle chanson francaise', 'afrobeat', 'soul blues', 'afropop',  'electro-pop francais', 'neo r&b', 'northern soul', 'nz reggae', 'sophisti-pop', 'world', 'rare groove', 'smooth jazz', 'musica tradicional cubana', 'pop soul', 'jazz rap', 'adult standards', 'traditional soul', 'british invasion', 'morna', 'philly soul', 'funk rock', 'uk contemporary jazz']

    unwanted_0 = ['bass house','nu disco', 'funky house', 'filter house', 'deep house', 'new french touch', 'organic electronic', 'jazz house', 'melodic house', 'neapolitan funk', 'french indie pop', 'organic house', 'deep disco house',  'tropical house',  'funktronica', 'progressive house', 'edm', 'hamburg electronic', 'balearic', 'uk dance', 'house', 'german techno', 'rock', 'indie napoletano', 'lgbtq+ hip hop', 'escape room',  'post-disco',  'vapor soul', 'austindie', 'tech house', 'german house', 'danish electronic', 'lo-fi house', 'disco house', 'uk house', 'beach house', 'afropop', 'electro jazz', 'cumbia funk', 'deep euro house',  'indietronica', 'mellow gold', 'minimal techno', 'glam rock', 'french pop', 'raboday', 'piano rock', 'jazztronica', 'italo disco']

    wanted_1 = ['indie soul', 'neo soul', 'chillhop', 'soul', 'alternative r&b', 'bboy', 'trip hop', 'british soul', 'downtempo', 'funk', 'electronica', 'nu jazz', 'r&b', 'classic soul', 'french indie pop', 'french indietronica', 'uk contemporary r&b',  'auckland indie', 'urban contemporary', 'electro jazz', 'quiet storm', 'indie r&b', 'jazz funk', 'pop soul', 'souldies', 'sophisti-pop',  'contemporary r&b', 'indietronica', 'indie jazz', 'afrobeat', 'rock', 'art pop', 'instrumental funk', 'jazz boom bap', 'indie pop', 'chill lounge', 'psychedelic soul', 'jazztronica', 'motown', 'french synthpop', 'future garage', 'canadian contemporary r&b', 'southern soul', 'neo r&b', 'lo-fi beats',  'instrumental hip hop', 'yacht rock', 'acid jazz', 'jazz pop',  'electro-pop francais', 'soft rock', 'trance', 'float house', 'afrofuturism', 'northern soul', 'chamber pop', 'soul jazz', 'retro soul', 'canadian pop', 'pop', 'afropop', 'mellow gold', 'traditional soul', 'soul blues', 'memphis soul', 'shimmer pop']

    unwanted_1 = ['bass house','bossa nova', 'vocal jazz',  'mpb', 'samba', 'brazilian jazz', 'jazz', 'cool jazz', 'samba-jazz',  'jazz guitar', 'bebop', 'violao',  'contemporary vocal jazz',  'latin jazz', 'coverchill', 'tropicalia', 'italian adult pop', 'italian lounge', 'jazz saxophone', 'classic italian pop',  'samba-rock', 'jazz cover', 'bossa nova cover', 'velha guarda', 'jazz blues', 'manguebeat', 'torch song', 'chanson', 'italian jazz', 'psicodelia brasileira',  'smooth jazz', 'musica tradicional cubana', 'adult standards',  'british invasion', 'morna', 'uk contemporary jazz', 'funky house',  'organic electronic',  'melodic house', 'organic house', 'deep disco house',  'tropical house', 'progressive house', 'edm', 'hamburg electronic', 'balearic', 'uk dance', 'house', 'german techno',  'indie napoletano', 'post-disco',  'tech house', 'german house', 'danish electronic', 'lo-fi house', 'disco house', 'uk house', 'beach house',  'deep euro house',  'minimal techno', 'glam rock', 'french pop', 'italo disco']

    wanted_2 = ['indie soul', 'nu disco', 'indietronica', 'french indie pop', 'french indietronica', 'organic electronic', 'new french touch', 'filter house', 'french synthpop', 'bboy', 'jazz house', 'ethnotronica', 'organic house', 'funktronica', 'art pop', 'uk pop', 'electronica', 'chillwave', 'deep house', 'synth funk', 'german techno', 'shimmer pop', 'chill lounge', 'electropop', 'neapolitan funk', 'nu jazz', 'pop soul', 'electro-pop francais', 'brazilian boogie', 'aussietronica', 'balearic', 'deep soul house', 'auckland indie',  'electro jazz', 'funk', 'pop', 'mellow gold',   'new rave', 'downtempo', 'nz pop', 'latintronica',  'neo-synthpop',  'afro-funk']

    unwanted_2 = ['bass house','bossa nova', 'vocal jazz', 'mpb', 'samba', 'brazilian jazz', 'jazz', 'cool jazz', 'samba-jazz', 'jazz guitar', 'bebop', 'violao', 'contemporary vocal jazz', 'latin jazz', 'coverchill', 'tropicalia', 'italian adult pop', 'italian lounge', 'jazz saxophone', 'classic italian pop', 'samba-rock', 'jazz cover', 'bossa nova cover', 'velha guarda', 'jazz blues', 'manguebeat', 'torch song', 'chanson', 'italian jazz', 'psicodelia brasileira', 'smooth jazz', 'musica tradicional cubana', 'adult standards', 'british invasion', 'morna', 'uk contemporary jazz', 'alternative r&b', 'funky house', 'melodic house',  'deep disco house', 'tropical house', 'progressive house', 'edm', 'hamburg electronic', 'house', 'tech house', 'german house', 'danish electronic', 'lo-fi house', 'disco house', 'uk house', 'beach house', 'deep euro house', 'minimal techno', 'glam rock', 'french pop', 'italo disco']

    wanted_3 = ['nu disco', 'indie soul', 'funky house', 'filter house', 'deep house', 'new french touch', 'organic electronic', 'jazz house', 'electronica', 'afrobeat', 'melodic house', 'neapolitan funk', 'french indie pop', 'organic house', 'deep disco house', 'alternative r&b', 'tropical house', 'french indietronica', 'downtempo', 'funktronica', 'progressive house', 'edm', 'hamburg electronic', 'balearic', 'uk dance', 'house', 'german techno', 'rock', 'indie napoletano', 'lgbtq+ hip hop', 'escape room', 'indie jazz', 'post-disco', 'quiet storm', 'vapor soul', 'austindie', 'disco', 'tech house', 'german house', 'danish electronic', 'lo-fi house', 'disco house', 'uk house', 'beach house', 'afropop', 'electro jazz', 'cumbia funk', 'deep euro house', 'nu jazz', 'indietronica', 'mellow gold', 'minimal techno', 'glam rock', 'french pop', 'uk alternative pop', 'raboday', 'art pop', 'piano rock', 'jazztronica', 'italo disco']

    unwanted_3 = ['bossa nova', 'vocal jazz', 'mpb', 'samba', 'brazilian jazz', 'jazz', 'cool jazz', 'samba-jazz', 'jazz guitar', 'bebop', 'violao', 'contemporary vocal jazz', 'latin jazz', 'coverchill', 'tropicalia', 'italian adult pop', 'italian lounge', 'jazz saxophone', 'classic italian pop', 'samba-rock', 'jazz cover', 'bossa nova cover', 'velha guarda', 'jazz blues', 'manguebeat', 'torch song', 'chanson', 'italian jazz', 'psicodelia brasileira', 'smooth jazz', 'musica tradicional cubana', 'adult standards', 'british invasion', 'morna', 'uk contemporary jazz', 'alternative r&b','chillhop', 'soul', 'alternative r&b', 'bboy', 'trip hop', 'british soul', 'downtempo', 'funk', 'electronica', 'nu jazz', 'r&b', 'classic soul', 'french indie pop', 'french indietronica', 'uk contemporary r&b',  'auckland indie', 'urban contemporary', 'electro jazz', 'quiet storm', 'indie r&b', 'jazz funk', 'pop soul', 'souldies', 'sophisti-pop',  'contemporary r&b', 'indietronica', 'indie jazz','bass house']

    for i in range(4):
        df[f'wanted_{i}'] = 0
        df[f'unwanted_{i}'] = 0
    
    # Function to check and update the new columns based on genre
    def update_columns(row):
        # Safely convert string to list
        try:
            genre_list = ast.literal_eval(row['genres'])
        except (ValueError, SyntaxError):
            # Handle the case where the conversion fails
            genre_list = []

        # Update for wanted_0 and unwanted_0
        row['wanted_0'] = 1 if any(genre in genre_list for genre in wanted_0) else 0
        row['unwanted_0'] = 1 if any(genre in genre_list for genre in unwanted_0) else 0

        # Similar updates for wanted_1, unwanted_1, ..., wanted_3, unwanted_3
        row['wanted_1'] = 1 if any(genre in genre_list for genre in wanted_1) else 0
        row['unwanted_1'] = 1 if any(genre in genre_list for genre in unwanted_1) else 0

        row['wanted_2'] = 1 if any(genre in genre_list for genre in wanted_2) else 0
        row['unwanted_2'] = 1 if any(genre in genre_list for genre in unwanted_2) else 0

        row['wanted_3'] = 1 if any(genre in genre_list for genre in wanted_3) else 0
        row['unwanted_3'] = 1 if any(genre in genre_list for genre in unwanted_3) else 0

        return row

    # Apply the function to each row
    df = df.apply(update_columns, axis=1)

    # Drop the original 'genres' column
    df = df.drop('genres', axis=1)
    df = df.drop('time_signature', axis=1)

    # One-hot encoding
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    columns_to_encode = ['mode']

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder()

    # Perform one-hot encoding
    encoded_data = encoder.fit_transform(df[columns_to_encode]).toarray()

    # Creating a DataFrame with the encoded data
    # Use encoder.get_feature_names_out() to get new column names
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))

    # Concatenate the original DataFrame with the new encoded DataFrame
    # Ensure to reset the index to align the rows correctly
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Dropping the original columns that were encoded
    df.drop(columns=columns_to_encode, inplace=True)

    # Min-Max scaling
   # Assuming 'df' is your DataFrame
    # List of columns to be scaled
    columns_to_scale = ["popularity", "duration_ms", "tempo", 'loudness']

    # Initialize the MinMaxScaler
    min_max_scaler = MinMaxScaler()

    # Perform Min-Max scaling on the specified columns
    df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])

    # Select numeric columns for the model input
    model_input = df.select_dtypes(include=['float64', 'int64'])

    # Reshape the input for LSTM model (if required by your model)
    model_input = model_input.values.reshape((model_input.shape[0], 1, model_input.shape[1]))
    print("Shape after encoding:", df.shape)

    return model_input

def main():
    st.title("Spotify Track Classifier")
    description = """
    This app features a song mood and intensity prediction model designed for playlist classification tasks. 
    
    The system employs two models trained on a dataset of 3,089 songs labeled by humans to determine if a song is 
    suitable for a client playlist (a restaurant) and, if so, which playlist it fits based on the mood, along with the model's 
    confidence level in its prediction. 
    
    Model 1 performs binary classification to ascertain song suitability (yes1 or no0), and if suitable,
    Model 2 classifies suitable songs into one of four playlists: Light(0) (bossa nova, soul, etc.), Groovy(1) (RnB, 
    Contemporary, etc.), Upbeat(2) (low tempo Light electronic and groovy sounds), and More Upbeat(3) (mid tempo light nu-disco 
    and light house).
    """
    st.markdown(description)

    # Initialize playlist_id with a default value
    playlist_id = None

    # Spotify Playlist URL input (changed from ID to URL)
    playlist_url = st.text_input("Enter Spotify Playlist URL:")
    if playlist_url:
        playlist_id = get_playlist_id_from_url(playlist_url)  # Extract ID from URL
        if playlist_id:
            pass  # Placeholder, replace with your code that uses playlist_id
            # The rest of your code where you use playlist_id remains the same...
        else:
            st.write("Invalid playlist URL. Please check the URL and try again.")
    else:
        st.write("Please enter a Spotify playlist URL.")


    # Initialize selected_data as an empty DataFrame
    selected_data = pd.DataFrame()
    predictions_made = False  # Flag to check if predictions have been made

    if playlist_id:
        token = get_token()  # Get Spotify token
        data = get_playlist_songs_attributes(token, playlist_id)  # Fetch playlist tracks and their attributes

        if not data.empty:
            st.write("Data Preview:")
            preview_columns = ['name', 'artist', 'album', 'genres', 'release_date', 'tempo', 'spotify_url']
            st.write(data[preview_columns])

            if st.button('Predict'):
                predictions_made = True  # Set flag to True when Predict button is pressed

                # Preprocess data
                preprocessed_data = preprocess_data(data)

                # Predict with first model
                predictions_model1 = model1.predict(preprocessed_data)
                data['Prediction_Model1'] = predictions_model1.flatten()  # Flatten to 1D

                # Filter rows for second model
                indices = predictions_model1.flatten() > 0.5  # Flatten and compare
                selected_data = preprocessed_data[indices]

                if len(selected_data) > 0:
                    predictions_model2 = model2.predict(selected_data)
                    predictions_model2_classes = predictions_model2.argmax(axis=1)
                    predictions_model2_confidence = predictions_model2.max(axis=1)  # Get max probability for confidence

                    # Assign predictions back to the original dataframe
                    data.loc[indices, 'Mood_Definition'] = predictions_model2_classes
                    data.loc[indices, 'Mood_Confidence'] = predictions_model2_confidence  # Add confidence
                else:
                    data['Mood_Definition'] = np.nan
                    data['Mood_Confidence'] = np.nan  # Handle case with no data


    if predictions_made and not data.empty:
        # Select only the specified columns for the final display
        final_columns = preview_columns + ['Prediction_Model1', 'Mood_Definition', 'Mood_Confidence']
        # Ensure the final dataframe contains only the desired columns
        final_data = data.loc[:, final_columns]
        st.dataframe(final_data.style.apply(highlight_rows, axis=1))
    
    



        



if __name__ == "__main__":
    main()







