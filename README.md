# SongClassifier
Song mood and intensity prediction model for a playlists classifications tasks.

Tool I created ro speed up my music curation process. I run music curation business offering bespoke moods under the form of different playlist for restaurants, bars, hotels and retail spaces. These playlists usually follow a precise level of energy and style. They are updated monthly. It takes a demanding and tedious amount of time to listen to songs to analyze if it fits a certain mood. 

I pulled all extra metadata and features of songs I labeled with different moods over the years and trained two machine learning models to predict (1) if a song is a good fit for the space, and if so, (2) in which one of the four playlists it fits best, and what is the level of confidence of this classification task.  I Trained 2 bidirectional LSTM Models on labeled data :

<a href="https://songclassifier.streamlit.app/" target="_blank">Link to Song Classifier on Streamlit</a>


Model 1 : bidirectional LSTM- Binary classification (Suitable or not for this restaurant ?)

Training data:
  Label 0: All variety of chosen that do not fit any playlist (1300 songs) <br>
  Label 1: All originally labeled songs ( of the 4 classes below)  in playlists ( 1789 songs)

Model 2: bidirectional LSTM- Classification (4 classes) ( if suitable, fits in what playlist ?) + confidence level

Training data:
  Label 0: Playlist 1 : 454 songs
  Label 1: Playlist 2 : 909 songs
  Label 2: Playlist 3 : 379 songs
  Label 3: Playlist 4 : 191 songs

Song metadata and audio features trained pulled from spotify API: 

-danceability: A measure from 0.0 to 1.0 indicating how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A higher value indicates that a track is more danceable.

-energy: A measure from 0.0 to 1.0 that represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale.

-loudness: A measure in decibels (dB) of the overall loudness of a track. It is averaged across the entire track and is useful for comparing the loudness of tracks. Loudness values are typically between -60 and 0 db.

-mode: The modality of the track, indicating the scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
speechiness: A measure from 0.0 to 1.0 indicating the presence of spoken words in a track. A track with high speechiness sounds more like talk. Tracks with a speechiness above 0.66 are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

-acousticness: A measure from 0.0 to 1.0 indicating how acoustic a track is. A score of 1.0 means the track is most likely to be an acoustic track.
instrumentalness: A measure from 0.0 to 1.0 indicating whether a track contains no vocal content. Tracks with a measure close to 1.0 are most likely instrumental.

-liveness: A measure from 0.0 to 1.0 indicating the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

-valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g., happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g., sad, depressed, angry).
![Screen Shot 2024-03-25 at 11 07 28 AM](https://github.com/LouisLouie1/SongClassifier/assets/122399843/996db2ba-0794-441d-9f3d-8697610644cd)










