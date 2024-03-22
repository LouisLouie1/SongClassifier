# SongClassifier
Song mood and intensity prediction model for a playlists classifications tasks.

Internal tool I created for my music curation process. I run music curation business offering bespoke moods under the form of different playlist for restaurants, bars, hotels and retail spaces. These playlists usually follow a precise level of energy and style. They are updated monthly. It takes a demanding and tedious amount of time to listen to songs to analyze if it fits a certain mood. 

As an experiment, I pulled all extra metadata and features of songs I labeled with different moods over the years and trained machine learning models to predict if a song is a good fit for the place, and if so, in what playlist. Trained 2 bidirectional LSTM Models on labeled data :

Model 1 : bidirectional LSTM- Binary classification ( suitable or not for this restaurant ?)

Training data:
  Label 0: All variety of chosen that do not fit any playlist (1300 songs) <br>
  Label 1: All originally labeled songs ( of the 4 classes below)  in playlists ( 1789 songs)

Model 2: bidirectional LSTM- Classification (4 classes) ( if suitable, fits in what playlist ?)

Training data:
  Label 0: Playlist 1 : 454 songs
  Label 1: Playlist 2 : 909 songs
  Label 2: Playlist 3 : 379 songs
  Label 3: Playlist 4 : 191 songs

Song metadata and audio features trained on : 

  release_date: popularity: A numeric score indicating the popularity of the track on Spotify, where higher values represent greater popularity.
  duration_ms: The duration of the track in milliseconds.
  spotify_url: The URL to the track on Spotify.
  genres: A list of genres associated with the artist, as classified by Spotify
  danceability: 
  energy: 
  loudness:.
  mode: The modality of the track Major speechiness:
  acousticness: 
  instrumentalness: track contains no vocal content;
  liveness: audience in the recording
  valence: musical positiveness 
  tempo: The overall estimated tempo of a track in beats per minute (BPM).



