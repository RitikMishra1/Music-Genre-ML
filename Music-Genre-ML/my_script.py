import os
import numpy as np
import librosa
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report





def scrape_lastfm_dataset(url):
    # Send a GET request to the Last.fm website
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the elements that contain the genre names
    genre_elements = soup.find_all('a', class_='genre-link')
    genres = [genre.text.strip().lower() for genre in genre_elements]
    
    # Find the elements that contain the song names and audio links
    song_elements = soup.find_all('a', class_='chartlist-name')
    songs = [song.text.strip() for song in song_elements]
    audio_links = [song['href'] for song in song_elements]
    
    return genres, songs, audio_links


# Specify the URL of the Last.fm website for web scraping
lastfm_url = 'https://www.last.fm/charts'

# Scrape the Last.fm dataset
genres, songs, audio_links = scrape_lastfm_dataset(lastfm_url)

# Specify the path to the dataset directory
dataset_path = '/Users/ritikmishra/Desktop/Music-Genre-ML'

# Create the dataset directory if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

# Download audio files and save them in the dataset directory
for genre, song, audio_link in zip(genres, songs, audio_links):
    audio_file_path = os.path.join(dataset_path, f'{genre}', f'{song}.mp3')
    os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
    response = requests.get(audio_link)
    with open(audio_file_path, 'wb') as file:
        file.write(response.content)

# Specify the audio features to extract
features = ['chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc']

# Initialize empty lists to store features and labels
X = []
y = []

# Iterate over the genres and extract features
for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    for file in os.listdir(genre_path):
        if file.endswith('.mp3'):
            audio_path = os.path.join(genre_path, file)
            audio, sr = librosa.load(audio_path, duration=30)  # Load audio file
            feature_vector = []
            for feature in features:
                if feature == 'mfcc':
                    feature_vector.extend(np.mean(librosa.feature.mfcc(audio, sr), axis=1))
                else:
                    feature_vector.append(np.mean(getattr(librosa.feature, feature)(audio, sr)))
            X.append(feature_vector)
            y.append(genre)

# Convert features and labels to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y = to_categorical(y, num_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature vectors
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the deep learning model
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping, checkpoint])

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Generate predictions
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Convert label indices back to genre labels
y_pred = label_encoder.inverse_transform(y_pred)
y_true = label_encoder.inverse_transform(y_true)

# Generate classification report
report = classification_report(y_true, y_pred)
print(report)

# Save the label encoder
label_encoder.save('label_encoder.pkl')