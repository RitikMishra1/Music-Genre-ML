# Music Genre Classification

This repository contains code for a music genre classification project. The goal of this project is to classify music into different genres using machine learning techniques.

## Webscrapping

Website used for this project can be found at ( https://www.last.fm/ ). Please download and extract the dataset into the `dataset` folder.

## Installation

1. Clone the repository:

git clone https://github.com/RitikMishra1/Music-Genre-ML.git

2. Create a virtual environment and activate it:

python3 -m venv env
source env/bin/activate

3. Install the required dependencies:

pip install -r requirements.txt

## Usage

To train the music genre classifier and evaluate its accuracy, run the following command:

python my_script.py


## License

This project is licensed under the MIT License. See the License.txt file for details.


#What this Code can do ? 

It begins by importing necessary libraries and modules for data handling, web scraping, feature extraction, model building, evaluation, and saving.

The scrape_lastfm_dataset(url) function is defined to scrape the Last.fm website and extract genre names, song names, and audio links. It uses the requests library to send a GET request to the specified URL and the BeautifulSoup library to parse the HTML response and extract relevant information.

The Last.fm website URL is specified, and the scrape_lastfm_dataset() function is called to obtain the genre names, song names, and audio links.

The script creates a directory to store the dataset if it doesn't exist already.

It downloads the audio files corresponding to each genre from the obtained audio links and saves them in the dataset directory. It uses the requests library to download the audio files.

A list of audio features to extract is specified, including chroma_stft, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, and mfcc.

Empty lists are initialized to store the extracted features (X) and corresponding genre labels (y).

The script iterates over the genres and the audio files within each genre. For each audio file, it uses the librosa library to load the audio and extracts the specified features from the audio. The extracted features are then appended to X, and the corresponding genre label is appended to y.

The features (X) and labels (y) are converted to numpy arrays.

The labels (y) are encoded using LabelEncoder, which assigns a numerical label to each unique genre and converts the labels to integers.

The labels (y) are further one-hot encoded using to_categorical from TensorFlow Keras, converting them into binary vectors for multi-class classification.

The dataset is split into training and testing sets using train_test_split from sklearn.model_selection.

The feature vectors in the training and testing sets are normalized using StandardScaler from sklearn.preprocessing.

A deep learning model is built using the Sequential API of TensorFlow Keras. The model architecture consists of dense layers with batch normalization, activation functions, and dropout regularization.

The model is compiled with the Adam optimizer, using categorical cross-entropy as the loss function and accuracy as the metric to optimize.

Callbacks are defined, including early stopping and model checkpointing, to monitor the validation loss and accuracy during training and save the best model.

The model is trained on the training data using the specified batch size, number of epochs, and the defined callbacks. The validation data is used to monitor the model's performance and determine the optimal stopping point.

The trained model is evaluated on the testing data, and the test accuracy is printed.

Predictions are generated on the testing data using the trained model. The predicted label indices are converted back to genre labels using the LabelEncoder.

A classification report is generated using classification_report from sklearn.metrics, showing metrics such as precision, recall, F1-score, and support for each genre.

The LabelEncoder used for encoding genre labels is saved as a pickle file for future use.

This code demonstrates a complete pipeline for music genre classification, from web scraping to feature extraction, model training, evaluation, and saving necessary components.






