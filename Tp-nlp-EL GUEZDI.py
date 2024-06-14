# EL GUEZDI Mohamed Zinelabidine

# Install and load necessary packages
!pip install kaggle

# Download and unzip the dataset
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
!unzip imdb-dataset-of-50k-movie-reviews.zip

# Import libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the dataset
dataframe = pd.read_csv("IMDB Dataset.csv")

# Preprocess reviews
dataframe["review"] = dataframe["review"].str.replace(r'<[^<>]*>', '', regex=True)  # Remove HTML tags
dataframe["review"] = dataframe["review"].str.replace(r'[^\w\s]', '', regex=True)   # Remove punctuation
dataframe["review"] = dataframe["review"].str.lower()                                # Convert to lowercase

# Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
dataframe["review"] = dataframe["review"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatize words
nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()
dataframe["review"] = dataframe["review"].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Count words that appear at least 5 times
all_text = ' '.join(dataframe['review'])
words = all_text.split()
word_counts = Counter(words)
words_at_least_5 = [word for word, count in word_counts.items() if count >= 5]

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(dataframe["review"])
X = tokenizer.texts_to_sequences(dataframe["review"])
X = pad_sequences(X, maxlen=200, padding='post')

# Encode labels
le = LabelEncoder()
dataframe["sentiment"] = le.fit_transform(dataframe["sentiment"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, dataframe["sentiment"], test_size=0.2, random_state=42)

# Build the LSTM model
model_lstm = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001)

# Compile and train the model
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64, callbacks=[early_stopping, lr_scheduler])