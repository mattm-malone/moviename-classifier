import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

def get_features(text_series):
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)

def data():
    print("loading")
    df = pd.read_csv('data/movie_new.tsv', delimiter='\t', encoding='ISO-8859-1')
    print("Done")

    print("Cleaning Up Data")
    indexes = df[ df['genre'].str.find('Adult') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('News') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Talk-Show') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Game-Show') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Short') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Reality-TV') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Film-Noir') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Sport') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Musical') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Music') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Documentary') != -1 ].index
    df.drop(indexes, inplace=True)
    indexes = df[ df['genre'].str.find('Biography') != -1 ].index
    df.drop(indexes, inplace=True)


    df['genre'] = df['genre'].str.replace('History', 'War')
    df['genre'] = df['genre'].str.replace('War', 'Action')
    df['genre'] = df['genre'].str.replace('Sci-Fi', 'Fantasy')
    df['genre'] = df['genre'].str.replace('Western', 'Action')
    df['genre'] = df['genre'].str.replace('Crime', 'Drama')
    df['genre'] = df['genre'].str.replace('Mystery', 'Thriller')
    df['genre'] = df['genre'].str.replace('Adventure', 'Action')

    ser = pd.Series(df['genre'].str.split(','))
    for index, lis in enumerate(ser):
        ser.iat[index] = pd.unique(lis)[:2]

    df['genre'] = ser
    print("Done")

    genreC = {}
    for genres in df.genre:
        for genre in genres:
            try:
                genreC[genre] = genreC[genre] + 1
            except KeyError:
                genreC[genre] = 0

    num_classes = len(genreC)
    print("transforming")
    mlb = MultiLabelBinarizer()
    mlb.fit(df.genre)
    labels = mlb.classes_
    sentences = df['movie'].values
    y = mlb.transform(df['genre'])
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=2500)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences_train)
    x_train = tokenizer.texts_to_sequences(sentences_train)
    x_test = tokenizer.texts_to_sequences(sentences_test)
    vocab_size = len(tokenizer.word_index) + 1

    x_train = pad_sequences(x_train, maxlen=10)
    x_test = pad_sequences(x_test, maxlen=10)
    print("done")

    countGenres = 0
    for genres in df.genre:
        countGenres = countGenres + len(genres)

    class_weights = {}
    for index, label in enumerate(labels):
        class_weights[index] = countGenres/genreC[label]
        print(label + " " + str(countGenres/genreC[label]))

    return x_train, y_train, x_test, y_test, class_weights, vocab_size

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

def classify_movies_conv1d(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=10))
    model.add(Conv1D({{choice([32, 64, 128])}}, {{choice([4, 5])}}, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(9, activation='relu'))
    model.add(Dense(9, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=5),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='min')
    ]

    result = model.fit(x_train, y_train,
              batch_size=32,
              epochs=20,
              verbose=True,
              validation_data=(x_test, y_test), class_weight=class_weights,
              callbacks = callbacks)
    
    #score = model.evaluate(x_test, y_test, verbose=0)

    validation_acc = np.amax(result.history['val_categorical_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=classify_movies_conv1d,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

X_train, Y_train, X_test, Y_test, weights, vsize = data()

print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)
best_model.save('best_model')