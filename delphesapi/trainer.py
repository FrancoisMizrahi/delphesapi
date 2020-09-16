# Imports nécessaires
import pandas as pd
import numpy as np
import string

# Lecture et stockage de la base de données

from delphes.real_data import Delphes

class Trainer:


    def __init__(self):
        self.data = Delphes().get_data()


    # Remove the undesirable elements in the entire dataframe
    def rmurl_df(self, df, column_name):
        '''
        This function removes all the URLs, the #hashtag and the @user of a column made of strings.
        Be careful to apply it BEFORE all the other preprocessing steps (if not it wont'
        be recognized as a URL)
        '''
        df = df.copy()
        df[column_name] = df[column_name].str.replace('http\S+|www.\S+|@\S+|#\S+', '', case=False)
        return df


    # Lowercase the tweet's column
    def lower_df(self, df, column_name):
        '''
        This function lowercases a column made of strings and return the dataframe.
        '''
        df = self.rmurl_df().copy
        df[column_name] = df[column_name].str.lower()
        return df


    # Remove the numbers in the tweet's column
    def rmnumbers_df(self, df, column_name):
        '''
        This function removes all the digits of a column made of strings.
        '''
        df = df.copy
        def remove_numbers(text):
            return ''.join(word for word in text if not word.isdigit())
        df[column_name] = df[column_name].apply(remove_numbers)
        return df


    # Remove the undesirable punctuations in the tweet's column
    # Remove the undesirable punctuations in the tweet's column
    def rmpunct_df(self, df, column_name):
        '''
        This function removes all the punctuations, all the "rt" and remove multiple spaces
        of a column made of strings.
        '''

        punct = string.punctuation
        df = df.copy()
        def replace_punct(text):
            for punctu in punct:
                text = text.replace(punctu, ' ')
                text = text.replace(' rt ','')
                text = " ".join(text.split())
            return text
        df[column_name] = df[column_name].apply(replace_punct)
        return df

    # Remove the stopwords in the tweet's column
    def rmstopwords_df(self, df, column_name):
        '''
        This function removes all the stopwords of a column made of strings.
        '''
        df = df.copy()
        stop_words = stopwords.words('english')
        def remove_stopwords(text):
            for word in stop_words:
                text = text.replace(f' {word} ', ' ')
            return text
        df[column_name] = df[column_name].apply(remove_stopwords)
        return df


    # Lemmatize a column in a dataset
    def lemmatize_df(self, df, column_name):
        '''
        This function lemmatize the words of a column made of strings.
        '''
        df = df.copy()
        def lemmatize(text):
            lemmatizer = WordNetLemmatizer()
            retour = []
            for word in text:
                retour.append(lemmatizer.lemmatize(word))
            text = ''.join(word for word in retour)
            return text
        df[column_name] = df[column_name].apply(lemmatize)
        return df






    # Erase all the words that are 1-letter or 2-letters long
    def erase_fewletter_df(self, df, column_name):
        '''
        One or two letters words are deleted from the dataset.
        '''
        df = df.copy()
        def tester(text):
            text = ' '.join( [w for w in text.split() if len(w)>2] )
            return text
        df[column_name] = df[column_name].apply(tester)
        return df




    # Remove the undesirable emojis in the entire dataframe
    def rmemojis_df(self, df):
        '''
        This function removes all the emojis of a column made of strings.
        Be careful to translate in latin alphabet before applying this function :
        it also removes cyrillic alphabet.
        '''
        df = df.copy()
        df = df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        return df

    # Sentence embedding
    def embed_sentence(self, word2vec, sentence):
        '''
        Embed one sentence.
        '''
        y = []
        for word in sentence:
            if word in word2vec.wv.vocab.keys():
               y.append(word2vec[word])
        return np.array(y)

    #Sentence embedding
    def embedding(self, word2vec, sentences):
        '''
        Embed  set of sentences.
        '''
        y = []
        for sentence in sentences:
            y.append(embed_sentence(word2vec, sentence))
        return y


    # Cette fonction retourne automatiquement X_train, X_test, y_train, y_test de notre base de données twitter.

    # Cette fonction retourne automatiquement X_train, X_test, y_train, y_test de notre base de données twitter.
    def get_train_test_objects(self, df, column):
        '''
        Les étapes que cette fonction réalise sont en commentaires.
        '''
        # Copie de la base de données pour éviter les problèmes d'assignation abusive.
        df = df.copy()
        # Récupération de tous les tweets et du nom du député qui les a posté. Création de la cible y.
        df = df[[column, 'content']]
        y = pd.get_dummies(df[column])
        # Transformation des tweets en suite de mots (strings) dans une liste.
        sentences = df['content']
        sentences_inter = []
        for sentence in sentences:
            sentences_inter.append(sentence.split())
        # Séparation des données d'entraînement et de test
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences_inter, y, test_size = 0.3)
        # Vectorisation des phrases
        word2vec = Word2Vec(sentences=sentences_train)
        # Création des données d'entrée.
        X_train = embedding(word2vec,sentences_train)
        X_test = embedding(word2vec,sentences_test)
        X_train_pad = pad_sequences(X_train, padding='post',value=-1000, dtype='float32')
        X_test_pad = pad_sequences(X_test, padding='post',value=-1000, dtype='float32')
        # Création des données cibles.
        y_train = y_train.values
        y_test = y_test.values
        # Sorties de la fonction
        return X_train_pad, y_train, X_test_pad, y_test, word2vec


    # Modeling
    def init_model(self):
        '''
        Initialise un modèle avec les couches spécifiées. Le compile pour effectuer une classification multiple.
        '''
        model = Sequential()
        model.add(layers.Masking(mask_value = -1000))
        model.add(layers.LSTM(15, activation='tanh'))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
        return model

    ####################### Prediction ############################
    # Renvoie le député le plus proche de votre tweet
    def predict_deputy(self, df, model, tweet, by_tweet = False):
        '''
        La fonction prend la base de données originale (par député), un modèle entraîné et un texte en entrée.
        Elle renvoie le député le plus proche du texte proposé.
        Attention : le texte en entrée doit être une liste d'au moins deux éléments (strings).
        Quand by_tweet = False, on ressort le député le plus proche de l'ENSEMBLE des tweets.
        Quand by_tweet = True, on sort le député le plus proche POUR CHAQUE tweet.
        '''
        df = self.rmemojis_df.copy()
        tweet_inter = []
        for tw in tweet:
            tweet_inter.append(tw.split())
        X_example = embedding(word2vec,tweet_inter)
        X_example_pad = pad_sequences(X_example, padding='post',value=-1000, dtype='float32')
        prediction = model.predict(X_example_pad)
        if not by_tweet:
            deputy = list(df['name'])[prediction.sum(axis=0).argmax()]
            return deputy
        else:
            deputies_by_tweet = []
            for element in prediction:
                deputies_by_tweet.append(list(df['name'])[element.argmax()])
            return deputies_by_tweet
