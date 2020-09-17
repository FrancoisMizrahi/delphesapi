import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flask import Flask
from flask import request, escape
# from flask_cors import CORS
from gensim.models import Word2Vec
from flask import Response
from tensorflow.keras import models

from delphesapi.input_interface import InputInterface

PATH_TO_MODEL = "delphesapi/data"
DATA_REF_PATH = "raw_data/cleaned_tweet_df"
MODEL_PATH = {"group":"delphesapi/data/model_group",
            "country":"delphesapi/data/model_country",
            "name":"delphesapi/data/model_deputies"}
WORD2VEC_PATH = {"group":"delphesapi/data/word2vec_group",
                "country":"delphesapi/data/word2vec_country",
                "name":"delphesapi/data/word2vec_deputies"}

clean_df = pd.read_pickle("delphesapi/data/clean_df")

app = Flask(__name__)
# CORS(app)

@app.route('/')
def hello():
    #get param from http://127.0.0.1:5000/?name=value
    name = request.args.get("name", "zebi")
    return f'Hello, {escape(name)}!'

@app.route('/get_input', methods=["POST"])
def get_input():
    user_input = request.get_data().decode()
    if user_input[:3] == "tex":
        print(user_input)
        text = []
        entry = user_input.replace("+", " ").replace("text=", "")
        text.append(entry)
    elif user_input[:3] == "url":
        url = user_input.replace("url=", "").replace("%3A", ":").replace("%2F", "/")
        input_interface = InputInterface()
        text = []
        entry = input_interface.text_from_url(url)
        text.append(entry)
    elif user_input[:3] == "twe":
        username = user_input.replace("tweet=", "")
        input_interface = InputInterface()
        text = input_interface.text_from_tweets(username)
    else:
        print("unsupported key")

    feature = 'group'
    model = models.load_model(MODEL_PATH[feature])
    word2vec = Word2Vec.load(WORD2VEC_PATH[feature])
    result = return_result(text, feature, model, word2vec)
    return Response(result, status=201, mimetype='application/json')


def format_input(text, word2vec):
    sentences_inter = []
    for sentence in text:
        sentences_inter.append(sentence.split())
    X_train = embedding(word2vec,sentences_inter)
    formated_input = pad_sequences(X_train, padding='post',value=-1000, dtype='float32')
    return formated_input

def return_result(formated_input, feature, model, word2vec):
    if feature=='country' or feature =='group':
        pond = (clean_df.groupby(feature).count()['content'].values)**0.75
        result = (1000*np.mean(model.predict(format_input(formated_input, word2vec)), axis=0)/pond).argmax()
        return pd.get_dummies(clean_df[feature]).columns[result]
    else:
        result = np.mean(model.predict(format_input(formated_input, word2vec)), axis=0).argmax()
        return pd.get_dummies(clean_df[feature]).columns[result]

# Sentence embedding
def embed_sentence(word2vec, sentence):
    '''
    Embed one sentence.
    '''
    y = []
    for word in sentence:
        if word in word2vec.wv.vocab.keys():
           y.append(word2vec[word])
    return np.array(y)

#Sentence embedding
def embedding(word2vec, sentences):
    '''
    Embed  set of sentences.
    '''
    y = []
    for sentence in sentences:
        y.append(embed_sentence(word2vec, sentence))
    return y

# pipeline_def = {'pipeline': model}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

