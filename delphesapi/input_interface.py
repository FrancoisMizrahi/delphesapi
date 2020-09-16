from tweepy import OAuthHandler
from googletrans import Translator
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
import re
import os

import pandas as pd
import requests
import diffbot
import string

class InputInterface:

    def __init__(self):
        self.consumer_key = os.getenv("CONSUMER_KEY")
        self.consumer_secret = os.getenv("CONSUMER_SECRET")
        self.access_token_key = os.getenv("ACCESS_TOKEN_KEY")
        self.access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

        self.auth = OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token_key, self.access_token_secret)
        self.api = API(self.auth, wait_on_rate_limit=True)


    def trans_tweet(self, tweet):
        tra = Translator()
        return tra.translate(tweet, dest="en").text

    def rmv_par(self, text):
        text = text.replace("«","")
        text = text.replace("»","")
        text = text.replace('"',"")
        text = text.replace("'","")
        return text

    def rmurl(self, text):
        '''
        This function removes all the URLs, the #hashtag and the @user of a column made of strings.
        Be careful to apply it BEFORE all the other preprocessing steps (if not it wont'
        be recognized as a URL)
        '''
        result = re.sub(r"http\S+|www.\S+|@\S+|#\S+", "", text)
        return result

    def lower(self, text):
        '''
        This function lowercases a column made of strings.
        '''
        text = text.lower()
        return text

    def remove_numbers(self, text):
        return ''.join(word for word in text if not word.isdigit())

    def replace_punct(self, text):
        punct = string.punctuation
        for punctu in punct:
            text = text.replace(punctu, ' ')
            text = text.replace(' rt ','')
            text = " ".join(text.split())
        return text

    def deEmojify(self, inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')

    def delete_paragraph(self, paragraph):
        without_line_breaks = a_string.replace("\n", " ")
        return without_line_breaks

    def preprocessing (self, text):
        text = self.rmv_par(text)
        text = self.rmurl(text)
        text = self.lower(text)
        text = self.remove_numbers(text)
        text = self.replace_punct(text)
        text = self.deEmojify(text)
        return text

    def text_from_url(self, url):
        # url = input("Write a Url: ")
        new_url = f'https://extractorapi.com/api/v1/extractor/?apikey=f6d8320079329a09fedc7c3c59cc581740f904d3&url={url}'
        response = requests.get(new_url)
        text = response.json()["text"]
        text = self.trans_tweet(text)
        text = self.preprocessing(text)
        return text

    def text_from_tweets(self, username):
        all_tweets = []
        content = []
        content_list = []
        # import ipdb
        # ipdb.set_trace()
        for tweet in Cursor(self.api.user_timeline,id=username,tweet_mode='extended').items(10):
            if "retweeted_status" in dir(tweet):
                tweet=tweet.retweeted_status.full_text
                content_list.append(tweet)
            else:
                tweet=tweet.full_text
                content_list.append(tweet)
                content.append(len(content_list))
        all_tweets.append(content_list)
        new_tweets = []
        for i in all_tweets[0]:
            text = self.trans_tweet(i)
            text = self.preprocessing(i)
            new_tweets.append(text)
        return new_tweets

    def simple_text(self, text):
        text = input("Write something: ")
        text = self.delete_paragraph(text)
        text = self.trans_tweet(text)
        text = self.preprocessing(text)
        return text

if __name__ == '__main__':
    print(InputInterface().text_from_url("https://www.politico.eu/article/while-brussels-starts-plenary-strasbourg-suffers-from-lack-of-meps/"))
    print("\n##############################\n")
    print(InputInterface().text_from_tweets("the_simonpastor"))














