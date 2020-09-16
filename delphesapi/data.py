import sys
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import requests
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys
import schedule
import time



class Tweeter:


    def get_data(self):
        tree = ET.parse('/Users/simonpastor/Downloads/europa.xhtml')
        root = tree.getroot()
        columns = ["mep_id","name", "country", "group","nat_group"]
        root_2 = list(list(list(root.getchildren())[1])[0])[0]

        name = []
        country=[]
        political_group = []
        mep_id = []
        nat_group = []
        for mep in root_2.iter():
            if mep.find("fullname") != None:
                name.append(mep.find("fullname").text)
            else:
                pass
            if mep.find("country") != None:
                country.append(mep.find("country").text)
            else:
                pass
            if mep.find("politicalgroup") != None:
                political_group.append(mep.find("politicalgroup").text)
            else:
                pass
            if mep.find("id") != None:
                mep_id.append(mep.find("id").text)
            else:
                pass
            if mep.find("nationalpoliticalgroup") != None:
                nat_group.append(mep.find("nationalpoliticalgroup").text)
            else:
                pass


            return name, country, politicalgroup, mep_id, nat_group


    def fetch_page(self, page):
        response = requests.get(page)
        soup = BeautifulSoup(response.content, "html.parser")
        element = soup.find("body").find("main").find(class_="mt-5 bg-light-gradient border")
        element_1 = element.find_all(class_="row")[1].find(class_="separator separator-dotted separator-1x border-secondary mb-3")
        element_2 = element_1.find(class_="erpl_social-share-horizontal d-flex flex-row align-items-center justify-content-center")
        element_3 = element_2.find_all("a")
        twitter = 0
        for i in element_3:
            if i.text.strip() == "Twitter":
                twitter = [i["href"]][0][20:]
        return twitter


    def final_df(self):
        from data.data import Tweeter
        name, country, politicalgroup, mep_id, nat_group = Tweeter().get_data()
        final = pd.DataFrame(columns = columns)
        final["mep_id"] = mep_id
        final["name"] = name
        final["country"] = country
        final["group"] = political_group
        final["nat_group"] = nat_group
        return final



    def fetch_all(self)
        from data.data import Tweeter
        final = Tweeter().final_df()
        twitt_list = []
        for i in final["mep_id"]:
            x = fetch_page(f"https://www.europarl.europa.eu/meps/en/{i}")
            twitt_list.append(x)
        return twitt_list

    def trial_list(self):
        from data.data import Tweeter
        final = Tweeter().final_df()
        trial = []
        for i in final["twitter"]:
            if i == 0:
                trial.append(0)
            else:
                x = i.replace("?lang=en","").replace("?lang=fr","").replace("?lang=it","").replace("lang=de","").replace("com/","")\
                      .replace("om/","").replace("/","").replace("/status/456143806295855104","")\
                     .replace("?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor","").replace("status456143806295855104","")\
                     .replace("?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor","").replace("?","")\
                    .replace("?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor","")
                trial.append(x)
        return trial


    def final_fun(self):
        from data.data import Tweeter
        trial = Tweeter().trial_list()
        final = Tweeter().final_df()


        zeros = []
        na = []
        for i in trial:
            if i == 0:
                zeros.append(0)
            elif i == "":
                print(trial.index(i))
            elif i == "cchannelUCdnst8BQ86zzBA5_5Pl-FXw":
                print(trial.index(i))
            else:
                pass


        final["twitter"] = trial
        final["twitter"][58] = "LarsPatrickBerg"
        final["twitter"][65] = "AdamBielan"
        final["twitter"][71] = "BenoitBiteau"
        final["twitter"][77] = "gahler_michael"
        final["twitter"][87] = "BotengaM"
        final["twitter"][93] = "saskiabricmont"
        final["twitter"][95] = "Bruna_Annika"
        final["twitter"][100] = "delarabur"
        final["twitter"][105] = "JerzyBuzek"
        final["twitter"][149] = "petervdalen"
        final["twitter"][180] = 0
        final["twitter"][205] = "FidanzaCarlo"
        final["twitter"][235] = "AlexandraGeese"
        final["twitter"][256] = "MarketkaG"
        final["twitter"][258] = "bgroothuis"
        final["twitter"][330] = "ElsiKatainen"
        final["twitter"][336] = "BeataKempa_MEP"
        final["twitter"][356] = "andreykovatchev"
        final["twitter"][389] = "peterliese"
        final["twitter"][394] = "Loekkegaard_MEP"
        final["twitter"][418] = "PedroMarquesMEP"
        final["twitter"][419] = "fulviomartuscie"
        final["twitter"][460] = "HNeumannMEP"
        final["twitter"][465] = 0
        final["twitter"][466] = "VilleNiinisto"
        final["twitter"][474] = "JanOlbrycht"
        final["twitter"][525] = "sirarego"
        final["twitter"][530] = "Frederiqueries"
        final["twitter"][536] = 0
        final["twitter"][577] = "nicosemsrott"
        final["twitter"][605] = "LineaLidell"
        final["twitter"][641] = 0
        final["twitter"][651] = "guyverhofstadt"
        final["twitter"][662] = "AxelVossMdEP"
        final["twitter"][684] = "larawoltersEU"

        return final

    def get_tweets(self):

        from data.data import Tweeter
        final = Tweeter().final_df()
        account_1 = final["twitter"]
        account_1 = list(account_1)
        all_tweets = []
        content = []
        while len(account_1) > 0:
            print(len(account_1))
            if account_1[0] == 0:
                all_tweets.append(0)
                del(account_1[0])
            else:
                content_list = []
                for tweet in Cursor(api.user_timeline,id=account_1[0],tweet_mode='extended').items(300):
                    if "retweeted_status" in dir(tweet):
                        tweet=tweet.retweeted_status.full_text
                        content_list.append(tweet)
                    else:
                        tweet=tweet.full_text
                        content_list.append(tweet)
                content.append(len(content_list))
                all_tweets.append(content_list)
                del(account_1[0])
        return all_tweets



    def final_dataframe(self):
        from data.data import Tweeter
        final = Tweeter().final_df()
        final.content = Tweeter().get_tweets
        data_final = final.drop(labels=drop)
        data_final.to_pickle("final4_clean.csv")

