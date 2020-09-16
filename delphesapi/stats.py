#### Import ###
## pandas 1.1.1 necessary ##

import pandas as pd
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from gensim.models import Word2Vec
from plotly.graph_objects import Scatter3d
import matplotlib.pyplot as plt
import random
from sklearn.metrics.pairwise import cosine_similarity
from delphes.real_data import Delphes


### Merging ###

class Stats:

    def __init__(self):
        """ Appel des fonctions dans real_data """

        self.data = Delphes().get_data()
        self.X = Delphes().df_X(self.data)


    def dico_index_array(self):
        """ creation d'un dictionnaire key: index, value:array """

        dictionary = {}
        for i in self.X["index"]:
            temporary = []
            try :
                for j in range(300):
                    temporary.append(self.X[j][i])
            except KeyError :
                break
            dictionary[i] = np.array(temporary)

        """ NaN deleted et remplacer par des 0 """

        for i,j in dictionary.items():
            dictionary[i] = np.nan_to_num(j)

        return dictionary



    def cosinusx(self, x):
        dictionary = self.dico_index_array()
        distances = {}
        try :
            for i in dictionary.keys():
                    y = dictionary[i]
                    result = cosine_similarity(x.reshape(-1, 1).transpose(), y.reshape(-1,1).transpose())
                    distances[i] = result[0][0]
        except ValueError:
            pass
        return distances

    def max_lists(self, x,top_x):
        distances = self.cosinusx(x)
        maximum_list = []
        maximum_indexes = []
        for i, j in distances.items():
            if len(maximum_list) < top_x:
                maximum_list.append(j)
                maximum_indexes.append(i)
            else:
                if j > min(maximum_list):
                    maximum_list.append(j)
                    del maximum_list[maximum_list.index(min(maximum_list))]
                    maximum_indexes.append(i)
                    del maximum_indexes[maximum_indexes.index(min(maximum_indexes))]
                else:
                    pass
        top100 = dict(zip(maximum_indexes,maximum_list))
        top100_df = pd.DataFrame([top100]).transpose().reset_index()
        return self.data.merge(top100_df,on="index")



    def stats_country(self, x,top_x):
        df = self.max_lists(x,top_x)
        country_stats1=[]
        country_stats2=[]
        init = []
        sum = 0
        for i in df.groupby("country").count().reset_index()["country"].values:
            country_stats1.append(i)
        for i in df.groupby("country").count().reset_index()[["country","index"]]["index"].values:
            init.append(i)
            sum += i
        for i in init:
            country_stats2.append((i/sum)*100)
        country = dict(zip(country_stats1,country_stats2))
        country_viz = pd.DataFrame([country]).transpose().reset_index()
        return country_viz


    def group_encoding(self, values):
        if values == "United Left":
            return (103/255, 0, 0)
        elif values == "Socialists & Democrats":
            return (201/255, 0, 0)
        elif values == 'Greens':
            return (6/255, 137/255, 0)
        elif values == 'Renew Europe':
            return (251/255, 247/255, 45/255)
        elif values == "EPP":
            return (120/255, 119/255, 235/255)
        elif values == 'European Conservatives and Reformists':
            return (4/255, 112/255, 180/255)
        elif values == 'Identity & Democracy':
            return (0, 0, 169/255)
        elif values == 'Non-Attached':
            return (120/255, 119/255, 127/255)


    def name_order(self, df):
        dict_test = {}
        list1 = []
        list2 = []
        order = {}
        for i in df["index"]:
            list1.append(i)
        for j in df[0]:
            list2.append(j)
        unordered = dict(zip(list1,list2))
        try:
            order["United Left"] = unordered['Group of the European United Left - Nordic Green Left']
        except KeyError:
            pass
        try:
            order['Socialists & Democrats'] = unordered['Group of the Progressive Alliance of Socialists and Democrats in the European Parliament']
        except KeyError:
            pass
        try:
            order['Greens'] = unordered['Group of the Greens/European Free Alliance']
        except KeyError:
            pass
        try:
            order['Renew Europe'] = unordered['Renew Europe Group']
        except:
            pass
        try:
            order["EPP"] = unordered["Group of the European People's Party (Christian Democrats)"]
        except KeyError:
            pass
        try:
            order["European Conservatives and Reformists"] = unordered['European Conservatives and Reformists Group']
        except KeyError:
            pass
        try:
            order["Identity & Democracy"] = unordered["Identity and Democracy Group"]
        except KeyError:
            pass
        try:
            order['Non-Attached'] = unordered['Non-attached Members']
        except KeyError:
            pass
        df = pd.DataFrame([order]).transpose().reset_index()
        return df


    def stats_group(self, x,top_x):
        df = self.max_lists(x,top_x)
        country_stats1=[]
        country_stats2=[]
        init=[]
        sum = 0
        for i in df.groupby("group").count()[0].reset_index()["group"].values:
            country_stats1.append(i)
        for i in df.groupby("group").count().reset_index()[["group","index"]]["index"].values:
            init.append(i)
            sum += i
        for i in init:
            country_stats2.append((i/sum)*100)
        group = dict(zip(country_stats1,country_stats2))
        group_viz = pd.DataFrame([group]).transpose().reset_index()
        group_viz = name_order(group_viz)
        group_viz["group_colors"] = group_viz["index"].apply(group_encoding)
        return group_viz


    def stats_sex(self, x,top_x):
        df = self.max_lists(x,top_x)
        sex_stats1=[]
        sex_stats2=[]
        init=[]
        sum = 0
        for i in df.groupby("sex").count()[0].reset_index()["sex"].values:
            sex_stats1.append(i)
        for i in df.groupby("sex").count().reset_index()[["sex","index"]]["index"].values:
            init.append(i)
            sum += i
        for i in init:
            sex_stats2.append((i/sum)*100)
        group = dict(zip(sex_stats1,sex_stats2))
        group["Men"] = group[0.0]
        group["Women"] = group[1.0]
        del group[0]
        del group[1]
        group_viz = pd.DataFrame([group]).transpose().reset_index()
        return group_viz

    def stats_age(self, x,top_x):
        df = self.max_lists(x,top_x)
        age_stats1=[]
        age_stats2=[]
        init=[]
        sum = 0
        for i in df.groupby("age").count()[0].reset_index()["age"].values:
            age_stats1.append(i)
        for i in df.groupby("age").count().reset_index()[["age","index"]]["index"].values:
            init.append(i)
            sum += i
        for i in init:
            age_stats2.append((i/sum)*100)
        group = dict(zip(age_stats1,age_stats2))
        group_viz = pd.DataFrame([group]).transpose().reset_index()
        return group_viz


    def color_generator(self, colors):
        color_list = {}
        color_output = []
        count = 0
        for i in range(0,colors):
            color_list[count] = ((random.random(), random.random(), random.random()))
            count += 1
        for i in color_list.values():
            color_output.append(i)
        return color_output

    def group_color(self, x):
        color_order=[]
        for i in x["group_colors"]:
            color_order.append(i)
        return color_order


    def graphs(self, x,top_x):
        minutes = 0
        mins = top_x/10
        while mins > 0:
            mins = mins - 60
            minutes += 1
        print(f'Estimated Wait Time : {minutes} mins')
        fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(30,20))
        #1st one
        variable = self.stats_country(x,top_x)
        colors = variable["index"].count()
        axs[0,0].bar(variable["index"],variable[0],color=self.color_generator(colors))
        axs[0,0].set_title("Proportion of MEP's Nationalities",fontsize=20)
        #2nd one
        variable = self.stats_group(x,top_x)
        colors = self.group_color(variable)
        axs[0,1].bar(variable["index"],variable[0],color=colors)
        axs[0,1].set_title("Proportion by European Political Group",fontsize=20)
        #3rd one
        variable = self.stats_sex(x,top_x)
        colors = variable["index"].count()
        axs[1,0].bar(variable["index"],variable[0],color=self.color_generator(colors))
        axs[1,0].set_title("Proportion of Men and Women",fontsize=20)
        #4th one
        variable = self.stats_age(x,top_x)
        colors = variable["index"].count()
        axs[1,1].bar(variable["index"],variable[0])
        axs[1,1].set_title("Age Spread",fontsize=20)
        plt.suptitle("Statistics on the Top100 Most Similar Tweets",fontsize=40)
        plt.show()



if __name__ == '__main__':
    Stats().graphs(dictionary[50],100)
