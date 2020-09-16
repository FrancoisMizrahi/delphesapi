import numpy as np
import pandas as pd

class Delphes:

  def get_data(self):
    """ DataFrame avec l'âge et le sex """

    path_data = './raw_data/ultime_dataframe.h5'
    data = pd.read_hdf(path_data, 'h5')
    return data


  def df_X(self, df):
    """X garde les index"""

    data = self.get_data()
    list_extractor = []
    list_extractor.append("index")
    for i in (range(300)):
        list_extractor.append(i)
    X = data[list_extractor]
    return X
