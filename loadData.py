import re, os
import pandas as pd

class Data:
    DATA_HOME = "data/"
    
    def __init__(self, data):
        self.data = data

    def load_data(self):
        
        return pd.read_csv(Data.DATA_HOME + self.data)
        #print(df.sample(n=25).head(n=25))
        
#END Data
