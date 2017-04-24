import pandas as pd
import numpy as np

def fillMissing(inputcsv, outputcsv):
    
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    
    # replace NA's with mean
    df = df.fillna(df.mean())
  
    # replace negative values with 0
    num = df._get_numeric_data()
    num[num < 0] = 0
    # write filled outputcsv
    df.to_csv(outputcsv, index=False)
    
# Usage:
fillMissing('predictionSGD.csv', 'prediction.csv')
