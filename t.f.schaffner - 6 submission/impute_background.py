#!/usr/bin/env python

import pandas as pd
import numpy as np
from os.path import join

def fillMissing(inputcsv, outputcsv):
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    # replace NA's with mode
    df = df.fillna(df.mode().iloc[0])
    # if still NA, replace with 1
    df = df.fillna(value=1)
    # replace negative values with 1
    num = df._get_numeric_data()
    num[num < 0] = 1
    # write filled outputcsv
    num.to_csv(outputcsv, index=False)
    
base_dir = "data"
background_file = join(base_dir, "background.csv")
output_file = join(base_dir, "background.out.csv")
fillMissing(background_file, output_file)
filleddf = pd.read_csv(output_file, low_memory=False)

