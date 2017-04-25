#!/usr/bin/env python

import pandas as pd
from os.path import join

base_dir = "data"
train_file = join(base_dir, "train.csv")
out_file = join(base_dir, "train.out.csv")

df = pd.read_csv(train_file)
drop_indices = [i for i, row in df.iterrows() if row.isnull().drop(['challengeID']).all()]
new_df = df.drop(df.index[drop_indices])
new_df.to_csv(out_file, index=False)

