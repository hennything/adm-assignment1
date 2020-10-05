import pandas as pd
import numpy as np

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

user_IDs = ratings['User ID'].unique()
movie_IDs = ratings['Movie ID'].unique()

df = pd.DataFrame(index=user_IDs, columns=movie_IDs)

for index, row in ratings.iterrows():
    current_row = df.iloc[row['User ID']]
    current_row[row['Movie ID']] = row['Rating']

    #print(row['User ID'], row['Movie ID'], row['Rating'])

print(df.head(10))

