import pandas as pd
import numpy as np

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

user_IDs = ratings['User ID'].unique()
movie_IDs = ratings['Movie ID'].unique()

print('There are in total ' + str(len(movie_IDs)) + ' movies in the database')
print('There are in total ' + str(len(user_IDs)) + ' users in the database')

M = pd.DataFrame(index=user_IDs, columns=movie_IDs)

for index, row in ratings.iterrows():
    current_row = M.iloc[row['User ID'] - 1]
    current_row[row['Movie ID']] = row['Rating']

U = pd.DataFrame(np.random.randint(1, 5, size=(len(user_IDs), 2)))
V = pd.DataFrame(np.random.randint(1, 5, size=(2, len(movie_IDs))))
