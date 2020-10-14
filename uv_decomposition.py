import pandas as pd 
import numpy as np 
from math import sqrt
from sklearn.model_selection import KFold

##### will set this up to do a test training split
def preprocess(table, train=True):
    table = table.copy()

    uid = table['User ID'].unique().tolist()
    mid = table['Movie ID'].unique().tolist()

    table.fillna(-1, inplace=True)

    return table[['User ID', 'Movie ID', 'Rating']].values

##### initializing U and V such that when the dot product is taken M will contain the global average in each cell
def initialize(train, d):
    n_user = len(np.unique(train[:, 0]))
    n_item = len(np.unique(train[:, 1]))

    glb_avg = np.mean(train[:, 2])

    U = np.full((n_user, d), sqrt(glb_avg/float(d)))
    V = np.full((d, n_item), sqrt(glb_avg/float(d)))

    return U, V

def optimize_elem_u(U, V, u, i):
    print(utility_matrix[u][i] - np.dot(U, V)[u][i])
    # print(U[u][i] * V[i][] )

# def optimize_elem_v(U, V, v, i):

##### this is currently where we are stuck
def optimize(U, V, train, d):
    for i in range(d):
        for u in range(U.shape[0]):
            optimize_elem_u(U, V, u, i)
            # call funciton to update element of U, def optimize_elem_u():
            # print(i, u)
        # for v in range(V.shape[1]):   
            # call funciton to update element of V, def optimize_elem_v():
            # print(i, v)
#####
ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python', names=["User ID", "Movie ID", "Rating", "Timestamp"])
utility_matrix = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').to_numpy()
m, n = utility_matrix.shape
train = preprocess(ratings)

d = 5

U, V = initialize(train, d)

optimize(U, V, train, d)

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
