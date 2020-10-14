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

# this needs to be looking at a single row of the UV matrix and compare it to a single row in the M matrix 
def optimize_elem_u(U, V, u, i):
    iss = M[u] - np.dot(U[u][i], V[i])

    # np.where(~np.isnan())
    # test
    print(np.where(M[u] >= 1))

# def optimize_elem_v(U, V, v, i):

##### this is currently where we are stuck
def optimize(U, V, train, d):
    # in here we still have to insert some sort of stop condition based on weather or not the difference is making significant steps
    for i in range(d):
        for u in range(U.shape[0]):
            optimize_elem_u(U, V, u, i)
            # U[u][i] = 
            # call funciton to update element of U, def optimize_elem_u():
            # print(i, u)
        # for v in range(V.shape[1]):   
            # V[v][i] =     or is it V[i][v]?
            # call funciton to update element of V, def optimize_elem_v():
            # print(i, v)


#####
ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python', names=["User ID", "Movie ID", "Rating", "Timestamp"])
M = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').fillna(-1).to_numpy()
m, n = M.shape

# test training split isnt implemented in preprocess yet, once that is done then we can implement kfold cross validation
train = preprocess(ratings)
# test = preprocess(ratings, False)

# this is the rank of the two uv matrices, I keep using the term rank but dont actually know if its the right term. 
d = 5

U, V = initialize(train, d)

optimize(U, V, train, d)

# P = np.dot(U, V)
