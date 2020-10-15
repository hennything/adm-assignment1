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

def optimise_u(U, V, row_nr, M_row, i):
    # print(
    #     "the row:", row, "\n",
    #     "\nrow number: ",row_nr, "\n",
    #     "\nrow from M:",  M_row, "\n",
    #     
    V_temp = V.copy()
    V_cols = V_temp[:, np.where(~np.isnan(M_row))].reshape(2, -1)
    iss = np.dot(U[row_nr, :], V_cols) - V_cols[i, :]
    return (V_cols[i, :] * (M_row[np.where(~np.isnan(M_row))] - iss)).sum() / (V_cols[i, :]**2).sum()

def optimise_v(U, V, col_nr, M_col, i):
    U_temp = U.copy()
    U_rows = U_temp[np.where(~np.isnan(M_col)), :].reshape(2, -1)
    iss = np.dot(V[:, col_nr], U_rows) - U_rows[i, :]
    return (U_rows * (M_col[np.where(~np.isnan(M_col))] - iss)).sum() / (U_rows[i, :]**2).sum()

def rmse(U, V, M):
    # print(np.nansum(M))
    print(np.where(~np.isnan(M))[0].sum())
    print(np.dot(U, V)[np.where(~np.isnan(M))].sum())
    # print(np.where(~np.isnan(M))[0].sum() - np.dot(U, V)[np.where(~np.isnan(M))].sum())
    print(sqrt(np.where(~np.isnan(M))[0].sum() / np.dot(U, V)[np.where(~np.isnan(M))].sum()))


#####
def fit(U, V, M, d):
    print(U.shape, V.shape, M.shape)
    for i in range(d):
        for u in range(U.shape[0]):
            # print(U[u, i])
            U[u, i] = optimise_u(U, V, u, M[u], i)
            # print(U[u, i])
            # if u == 5:
                # break
        for v in range(V.shape[1]):
            # print(V[i, v])
            V[i, v] = optimise_v(U, V, v, M[:, v], i)
            # print(V[i, v])
            # if v == 5:
                # break
    rmse(U, V, M)
    # print(np.dot(U, V))

#####
ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python', names=["User ID", "Movie ID", "Rating", "Timestamp"])
M = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').to_numpy()

print(type(M))
# M = np.array([[5, 2, 4, 4, 3], [3, 1, 2, 4, 1], [2, np.nan, 3, 1, 4], [2, 5, 4, 3, 5], [4, 4, 5, 4, np.nan]]).astype(float)

# test training split isnt implemented in preprocess yet, once that is done then we can implement kfold cross validation
train = preprocess(ratings)
# test = preprocess(ratings, False)

# this is the rank of the two uv matrices, I keep using the term rank but dont actually know if its the right term. 
d = 2
# U = np.full((5, 2), 1).astype(float)
# V = np.full((2, 5) ,1).astype(float)

U, V = initialize(train, d)
# print(np.dot(U, V ))

print(U.shape, V.shape)
rmse(U, V, M)
fit(U, V, M, d)
print("")
# fit(U, V, M, d)
# print("")
# fit(U, V, M, d)
# print("")
# fit(U, V, M, d)
# print("")
# fit(U, V, M, d)
# print("")
# fit(U, V, M, d)
# print("")
# fit(U, V, M, d)
# print("")
# fit(U, V, M, d)
# print("")
# fit(U, V, M, d)
# print("")
# fit(U, V, M, d)


# P = np.dot(U, V)
