import pandas as pd 
import numpy as np 
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


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
    # print("optimise U:")
    V_temp = V.copy()
    V_cols = V_temp[:, np.where(~np.isnan(M_row))].reshape(d, -1)
    internal_sum = np.dot(U[row_nr, :], V_cols) - V_cols[i, :]
    m_rj = M_row[np.where(~np.isnan(M_row))] #for all j where M has a value
    V_sj = V_cols[i, :] #for all j where M has a value

    return (V_sj * (m_rj - internal_sum)).sum() / (V_sj**2).sum()

def optimise_v(U, V, col_nr, M_col, i):
    # print("optimise V:")
    U_temp = U.copy()
    U_rows = U_temp[np.where(~np.isnan(M_col)), :].reshape(d, -1)
    internal_sum = np.dot(V[:, col_nr], U_rows) - U_rows[i, :]
    M_is = M_col[np.where(~np.isnan(M_col))] #for all i where M has a value
    U_ir = U_rows[i, :] #for all i where M has a value

    return (U_ir * (M_is - internal_sum)).sum() / (U_ir**2).sum()

def rmse(U, V, M):
    # print("RMSE: \n")
    # print("\n")
    # print(sqrt(mean_squared_error(M[np.where(~np.isnan(M))], np.dot(U, V)[np.where(~np.isnan(M))])), "RMSE")
    return sqrt(mean_squared_error(M[np.where(~np.isnan(M))], np.dot(U, V)[np.where(~np.isnan(M))]))

# def fit(U, V, M, d):
#     current = rmse(U, V, M)
#     previous = 0
#     copy_V = V.copy()
#     copy_U = U.copy()
#     diff = 1
#     print(diff, "diff")
#     while diff > 0.0001:
#         copy_U = U.copy()
#         copy_V = V.copy()
#         for i in range(d):
#             for u in range(U.shape[0]):
#                 U[u, i] = optimise_u(U, V, u, M[u], i)
#             for v in range(V.shape[1]):
#                 V[i, v] = optimise_v(U, V, v, M[:, v], i)
#         print(np.dot(U, V))
            
#         previous = current
#         current = rmse(U, V, M)
#         diff = previous - current
#         print(diff, "diff")
#     return copy_U, copy_V

#####
def fit(U, V, M, d):
    current = rmse(U, V, M)
    previous = 0
    diff = 1
    copy_u = 0
    copy_v = 0
    while diff > 0.0001:
        copy_u = U.copy()
        copy_v = V.copy()
        for i in range(d):
            for u in range(U.shape[0]):
                U[u, i] = optimise_u(U, V, u, M[u], i)
            for v in range(V.shape[1]):
                V[i, v] = optimise_v(U, V, v, M[:, v], i)
    
        previous = current
        current = rmse(U, V, M)
        diff = (previous - current)
        # print(diff, "dif")
        print(current)
        # print(np.dot(U, V))
    return copy_u, copy_v
    # print(np.dot(copy_u, copy_v))

#####
ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python', names=["User ID", "Movie ID", "Rating", "Timestamp"])
# M = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').to_numpy()

# print(type(M))
M = np.array([[5, 2, 4, 4, 3], [3, 1, 2, 4, 1], [2, np.nan, 3, 1, 4], [2, 5, 4, 3, 5], [4, 4, 5, 4, np.nan]]).astype(float)

# test training split isnt implemented in preprocess yet, once that is done then we can implement kfold cross validation
train = preprocess(ratings)
# test = preprocess(ratings, False)

# this is the rank of the two uv matrices, I keep using the term rank but dont actually know if its the right term. 
d = 2
U = np.full((5, 2), 1).astype(float)
V = np.full((2, 5) ,1).astype(float)

# U, V = initialize(train, d)
# print(np.dot(U, V ))

# print(U.shape, V.shape)
rmse(U, V, M)
new_u, new_v = fit(U, V, M, d)

print(np.dot(new_u, new_v))
# print(U, V)
# print("")
# print(np.dot(U, V))
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
