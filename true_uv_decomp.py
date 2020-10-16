import pandas as pd 
import numpy as np 
from math import sqrt
from sklearn.model_selection import KFold
from scipy.sparse.linalg import svds

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

# ##### Matrix
utility_matrix = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').to_numpy()
print(utility_matrix)
m, n = utility_matrix.shape
print(len(utility_matrix))
print(m, n)

##### will set this up to do a test training split
def preprocess(table, train=True):
    table = table.copy()

    uid = table['User ID'].unique().tolist()
    mid = table['Movie ID'].unique().tolist()

    table.fillna(-1, inplace=True)

    return table[['User ID', 'Movie ID', 'Rating']].values

##### initializing U and V such that when the dot product is taken M will contain the global average in each cell
def initialize(train):
    n_user = len(np.unique(train[:, 0]))
    n_item = len(np.unique(train[:, 1]))

    glb_avg = np.mean(train[:, 2])

    U = np.full((n_user, 2), sqrt(glb_avg/float(rank)))
    V = np.full((2, n_item), sqrt(glb_avg/float(rank)))

    # no need to actually return this values
    M = np.dot(U, V)

    return U, V, M

# ##### optimize U and V such that M closely approzimates the non-blank entries in the original matrix
# def optimize(U, V, M):

# this is working with the utility matrix when it should be working with the training matrix
def rmse():
    i, j = np.where(utility_matrix.notna())
    total_actual = 0
    total_train = 0
    M = np.dot(U, V)
    for row, column in zip(i, j):
        total_actual += utility_matrix.iloc[row, column]
        total_train += M[row, column]
    print(total_actual, total_train)

# np.square([-1j, 1], where)
def optimize(U, V, u):
    j = np.where(~np.isnan(utility_matrix))
    print(j)
    # print(U[u])




    # np.diff(u8_arr)

def fit(U, V, rank):
    #  prev, curr
    # diff = prev - curr
    
    for u in range(U.shape[0]):
        optimize(U, V, u)
            # U[u, i] = optimize(U, V, u, i)
        # for v in range(V.shape[0]):
        #     V[v, i] = optimize(U, V, v, i)


#     totalArray=np.zeros((values,values,values,values))
#     for row1 in range(len(matrix[0])):
#         for row2 in range(len(matrix[1])):
#             for row3 in range(len(matrix[2])):
#                 totalArray[row1,row2,row3] = matrix[0][row1]*matrix[1][row2]*matrix[2][row3]

#     print(matrix)            
#     print(totalArray)


train = preprocess(ratings)
test = preprocess(ratings, False)

# glb_avg = np.mean(train[:, 2])

rank =  2

U, V, M = initialize(train)
fit(U, V, 2)
# rmse()


##### code below give the index values which are not null
# i, j = np.where(utility_matrix.notna())

# print(i)
# print("\n")
# print(j)

# for row, column in zip(i, j):
#     print(utility_matrix.iloc[row, column])