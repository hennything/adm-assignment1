import pandas as pd 
import numpy as np 
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

np.random.seed(42)

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python', names=["User ID", "Movie ID", "Rating", "Timestamp"])

def optimize_u(i, train):
    X_ij = X[i, train[train[:,0] == i][:, 1]]
    num = np.sum(np.dot(X_ij, V[train[train[:,0] == i][:, 1], :]))
    den = np.sum(V[train[train[:,0] == i][:, 1], :] * V[train[train[:,0] == i][:, 1], :])
    # den = np.sum(np.dot(V[train[train[:,0] == i][:, 1], :].T, V[train[train[:,0] == i][:, 1], :]))
    
    return num / den

def optimize_v(j, train):
    X_ij = X[train[train[:,1] == j][:, 0], j]
    num = np.sum(np.dot(X_ij, U[train[train[:,1] == j][:, 0], :]))
    den = np.sum(U[train[train[:,1] == j][:, 0], :] * U[train[train[:,1] == j][:, 0], :])
    # den = np.sum(np.dot(U[train[train[:,1] == j][:, 0], :].T, U[train[train[:,1] == j][:, 0], :]))
    
    return num / den

# implement random
def uv_decomposition(train, rmse):
    itr = 0
    while itr < 10:
        current = rmse
        for i in range(U.shape[0]):
            U[i, :] = optimize_u(i, train)
        for j in range(V.shape[0]):
            V[j, :] = optimize_v(j, train)
        # print("first: ", current)
        print("rmsr: ", itr, sqrt(np.mean((X[train[:,0], train[:,1]] - np.dot(U, V.T)[train[:,0], train[:, 1]])**2)))
        itr += 1

# main, well meant to be 
#####
X = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').to_numpy()
m, n = X.shape

# n splits set to 1 for testing
kf = KFold(n_splits=2, shuffle=True, random_state=42)
k = 2
# array containing the index values for known elements of X, as a side not we could shuffle the rows in this list for random permutations
X_ij = np.argwhere(~np.isnan(X)) 

rmse_scores = []

for train_ix, test_ix in kf.split(X_ij):
    train, test = X_ij[train_ix], X_ij[test_ix]
    
    glb_avg = np.sum(X[train[:,0], train[:,1]]) / X[train[:,0], train[:,1]].shape[0]

    U = np.full((m, k), sqrt(glb_avg/float(k)))
    V = np.full((n, k), sqrt(glb_avg/float(k)))

    rmse = sqrt(np.mean((X[train[:,0], train[:,1]] - np.dot(U, V.T)[train[:,0], train[:, 1]])**2))

    uv_decomposition(train, rmse)
    # print("First rmse: ", rmse)


    # print(U.shape, V.shape)

