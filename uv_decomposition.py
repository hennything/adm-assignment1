import pandas as pd 
import numpy as np 
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# np.random.seed(42)

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python', names=["User ID", "Movie ID", "Rating", "Timestamp"])

def optimize_u(i, train):
    X_ij = X[i, train[train[:,0] == i][:, 1]]
    num = np.sum(np.dot(X_ij, V[train[train[:,0] == i][:, 1], :]))
    den = np.sum(np.dot(V[train[train[:,0] == i][:, 1], :].T, V[train[train[:,0] == i][:, 1], :]))
    
    return num / den

# TODO: implement error catching. 
# idea: if there are any nan present in the list skip and dont update that element... 
# or implementing a better test training split would probably be more efficient
# currently issue only persists here. The problem could however just aswell happening when optimizing u.
def optimize_v(j, train):
    X_ij = X[train[train[:,1] == j][:, 0], j]
    num = np.sum(np.dot(X_ij, U[train[train[:,1] == j][:, 0], :]))
    den = np.sum(np.dot(U[train[train[:,1] == j][:, 0], :].T, U[train[train[:,1] == j][:, 0], :]))
    
    # There must be some missing value because we are dealing with invalid scalars, some for of odd split
    return num / den

# implement stop condition
def uv_decomposition(train):
    # itr = 0
    diff = 1
    current = sqrt(np.mean((X[train[:,0], train[:,1]] - np.dot(U, V.T)[train[:,0], train[:, 1]])**2))
    while diff > 0.0001:
        for i in range(U.shape[0]):
            U[i, :] = optimize_u(i, train)
        
        for j in range(V.shape[0]):
            V[j, :] = optimize_v(j, train)

        previous = current
        current = sqrt(np.mean((X[train[:,0], train[:,1]] - np.dot(U, V.T)[train[:,0], train[:, 1]])**2))
        diff = previous - current
        previous = current

#####
X = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').to_numpy()
m, n = X.shape

kf = KFold(n_splits=5, shuffle=True, random_state=42)
k = 5

# location of existing ratings in matrix
X_ij = np.argwhere(~np.isnan(X)) 

rmse_scores = []

for train_ix, test_ix in kf.split(X_ij):
    train, test = X_ij[train_ix], X_ij[test_ix]
    
    glb_avg = np.sum(X[train[:,0], train[:,1]]) / X[train[:,0], train[:,1]].shape[0]

    U = np.full((m, k), sqrt(glb_avg/float(k)))
    V = np.full((n, k), sqrt(glb_avg/float(k)))

    uv_decomposition(train)

    actual = X[test[:,0], test[:,1]]
    pred =  np.dot(U, V.T)[test[:,0], test[:,1]]
    pred[pred > 5] = 5
    pred[pred < 1] = 1
    pred[np.isnan(pred)] = glb_avg # replacing nan values with the global average, due to fault in test train split

    rmse = sqrt(np.mean((actual - pred)**2))
    rmse_scores.append(rmse)

print("Average RMSE: {}".format(round(np.mean(rmse_scores), 3)))
