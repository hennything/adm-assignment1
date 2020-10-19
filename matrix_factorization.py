import pandas as pd 
import numpy as np 
from math import sqrt
from sklearn.model_selection import KFold
import random
import time

np.random.seed(42)

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python', names=["User ID", "Movie ID", "Rating", "Timestamp"])


def x_ij(i, j):
    return np.dot(U[i, :], M.T[:, j])

def e_ij(i, j):
    return X[i, j] - x_ij(i, j)

def update_U(learn_rate, reg, i, j):
    u_hat_ik = U[i] + learn_rate * (2 * (e_ij(i,j)) * M.T[:, j] - reg * U[i, :])
    return u_hat_ik

def update_M(learn_rate, reg, i, j):
    m_hat_kj = M.T[:, j] + learn_rate * (2 * e_ij(i, j) * U[i, :] - reg * M.T[:, j])
    return m_hat_kj

# num_factors=10, num_iter=75, reg=0.05, learn_rate=0.005
def matrix_factorization(train, num_iter=75, reg=0.05, learn_rate=0.005):
        itr = 0
        while itr < num_iter: 
            # randomly permute through the exisitng elements
            for i, j in np.take(train,np.random.rand(train.shape[0]).argsort(),axis=0,out=train):
                U[i, :] = update_U(learn_rate, reg, i, j)
                M.T[:, j] = update_M(learn_rate, reg, i, j)
            itr += 1            

#####
X = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').to_numpy()
m, n = X.shape

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# location of existing ratings in matrix
X_ij = np.argwhere(~np.isnan(X)) 

rmse_scores = []

# startTime = time.time() # testing

for train_ix, test_ix in kf.split(X_ij):
    train, test = X_ij[train_ix], X_ij[test_ix]

    K = 10
    U = np.random.rand(m,K)
    M = np.random.rand(n,K)


    matrix_factorization(train)


    actual = X[test[:,0], test[:,1]]
    pred = np.dot(U, M.T)[test[:,0], test[:,1]] 
    pred[pred > 5] = 5
    pred[pred < 1] = 1

    rmse = sqrt(np.mean((actual - pred)**2))
    rmse_scores.append(rmse)

# print("Total second: {}".format(time.time() - startTime)) # testing
print("Average RMSE: {}".format(round(np.mean(rmse_scores), 3)))