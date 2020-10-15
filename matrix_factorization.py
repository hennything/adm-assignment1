import pandas as pd 
import numpy as np 
from math import sqrt
from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
import random
import time


np.random.seed(42)

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python', names=["User ID", "Movie ID", "Rating", "Timestamp"])


def x_ij(i, j):
    # print("U: ", U.shape, " M.T: ", M.T.shape, "\n")
    return np.dot(U[i, :], M.T[:, j])

def e_ij(i, j):
    # print("e at i, j: ", X[i, j] - x_ij(i, j))
    return X[i, j] - x_ij(i, j)

# u′ik =uik +η·(2eij ·mkj −λ·uik)
def update_U(learn_rate, reg, i, j):
    # print("U at ik: ", U[i, :], "\n lr: ", learn_rate)
    # print("U at ik plus lr: ", U[i, :] + learn_rate)
    u_hat_ik = U[i] + learn_rate * (2 * (e_ij(i,j)) * M.T[:, j] - reg * U[i, :])
    # print(u_hat_ik)
    return u_hat_ik

# m′kj =mkj +η·(2eij ·uik −λ·mkj)
def update_M(learn_rate, reg, i, j):
    # print("M.T at ik: ", M.T[:, j], "\n lr: ", learn_rate)
    # print("M.T at ik plus lr: ", M.T[:, j] + learn_rate)
    m_hat_kj = M.T[:, j] + learn_rate * (2 * e_ij(i, j) * U[i, :] - reg * M.T[:, j])
    # print(m_hat_kj)
    return m_hat_kj

# dont make use of this function just yet, wrote it for testing but im planning on keeping it
# def rmse(train):
#     mse = np.mean((X[train[:,0], train[:,1]] - np.dot(U, M.T)[train[:,0], train[:,1]])**2)

# num_factors=10, num_iter=75, reg=0.05, learn_rate=0.005
def matrix_factorization(train, num_factors=10, num_iter=5, reg=0.05, learn_rate=0.005):
        # count = 0 # test counter
        itr = 0
        while itr < num_iter: 
            for i, j in train:
                err = e_ij(i, j)
                # in the paper they say to calculate the gradient of e_ij**2 but dont actually end up using it for anything... they do use it in the dates section
                err_sqr = err**2
                U[i, :] = update_U(learn_rate, reg, i, j)
                M.T[:, j] = update_M(learn_rate, reg, i, j)
            itr += 1
            # instead of iterating 75 times we could check the change in rmse but it would take too long...
            # or we could calculate the rmse anyways and in the case that we stopped learning then quit but that is probably a waste of computation.
            # rmse(train)
            

# main, well meant to be 
#####
X = ratings.pivot(index='User ID', columns='Movie ID', values='Rating').to_numpy()
m, n = X.shape

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# array containing the index values for known elements of X, as a side not we could shuffle the rows in this list for random permutations
X_ij = np.argwhere(~np.isnan(X)) 

rmse_scores = []

for train_ix, test_ix in kf.split(X_ij):
    train, test = X_ij[train_ix], X_ij[test_ix]

    # k = num_factors ?
    K = 10 
    U = np.random.rand(m,K)
    M = np.random.rand(n,K)

    startTime = time.time()

    matrix_factorization(train)

    executionTime = (time.time() - startTime)

    # rounding elements in the UM.T matrix
    # print(np.dot(U, M.T)[[test[:,0], test[:,1]]])

    UM_T_ = np.dot(U, M.T)[[test[:,0], test[:,1]]] 

    # rounding some numbers 
    UM_T_[np.where(UM_T_ > 5)] = 5
    UM_T_[np.where(UM_T_ < 1)] = 1


    print(executionTime/60)

    rmse = np.mean((X[test[:,0], test[:,1]] - UM_T_)**2)
    print(rmse)
    rmse_scores.append(rmse)
    # print("matrix fact ended with rmse: ", sqrt(rmse))
    # print(mse)

print(np.mean(rmse_scores))
    
    # rmse(U, M, test)
    # print()




