import pandas as pd 
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cost_function(X, Y, B):
    m = len(Y)
    j = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return j

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

rmse_s = []

# still need to fill missing/odd values with the global average
for train_ix, test_ix in kf.split(ratings):
    X_train, X_test = ratings.iloc[train_ix][['Movie ID', 'User ID', 'Rating']], ratings.iloc[test_ix][['Movie ID', 'User ID', 'Rating']]

    glb_avg = X_train['Rating'].mean()

    item_avgs = X_train.groupby('Movie ID')['Rating'].mean().rename("Item Average")
    user_avgs = X_train.groupby('User ID')['Rating'].mean().rename("User Average")

    pred = pd.merge(item_avgs, X_train, how="right", on=["Movie ID"]).fillna(glb_avg)
    pred = pd.merge(user_avgs, pred, how="right", on=["User ID"]).fillna(glb_avg)

    # m, c = np.linalg.lstsq(pred[['Item Average', 'User Average']], pred['Rating'], rcond=None)[0]

    # b = np.mean(pred['Rating']) - m * np.mean(pred['Item Average']) - c * np.mean(pred['User Average'])

    # this can be faster if we start with the inital coefficients and intercept possibly, just an idea.
    m = pred.shape[0]
    x0 = np.ones(m)
    X = np.array([x0, pred['Item Average'], pred['User Average']]).T

    B = np.array([0, 0, 0])
    Y = np.array(pred['Rating'])

    alpha = 0.01

    newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)
    print(newB)
    print(cost_history[-1])

    # print(cost_function(X, Y, B))

#########
#########


#     test_pred = pd.merge(item_avgs, X_test, how="right", on=["Movie ID"]).fillna(glb_avg)
#     test_pred = pd.merge(user_avgs, test_pred, how="right", on=["User ID"]).fillna(glb_avg)

#     test_pred['Predictoon'] = test_pred['Item Average']*m + test_pred['User Average']*c + b

#     rmse = sqrt(mean_squared_error(test_pred['Predictoon'], test_pred['Rating']))
#     rmse_s.append(rmse)

# print(round(np.mean(rmse_s), 3))
