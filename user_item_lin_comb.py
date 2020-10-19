import pandas as pd 
import numpy as np
from math import sqrt
# import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []

for train_ix, test_ix in kf.split(ratings):
    X_train, X_test = ratings.iloc[train_ix][['Movie ID', 'User ID', 'Rating']], \
                        ratings.iloc[test_ix][['Movie ID', 'User ID', 'Rating']]

    glb_avg = X_train['Rating'].mean()

    item_avgs = X_train.groupby('Movie ID')['Rating'].mean().rename("Item Average")
    user_avgs = X_train.groupby('User ID')['Rating'].mean().rename("User Average")

    pred = pd.merge(item_avgs, X_train, how="right", on=["Movie ID"]).fillna(glb_avg)
    pred = pd.merge(user_avgs, pred, how="right", on=["User ID"]).fillna(glb_avg)

    m, c = np.linalg.lstsq(pred[['Item Average', 'User Average']], pred['Rating'], rcond=None)[0]

    b = np.mean(pred['Rating']) - m * np.mean(pred['Item Average']) - c * np.mean(pred['User Average'])

    # b, m, c, = -2.34631653,  0.87408878,  0.78106055 # optimal values for b, m, c as per gradient descent. 
    
    test_pred = pd.merge(item_avgs, X_test, how="right", on=["Movie ID"]).fillna(glb_avg)
    test_pred = pd.merge(user_avgs, test_pred, how="right", on=["User ID"]).fillna(glb_avg)

    test_pred['Prediction'] = test_pred['Item Average']*m + test_pred['User Average']*c + b
    test_pred[test_pred['Prediction'] > 5] = 5
    test_pred[test_pred['Prediction'] < 1] = 1

    rmse = sqrt(mean_squared_error(test_pred['Prediction'], test_pred['Rating']))
    rmse_scores.append(rmse)

print("Average RMSE: {}".format(round(np.mean(rmse_scores), 3)))


