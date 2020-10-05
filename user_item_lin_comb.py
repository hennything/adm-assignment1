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

rmse_s = []

# still need to fill missing/odd values with the global average
for train_ix, test_ix in kf.split(ratings):
    X_train, X_test = ratings.iloc[train_ix][['Movie ID', 'User ID', 'Rating']], ratings.iloc[test_ix][['Movie ID', 'User ID', 'Rating']]

    glb_avg = X_train['Rating'].mean()

    item_avgs = X_train.groupby('Movie ID')['Rating'].mean().rename("Item Average")
    user_avgs = X_train.groupby('User ID')['Rating'].mean().rename("User Average")

    pred = pd.merge(item_avgs, X_train, how="right", on=["Movie ID"]).fillna(glb_avg)
    pred = pd.merge(user_avgs, pred, how="right", on=["User ID"]).fillna(glb_avg)

    m, c = np.linalg.lstsq(pred[['Item Average', 'User Average']], pred['Rating'], rcond=None)[0]

    b = np.mean(pred['Rating']) - m * np.mean(pred['Item Average']) - c * np.mean(pred['User Average'])

    # b, m, c, = -2.34631653,  0.87408878,  0.78106055

    test_pred = pd.merge(item_avgs, X_test, how="right", on=["Movie ID"]).fillna(glb_avg)
    test_pred = pd.merge(user_avgs, test_pred, how="right", on=["User ID"]).fillna(glb_avg)

    test_pred['Predictoon'] = test_pred['Item Average']*m + test_pred['User Average']*c + b

    rmse = sqrt(mean_squared_error(test_pred['Predictoon'], test_pred['Rating']))
    rmse_s.append(rmse)

print(round(np.mean(rmse_s), 3))