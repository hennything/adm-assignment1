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

for train_ix, test_ix in kf.split(ratings):
    X_train, X_test = ratings.iloc[train_ix][['Movie ID', 'User ID', 'Rating']], ratings.iloc[test_ix][['Movie ID', 'User ID', 'Rating']]

    glb_avg = X_train['Rating'].mean()

    item_avgs = X_train.groupby('Movie ID')['Rating'].mean().rename("Item Average")
    user_avgs = X_train.groupby('User ID')['Rating'].mean().rename("User Average")

    pred = pd.merge(item_avgs, X_train, how="right", on=["Movie ID"]).fillna(glb_avg)
    pred = pd.merge(user_avgs, pred, how="right", on=["User ID"]).fillna(glb_avg)

    m, c = np.linalg.lstsq(pred[['Item Average', 'User Average']], pred['Rating'], rcond=None)[0]

