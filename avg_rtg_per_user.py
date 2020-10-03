import pandas as pd 
import numpy as np 
from math import sqrt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_s = []

for train_ix, test_ix in kf.split(ratings):
    X_train, X_test = ratings.iloc[train_ix][['User ID', 'Rating']], ratings.iloc[test_ix][['User ID', 'Rating']]
    
    glb_avg = X_train['Rating'].mean()

    item_avgs = X_train.groupby('User ID')['Rating'].mean()
    
    pred = pd.merge(item_avgs, X_test, how="right", on=["User ID"]).fillna(glb_avg)

    rmse = sqrt(mean_squared_error(pred['Rating_x'], pred['Rating_y']))
    rmse_s.append(rmse)

print(np.mean(rmse_s))