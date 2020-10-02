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
    X_train, X_test = ratings.iloc[train_ix].Rating, ratings.iloc[test_ix].Rating
    glb_avg = X_train.mean()
    pred = np.full(X_test.shape[0], glb_avg)
    rmse = sqrt(mean_squared_error(X_test, pred))
    rmse_s.append(rmse)

print(np.mean(rmse_s))