import pandas as pd 
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                      names=["User ID", "Movie ID", "Rating", "Timestamp"])

kf = KFold(n_splits=5, shuffle=True)

for train_ix, test_ix in kf.split(ratings):
    X_train, X_test = ratings.iloc[train_ix], ratings.iloc[test_ix]
    y_train, y_test = ratings.iloc[train_ix].Rating, ratings.iloc[test_ix].Rating

    glb_avg = X_train.Rating.mean()

    pred = np.full(len(y_test), glb_avg)

    print(mean_squared_error(y_test, pred))
